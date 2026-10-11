"""Late, bounded stability repairs from this question's actual stage evidence.

Nothing in this module reads a question index, a baseline file or gold answers.
All models have finished before it runs. Unknown/incomplete evidence leaves
the existing result untouched. At most one new repair query is attempted.
"""
from copy import deepcopy
import re
import time

from .contracts import EntityCandidate, ExecutedGraph, GroundedSemanticCandidate, QueryGraphCandidate
from .evaluation import canonicalize_answer_id
from .lowering import lower_sparql
from .question_query_budget import (
    has_saved_repair_budget, execute_repair_from_saved_budget,
    charge_repair_preparation, remember_kg_label_rows, current_query_budget,
)
from .stability_named_constraint import compile_named_constraint_scope
from .stability_extrema_scope import (
    compile_stability_extrema_from_current_stages, lower_stability_extrema_evidence_query,
    resolve_stability_extrema_rows,
)
from .stability_answer_role import compile_rejected_compose_terminal_restore

_ENTITY = re.compile(r'[mg]\.[A-Za-z0-9_]+')
_NAMED_SHAPE = re.compile(r'^\s*(?:what|which)\b.*\bwith\b.*\bnamed\b.*\bdid\b.*(?:go\s+to|attend)\s*\?\s*$',re.I)
_DATE_PHASE = re.compile(r'\b(?:from|start|end|to)\b.*\b(?:earliest|latest|first|last)\b|\b(?:earliest|latest|first|last)\b.*\b(?:from|start|end|to)\b',re.I)


def _source_execution(question, result, kg):
    raw = result.get('selected_graph')
    ids = result.get('answer_ids')
    traces = result.get('traces')
    if (result.get('failure') or not isinstance(raw,dict) or not isinstance(ids,list)
            or not ids or not isinstance(traces,list) or len(ids)!=len(set(ids))
            or any(not isinstance(v,str) or not _ENTITY.fullmatch(v) for v in ids)
            or (raw.get('provenance') or {}).get('original_question')!=question):
        return None
    try:
        graph=QueryGraphCandidate(raw['graph_id'],deepcopy(raw['triples']),raw['answer_var'],
              deepcopy(raw['operators']),float(raw.get('score',0)),{},deepcopy(raw.get('provenance',{})))
        query=lower_sparql(graph,limit=None)
    except (KeyError,TypeError,ValueError):
        return None
    state=current_query_budget()
    saved=state.cache.get((id(kg),query)) if state is not None and state.question==question else None
    if not isinstance(saved,tuple) or len(saved)!=2 or not isinstance(saved[0],list):return None
    source_rows=saved[0]
    if (len(source_rows)!=len(ids) or any(not isinstance(row,dict) for row in source_rows)
            or {canonicalize_answer_id(row.get(graph.answer_var,'')) for row in source_rows}!=set(ids)):
        return None
    # An ID/signature match is insufficient: prove the unlimited current source
    # query and its entire returned ID count, with the exact final projection.
    for trace in reversed(traces):
        if not isinstance(trace,dict):continue
        output=trace.get('output')
        if trace.get('stage')!='deterministic_sparql_lowering_and_execution' or not isinstance(output,list):
            continue
        for item in output:
            if (isinstance(item,dict) and item.get('graph_id')==graph.graph_id
                    and item.get('sparql')==query and not item.get('error')
                    and type(item.get('row_count')) is int and type(item.get('answer_count')) is int
                    and item.get('row_count')==item.get('answer_count')==len(ids)):
                return ExecutedGraph(graph,list(ids),deepcopy(result.get('answers',[])),item['row_count'])
    return None


def _named_proposal(question, source, traces, linked_context, ontology):
    if not _NAMED_SHAPE.fullmatch(question):return None,None
    semantics=[];diagnostics=None;proposals={}
    for trace in traces:
        stage=trace.get('stage');output=trace.get('output')
        if stage=='semantic_path_generation':
            semantics=[x for x in output if isinstance(x,dict) and x.get('valid') is True] if isinstance(output,list) else []
            diagnostics=None
        elif stage=='complete_semantic_graph_guard' and isinstance(output,dict) and output.get('rejected_count',0):
            # The compact trace does not record the post-filter index mapping.
            semantics=[]
        elif stage=='semantic_subgraph_grounding_and_ranking':diagnostics=output
        elif stage=='grounded_semantic_candidates' and isinstance(output,list) and isinstance(diagnostics,dict):
            entities={}
            for record in diagnostics.get('semantic_graphs',[]):
                for anchor in record.get('anchors',[]):
                    for entity in anchor.get('candidates',[]):
                        entities[(record.get('graph_index'),anchor.get('anchor_id'),entity.get('id'))]=entity
            for raw in output:
                if not isinstance(raw,dict):continue
                provenance=raw.get('provenance') or {};index=provenance.get('semantic_graph_index')
                if type(index) is not int or not 0<=index<len(semantics):continue
                bindings={};valid=True
                for anchor,entity_id in (raw.get('anchor_bindings') or {}).items():
                    evidence=entities.get((index,anchor,entity_id))
                    if not isinstance(evidence,dict) or 'label' not in evidence or 'score' not in evidence:
                        valid=False;break
                    bindings[anchor]=EntityCandidate(entity_id,evidence['label'],evidence['score'])
                if not valid:continue
                grounded=GroundedSemanticCandidate(deepcopy(raw['compose_input']),bindings,
                         deepcopy(raw['relation_bindings']),raw['score'],deepcopy(provenance))
                entry=semantics[index]
                proposal,proof=compile_named_constraint_scope(question,source,entry.get('output'),grounded,
                    linked_context,ontology,decomposition=entry.get('decomposition'),context_question=question)
                if proposal is not None:proposals[proposal.sparql]=(proposal,proof)
    return next(iter(proposals.values())) if len(proposals)==1 else (None,None)


def _with_names_and_sentinel(graph, limit=257):
    query=lower_sparql(graph,limit=None)
    prefix='SELECT DISTINCT ?'+graph.answer_var
    if not query.startswith(prefix+'\nWHERE {') or 'stability_name' in query:
        raise ValueError('Unknown final projection for same-query labels')
    query=query.replace(prefix,prefix+' ?stability_name',1)
    end=query.rfind('\n}')
    if end<0:raise ValueError('Missing source query boundary')
    optional=('\n  OPTIONAL { ?'+graph.answer_var+' <http://rdf.freebase.com/ns/type.object.name> '
              '?stability_name . FILTER(LANG(?stability_name) = "en") }')
    return query[:end]+optional+query[end:]+f'\nLIMIT {limit}'


def _certified_query_names(rows,answer_var):
    name_sets={}
    for row in rows:
        label=row.get('stability_name') if isinstance(row,dict) else None
        if isinstance(label,str) and label.strip():
            identifier=canonicalize_answer_id(row.get(answer_var,''))
            key=' '.join(label.casefold().split())
            name_sets.setdefault(identifier,{}).setdefault(key,set()).add(label)
    names={i:sorted(next(iter(values.values())))[0] for i,values in name_sets.items() if len(values)==1}
    conflicts=sorted(i for i,values in name_sets.items() if len(values)>1)
    return names,conflicts


def _apply_stability_repairs(question,result,*,ontology,kg,linked_context):
    if not has_saved_repair_budget():return result
    # Keep the common path cheap: most questions do not satisfy any closed form.
    actor_possible=isinstance(result.get('rewrite_guard'),dict) and result['rewrite_guard'].get('fallback_used') is True
    if not (_NAMED_SHAPE.fullmatch(question) or _DATE_PHASE.search(question) or actor_possible):return result
    started=time.monotonic();source=_source_execution(question,result,kg)
    if source is None:
        charge_repair_preparation(started);return result
    traces=result['traces'];proposal=None;proof=None;mode=None
    if _NAMED_SHAPE.fullmatch(question):
        proposal,proof=_named_proposal(question,source,traces,linked_context,ontology)
        mode='named_relative'
    if proposal is None and _DATE_PHASE.search(question):
        proposal,proof=compile_stability_extrema_from_current_stages(question,source.graph,traces,ontology,
                        graph_id='stability_temporal_extrema')
        mode='temporal_extrema'
    if proposal is None and actor_possible:
        proposal,proof=compile_rejected_compose_terminal_restore(question,source.graph,traces,ontology,
                        context_question=question,rewrite_guard=result.get('rewrite_guard'))
        mode='declared_terminal_role'
    try:
        query=(lower_stability_extrema_evidence_query(proposal) if mode=='temporal_extrema'
               else _with_names_and_sentinel(proposal)) if proposal is not None else None
    except (ValueError,TypeError,KeyError):query=None
    charge_repair_preparation(started)
    if proposal is None or query is None:return result
    rows,receipt=execute_repair_from_saved_budget(kg,query)
    finalization=time.monotonic()
    audit={'mode':mode,'compile':proof,'execution':receipt,'status':'original_preserved',
           'source_graph_id':source.graph.graph_id,'source_answer_ids':list(source.answer_ids),
           'external_model_calls':0,'sparql':query}
    ids=None
    if rows is not None:
        if mode=='temporal_extrema':
            audit['raw_evidence_rows']=deepcopy(rows)
            audit['answer_filter_mode']='local_extrema_from_complete_same_query_source_record_date_and_type'
            ids,accepted=resolve_stability_extrema_rows(proposal,rows,current_source_ids=source.answer_ids,
                          current_source_row_count=source.row_count,query_result_complete=True,ontology=ontology)
            audit['acceptance']=accepted
        elif 0<len(rows)<257:
            values=[canonicalize_answer_id(r.get(proposal.answer_var,'')) for r in rows]
            if all(_ENTITY.fullmatch(v) for v in values):
                ids=list(dict.fromkeys(values))
                remember_kg_label_rows([{'entity':r.get(proposal.answer_var),'label':r.get('stability_name','')} for r in rows])
    if ids:
        # Rewrite only after a complete positive certificate. Preserve the real
        # query as executed (including evidence filtering for temporal extrema).
        prior_names={a.get('id'):a.get('label') for a in source.answers if isinstance(a,dict)
                     and isinstance(a.get('label'),str) and a['label'].strip()
                     and canonicalize_answer_id(a['label'])!=a.get('id')}
        row_names,conflicts=_certified_query_names(rows,proposal.answer_var)
        audit['same_query_label_conflict_ids']=conflicts
        answers=[{'id':i,'label':prior_names.get(i) or row_names.get(i) or i} for i in ids]
        proposal.sparql=query
        graph={'graph_id':proposal.graph_id,'triples':proposal.triples,'answer_var':proposal.answer_var,
               'operators':proposal.operators,'score':proposal.score,'provenance':proposal.provenance}
        graph['provenance']['stability_executed_query']=query
        if mode=='temporal_extrema':
            graph['provenance']['stability_local_extrema_from_complete_evidence']=audit['acceptance']
        result=dict(result,answer_ids=ids,answers=answers,selected_graph=graph)
        audit.update(status='applied',answer_ids=ids,answer_count=len(ids),raw_row_count=len(rows))
    traces.append({'stage':'late_stability_scope_repair','input':{'question':question},'output':audit})
    charge_repair_preparation(finalization)
    return result


def apply_stability_repairs(question,result,*,ontology,kg,linked_context):
    """Optional repair failures must never turn an existing success into error."""
    state=current_query_budget()
    charged_before=state.repair_seconds if state is not None else 0.0
    started=time.monotonic()
    try:
        return _apply_stability_repairs(question,result,ontology=ontology,kg=kg,linked_context=linked_context)
    except Exception as exc:
        if isinstance(result,dict) and isinstance(result.get('traces'),list):
            result['traces'].append({'stage':'late_stability_scope_repair','input':{'question':question},
                'output':{'status':'original_preserved','reason':'optional_stability_evidence_unavailable',
                          'error':f'{type(exc).__name__}: {str(exc)[:240]}','external_model_calls':0}})
        return result
    finally:
        if state is not None:
            elapsed=time.monotonic()-started
            already_charged=state.repair_seconds-charged_before
            unbilled=max(0.0,elapsed-already_charged)
            state.repair_seconds+=unbilled
            state.repair_preparation_seconds+=unbilled
