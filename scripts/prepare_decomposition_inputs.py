"""Create question/entity-only inference inputs; never inject gold answers or SPARQL."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semantic_guided_kbqa.data import GoldEntityStore

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--gold-entities',type=Path,required=True)
    parser.add_argument('--training-template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Output exists; refusing overwrite.')
    questions=json.loads(args.questions.read_text())
    template=json.loads(args.training_template.read_text())
    instruction=template[0]['instruction']
    links=GoldEntityStore.load(args.gold_entities).mapping()
    rows=[]
    for row in questions:
        q=str(row.get('question','')).strip()
        if not q:raise ValueError('Each CWQ source row needs question.')
        rows.append({'instruction':instruction,'input':json.dumps(
            {'question':q,'golden_entities':links.get(q,{})},ensure_ascii=False)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',len(rows),'question/entity-only inputs.')

if __name__=='__main__':main()
