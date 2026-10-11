"""Import required CWQ inputs from a local Full bundle; never import traces."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle-root',type=Path,required=True)
    args=parser.parse_args()
    source=args.bundle_root.resolve();target=ROOT/'assets'
    if target.exists():raise SystemExit('assets already exists; refusing overwrite.')
    if source==ROOT or source.is_relative_to(target):raise SystemExit('Invalid source bundle.')
    required=[source/'data/cwq/semantic_path_train.json',source/'data/cwq/compose_train.json',
              source/'data/cwq/operator_train.json',source/'ontology/fb_roles',
              source/'cache/train_gold_sparql_path_index_v1.json.gz']
    if any(not p.is_file() for p in required):raise SystemExit('Incomplete Full CWQ input bundle.')
    target.mkdir()
    for rel in ('data/cwq','ontology','cache'):
        shutil.copytree(source/rel,target/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log'))
    # Rebind cache provenance paths only when the same TRAIN bytes are present.
    cache=target/'cache/source_verified_template_consensus_index_v1.json.gz'
    if cache.is_file():
        with gzip.open(cache,'rt',encoding='utf-8') as handle:payload=json.load(handle)
        for entry in payload.get('source_manifest',[]):
            old=Path(entry['path']);new=target/'data/cwq'/old.name
            if not new.is_file():raise SystemExit('Missing copied TRAIN source: '+old.name)
            if old.is_file() and hashlib.sha256(old.read_bytes()).digest()!=hashlib.sha256(new.read_bytes()).digest():
                raise SystemExit('TRAIN copy mismatch: '+old.name)
            stat=new.stat();entry.update(path=str(new),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        with gzip.open(cache,'wt',encoding='utf-8') as handle:json.dump(payload,handle,ensure_ascii=False)
    print('CWQ runtime assets imported locally. assets/ is excluded from Git.')

if __name__=='__main__':main()
