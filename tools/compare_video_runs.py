"""Export comparable completed video runs from a durable task ledger."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from video.eval_common import atomic_json, summarize


def export(ledger_path,out_dir):
    ledger=json.loads(Path(ledger_path).read_text(encoding='utf-8'))
    rows=[]
    signatures={}
    for task in ledger['tasks']:
        if task['state']!='complete':
            continue
        result=json.loads((Path(task['output'])/'results.json').read_text(encoding='utf-8'))
        engine=task['spec']['engine']
        key=f"{task['spec']['dataset']}/{engine}"
        for r in result[key]['rows']:
            signature=(r['gt'],r['frame_sha256'])
            if r['case_id'] in signatures and signatures[r['case_id']]!=signature:
                raise ValueError(f"Noncomparable frames/GT: {r['case_id']}")
            signatures[r['case_id']]=signature
            rows.append(dict(engine=engine,dataset=task['spec']['dataset'],video=r['video'],text=r['text'],
                             split=task.get('phase','pilot'),gt=r['gt'],pred=r['pred'],
                             error=r['pred']-r['gt'],abs_error=abs(r['pred']-r['gt']),
                             pred_min_hits3=r.get('pred_min_hits3'),n_frames=r['n_frames'],sec=r['sec'],
                             frame_sha256=r['frame_sha256'],ref=task['ref']))
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'video_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups={}
    for row in rows:
        groups.setdefault(f"{row['split']}/{row['dataset']}/{row['engine']}",[]).append(dict(row,status='complete'))
    # Every group explicitly reports coverage; partial runs are never full benchmarks.
    pending=[dict(ref=t['ref'],state=t['state']) for t in ledger['tasks'] if t['state']!='complete']
    report=dict(phase=ledger['phase'],benchmark_complete=ledger['phase']=='full' and not pending,
                planned_tasks=len(ledger['tasks']),completed_tasks=sum(t['state']=='complete' for t in ledger['tasks']),selection=ledger.get('selection'),pending=pending,
                groups={k:summarize(v,len(v)) for k,v in groups.items()},rows=rows,
                note='Pilot videos are development data. Full benchmark remains incomplete while pending is nonempty.')
    atomic_json(out/'video_summary.json',report)
    print(f'Exported {len(rows)} cases; {len(pending)} tasks outstanding to {out}')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ledger',default='experiments/video_tasks.json')
    p.add_argument('--out-dir',default='outputs/video_benchmark')
    a=p.parse_args()
    export(a.ledger,a.out_dir)
