"""Durable Kaggle queue: build -> submit -> monitor -> fetch -> verify results.

Credentials are re-read from kaggle/*.json each cycle and never stored in the ledger.
Use --tick for one cycle or --watch to poll every 60 seconds.
"""
import argparse
from contextlib import contextmanager
import os
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from video.eval_common import atomic_json
from launchers.kaggle_video_launch import CONFIGS, prepare

DEFAULT_LEDGER = ROOT/'experiments/video_tasks.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(path, username, data_root, phase='pilot'):
    if path.exists():
        raise FileExistsError(f'Preserving existing task ledger: {path}')
    tasks = []
    if phase == 'pilot':
        selections = [('penguins',['penguins_1']), ('mot20',['MOT20-01'])]
        engines = ['baseline','sam3_native','countyolo_fusion','countyolo_old','countyolo_exnoise','countyolo_kf5']
    else:
        from video.eval_common import DATASETS
        selections = []
        for ds,(sub,annotation,_) in DATASETS.items():
            gt = json.loads((Path(data_root)/sub/'anno'/annotation).read_text(encoding='utf-8'))
            selections.extend((ds,[v]) for v in gt)
        engines = ['baseline','sam3_native','countyolo_fusion']
    for ds,videos in selections:
        for engine in engines:
            job = prepare(username,engine,ds,videos)
            tasks.append(dict(**job, state='prepared', attempts=0, created_at=now()))
    ledger = dict(schema_version=1, phase=phase, created_at=now(), updated_at=now(),
                  purpose='VideoCount unique-object counting, text-only; fixed CountVid frame sampling',
                  note='Pilot videos are development data. Do not tune thresholds on full evaluation videos.',
                  tasks=tasks, account_errors={})
    atomic_json(path,ledger)
    print(f'Prepared {len(tasks)} tasks in {path}')


def status_name(status):
    return str(getattr(status,'name',status)).rsplit('.',1)[-1].lower()


def safe_error(e):
    code = getattr(getattr(e,'response',None),'status_code',None)
    # Exception strings can include signed URLs or authentication details.
    return f'{type(e).__name__}' + (f' HTTP {code}' if code else '')


def verify_output(task, output):
    output = Path(output)
    status = json.loads((output/'job_status.json').read_text(encoding='utf-8'))
    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
    key = f"{task['spec']['dataset']}/{task['spec']['engine']}"
    r = result[key]
    if status['state'] != 'complete' or not r['complete'] or r['n'] != r['expected'] or r['failed']:
        raise ValueError('Remote evaluation incomplete')
    expected_videos = set(task['spec']['videos'])
    actual_videos = {row['video'] for row in r['rows']}
    if expected_videos and actual_videos != expected_videos:
        raise ValueError('Video selection mismatch')
    if task['spec']['engine'].startswith('countyolo'):
        manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
        expected=hashlib.sha256((ROOT/task['spec']['checkpoint']).read_bytes()).hexdigest()
        if manifest.get('checkpoint_sha256')!=expected:
            raise ValueError('Kaggle checkpoint differs from the declared local checkpoint')
    if task['spec'].get('countgd_checkpoint_sha256'):
        manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('countgd_checkpoint_sha256')!=task['spec']['countgd_checkpoint_sha256']:
            raise ValueError('CountGD-Box checkpoint differs from declared checksum')
    task['metrics'] = {k:r[k] for k in ('mae','rmse','bias','wape','fps','n','expected')}
    task['output'] = str(output)


def _tick(path, max_active=1):
    from kaggle_watch import accounts, api_for, fetch, fetch_log
    ledger = json.loads(path.read_text(encoding='utf-8'))
    keys = accounts()
    ledger['account_errors'] = {}
    ledger.pop('dispatch_hold',None)
    owners = sorted({t['ref'].split('/')[0] for t in ledger['tasks'] if t['state'] not in ('complete','failed')})
    for owner in owners:
        if owner not in keys:
            ledger['account_errors'][owner]='missing credentials'
            continue
        try:
            api = api_for(owner,keys[owner])
            # Validate credentials before any mutation or spending quota.
            api.kernels_list(mine=True,page_size=1)
        except Exception as e:
            ledger['account_errors'][owner]=safe_error(e)
            print(owner,ledger['account_errors'][owner],flush=True)
            continue
        owned=[t for t in ledger['tasks'] if t['ref'].startswith(owner+'/')]
        for task in owned:
            if task['state'] not in ('submitted','sending','submission_unknown','fetch_pending'):
                continue
            try:
                state=status_name(api.kernels_status(task['ref']).status)
                task['remote_state']=state
                task['checked_at']=now()
                if 'running' in state:
                    try:
                        task['live_log'], _ = fetch_log(task['ref'])
                    except Exception as e:
                        task['log_error']=safe_error(e)
                if 'complete' in state:
                    task['state']='fetch_pending'
                    atomic_json(path,ledger)
                    folder=fetch(task['ref'])
                    verify_output(task,folder)
                    task['state']='complete'
                    task['finished_at']=now()
                elif any(s in state for s in ('error','cancel','failed')):
                    task['state']='failed'
                    task['error']=f'Kaggle {state}'
                    try:
                        task['output']=str(fetch(task['ref']))
                    except Exception as e:
                        task['fetch_error']=safe_error(e)
                    if task.get('retry_current_source_once'):
                        archive=ROOT/'outputs/kaggle_archive'/(task['ref'].replace('/','__')+'__attempt'+str(task['attempts']))
                        if task.get('output'):
                            shutil.copytree(task['output'],archive,dirs_exist_ok=True)
                        archive.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(Path(task['folder'])/'run.py',archive/'packaged_run.py')
                        task.setdefault('history',[]).append(dict(ref=task['ref'],state='failed',output=str(archive),
                            code_sha256=task['code_sha256'],at=now(),reason=task.pop('retry_current_source_once')))
                        task.update(prepare(owner,task['spec']['engine'],task['spec']['dataset'],task['spec']['videos']))
                        task['state']='prepared'
                        for field in ('error','output','remote_state','last_error'):
                            task.pop(field,None)
                        print('Requeued once with repaired evaluator:',task['ref'],flush=True)
                print(task['ref'],task['state'],state,flush=True)
            except Exception as e:
                task['last_error']=safe_error(e)
                if task['state']=='fetch_pending':
                    task['fetch_failures']=task.get('fetch_failures',0)+1
                    if task['fetch_failures']>=3:
                        task.update(state='failed',error='Output download or verification failed three times')
                print(task['ref'],task['last_error'],flush=True)
        if any(t['state']=='failed' for t in owned):
            ledger['dispatch_hold']='A task failed; inspect logs before submitting additional jobs.'
            continue
        active=sum(t['state'] in ('submitted','sending','submission_unknown','fetch_pending') for t in owned)
        for task in owned:
            if active>=max_active:
                break
            if task['state']!='prepared':
                continue
            folder=Path(task['folder'])
            actual=hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
            if actual!=task['code_sha256']:
                task.update(state='failed',error='Packaged source changed after task preparation')
                continue
            task.update(state='sending',attempts=task['attempts']+1,submitted_at=now())
            atomic_json(path,ledger)
            try:
                result=api.kernels_push(str(folder))
                error=getattr(result,'error',None)
                if error:
                    if 'session count' in str(error).lower() or 'quota' in str(error).lower():
                        task.update(state='prepared',last_error=str(error))
                        break
                    task.update(state='failed',error=str(error))
                    break
                else:
                    task['state']='submitted'
                    active+=1
                    print('Submitted',task['ref'],flush=True)
            except Exception as e:
                # The request may have reached Kaggle: check by ref next cycle before resubmitting.
                code=getattr(getattr(e,'response',None),'status_code',None)
                state='failed' if code is not None and 400 <= code < 500 else 'submission_unknown'
                task.update(state=state,last_error=safe_error(e))
                if state!='failed':
                    active+=1
                else:
                    break
            atomic_json(path,ledger)
    ledger['updated_at']=now()
    atomic_json(path,ledger)
    counts={state:sum(t['state']==state for t in ledger['tasks']) for state in sorted({t['state'] for t in ledger['tasks']})}
    print(json.dumps(dict(states=counts,account_errors=ledger['account_errors']),ensure_ascii=False),flush=True)
    return ledger



class QueueBusy(RuntimeError):
    pass


@contextmanager
def queue_lock(path):
    """OS lock is released on process exit, including a crashed watcher."""
    lock=Path(str(path)+'.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+b') as f:
        if f.tell()==0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                raise QueueBusy('Another process is updating this queue') from None
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise QueueBusy('Another process is updating this queue') from None
        try:
            yield
        finally:
            f.seek(0)
            if os.name=='nt':
                msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def tick(path,max_active=2):
    with queue_lock(path):
        return _tick(path,max_active)


def expand_full(path,data_root):
    """Select CountYOLO configuration on pilot only, then evaluate remaining videos."""
    with queue_lock(path):
        ledger=json.loads(path.read_text(encoding='utf-8'))
        if ledger.get('phase')!='pilot' or not all(t['state']=='complete' for t in ledger['tasks']):
            return False
        from video.eval_common import DATASETS
        groups={}
        case_signatures={}
        for t in ledger['tasks']:
            results=json.loads((Path(t['output'])/'results.json').read_text(encoding='utf-8'))
            r=results[f"{t['spec']['dataset']}/{t['spec']['engine']}"]
            for row in r['rows']:
                signature=(row['gt'],row['frame_sha256'])
                if row['case_id'] in case_signatures and case_signatures[row['case_id']]!=signature:
                    raise ValueError('Pilot frame sampling or ground truth differs across engines')
                case_signatures[row['case_id']]=signature
            groups.setdefault(t['spec']['engine'],[]).extend(r['rows'])
        scores={}
        for engine,rows in groups.items():
            if engine.startswith('countyolo'):
                if {r['case_id'] for r in rows}!=set(case_signatures):
                    raise ValueError('Incomplete CountYOLO pilot comparison')
                scores[engine]=sum(abs(r['pred']-r['gt']) for r in rows)/len(rows)
        winner=min(scores,key=scores.get)
        ledger['selection']=dict(criterion='lowest MAE on pilot videos only',scores=scores,engine=winner,selected_at=now())
        engines=['baseline','sam3_native',winner]
        owner=ledger['tasks'][0]['ref'].split('/')[0]
        existing={(t['spec']['engine'],t['spec']['dataset'],tuple(t['spec']['videos'])):t for t in ledger['tasks']}
        for ds,(sub,annotation,_) in DATASETS.items():
            gt=json.loads((Path(data_root)/sub/'anno'/annotation).read_text(encoding='utf-8'))
            for video in gt:
                for engine in engines:
                    key=(engine,ds,(video,))
                    if key in existing:
                        existing[key]['include_in_full']=True
                        continue
                    job=prepare(ledger.get('account_routes',{}).get(engine,owner),engine,ds,[video])
                    ledger['tasks'].append(dict(**job,state='prepared',attempts=0,created_at=now(),phase='heldout',include_in_full=True))
        ledger['phase']='full'
        ledger['updated_at']=now()
        atomic_json(path,ledger)
        print('Selected on pilot:',winner,scores,flush=True)
        return True



def refresh_prepared(path):
    with queue_lock(path):
        ledger=json.loads(path.read_text(encoding='utf-8'))
        n=0
        for task in ledger['tasks']:
            if task['state']=='prepared':
                task.update(prepare(task['ref'].split('/')[0],task['spec']['engine'],
                                    task['spec']['dataset'],task['spec']['videos'],
                                    out_dir=task['folder'], sam3_dataset=task['spec'].get('sam3_dataset'),
                                    data_dataset=task['spec'].get('data_dataset')))
                n+=1
        atomic_json(path,ledger)
        print('Refreshed',n,'unsubmitted packages')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    p.add_argument('--init',action='store_true')
    p.add_argument('--refresh-prepared',action='store_true')
    p.add_argument('--username',default='nadthien')
    p.add_argument('--data-root',default=str(ROOT/'data/VideoCount'))
    p.add_argument('--phase',choices=['pilot','full'],default='pilot')
    p.add_argument('--tick',action='store_true')
    p.add_argument('--watch',action='store_true')
    p.add_argument('--full-after-pilot',action='store_true')
    p.add_argument('--interval',type=int,default=60)
    p.add_argument('--max-active',type=int,default=2)
    a=p.parse_args()
    if a.interval<30 or a.max_active<1:
        p.error('interval >=30 and max-active >=1 required')
    if a.init:
        initialize(a.ledger,a.username,a.data_root,a.phase)
    if a.refresh_prepared:
        refresh_prepared(a.ledger)
    if a.tick or a.watch:
        while True:
            try:
                ledger=tick(a.ledger,a.max_active)
            except QueueBusy as e:
                if not a.watch:
                    raise SystemExit(str(e))
                print(str(e),flush=True)
                time.sleep(a.interval)
                continue
            from compare_video_runs import export
            report_dir=ROOT/'outputs/video_benchmark' if a.ledger.resolve()==DEFAULT_LEDGER.resolve() else ROOT/'outputs'/a.ledger.stem
            export(a.ledger,report_dir)
            try:
                expanded=a.full_after_pilot and expand_full(a.ledger,a.data_root)
            except QueueBusy:
                expanded=False
            if expanded:
                continue
            if not a.watch or all(t['state'] in ('complete','failed') for t in ledger['tasks']):
                break
            if ledger.get('dispatch_hold') and not any(t['state'] in ('submitted','fetch_pending','submission_unknown','sending') for t in ledger['tasks']):
                break
            time.sleep(a.interval)


if __name__=='__main__':
    main()
