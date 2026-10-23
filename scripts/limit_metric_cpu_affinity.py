"""Limit one known evaluation process tree to a CPU subset, without code edits."""
import argparse,datetime,json,os,pathlib
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--cores',type=int,default=8);p.add_argument('--record',required=True);a=p.parse_args()
root=pathlib.Path('/proc')/str(a.pid)
cmd=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'tools/test.py' in cmd and 'manner-repro' in cmd,cmd
allowed=sorted(os.sched_getaffinity(a.pid));chosen=allowed[:a.cores]
assert len(chosen)==a.cores>0
parents={}
for d in pathlib.Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
        line=next(x for x in (d/'status').read_text().splitlines() if x.startswith('PPid:'))
        parents[int(d.name)]=int(line.split()[1])
    except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):continue
selected={a.pid}
while True:
    extra={pid for pid,ppid in parents.items() if ppid in selected}
    if extra<=selected:break
    selected|=extra
changes=[]
for pid in sorted(selected):
    task=pathlib.Path('/proc')/str(pid)/'task'
    if not task.exists():continue
    for tid in task.iterdir():
        try:
            before=sorted(os.sched_getaffinity(int(tid.name)))
            os.sched_setaffinity(int(tid.name),chosen)
            changes.append({'pid':pid,'tid':int(tid.name),'before':before})
        except ProcessLookupError:pass
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_pid=a.pid,command=cmd,cpus=chosen,changes=changes,reason='Constrain completed-inference metric stage to leave CPU quota for active GPU jobs; numerical methods unchanged')
pathlib.Path(a.record).write_text(json.dumps(record,indent=2))
print(json.dumps({'root_pid':a.pid,'processes':len(selected),'threads_adjusted':len(changes),'cpus':chosen}))
