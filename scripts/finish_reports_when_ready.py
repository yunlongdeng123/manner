"""Finish reporting after the existing evaluation; never starts an experiment."""
import datetime,json,os,pathlib,subprocess,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
target=R/'results/topogpt/run_20261023T152334Z/metrics.json'
parent=pathlib.Path('/proc/4645')
while not target.exists():
    if not parent.exists():
        raise RuntimeError('Existing TopoGPT evaluation stopped without metrics; inspect original log')
    time.sleep(20)
# The writer closes metrics immediately before exiting. Wait for the original
# shell as well, so final evidence is collected only after the evaluation exits.
while parent.exists():time.sleep(2)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
steps=[
    ('/root/miniconda3/bin/python','build_reproduction_report.py'),
    ('/root/miniconda3/envs/topogpt/bin/python','plot_reproduction_examples.py'),
    ('/root/miniconda3/bin/python','finalize_reproduction_status.py'),
    ('/root/miniconda3/bin/python','validate_delivery_ready.py'),
]
for python,script in steps:
    print('START',script,flush=True)
    subprocess.run([python,str(R/'scripts'/script)],cwd=R,env=env,check=True)
    print('DONE',script,flush=True)
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),reporting_complete=True,visual_review_pending=True,local_retrieval_pending=True,shutdown_executed=False)
(R/'manifests/final_reporting_complete.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
