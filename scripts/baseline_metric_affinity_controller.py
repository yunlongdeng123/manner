"""Apply CPU affinity once inference has saved the diagnostic predictions."""
import datetime,json,pathlib,subprocess,time
R=pathlib.Path('/root/autodl-tmp/manner-repro');pid=18656
out=pathlib.Path(json.loads((R/'manifests/maptr_baseline_control_started.json').read_text())['directory'])
proc=pathlib.Path('/proc')/str(pid)
assert b'maptr_baseline_control.py' in (proc/'cmdline').read_bytes()
deadline=time.monotonic()+3600
while not (out/'predictions.pkl').exists():
    assert proc.exists(),'Diagnostic process exited before prediction save'
    assert time.monotonic()<deadline,'Timed out waiting for diagnostic predictions'
    time.sleep(15)
subprocess.run(['/root/miniconda3/bin/python',str(R/'scripts/limit_metric_cpu_affinity.py'),'--pid',str(pid),'--cores','12','--record',str(R/'manifests/maptr_baseline_metric_cpu_affinity.json')],check=True)
(R/'manifests/maptr_affinity_controller_done.json').write_text(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=pid,completed=True),indent=2))
