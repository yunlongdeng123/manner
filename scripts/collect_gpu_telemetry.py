"""Record resource evidence while the authorized validation queue runs."""
import datetime,json,pathlib,subprocess,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def command(args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15).stdout.strip()
with (R/'logs/gpu_telemetry_20261023.jsonl').open('a',buffering=1) as out:
    while True:
        completed=list((R/'results/topogpt').glob('*/metrics.json'))+list((R/'results/umpe').glob('*/predictions.metrics.json'))
        record=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),gpu=command(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader']),apps=command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),memory_bytes=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),completed_metrics=[str(p) for p in completed])
        out.write(json.dumps(record)+'\n')
        if len(completed)>=3 or (R/'manifests/stop_gpu_telemetry').exists():break
        time.sleep(30)
