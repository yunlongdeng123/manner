import json, pathlib, subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def run(args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.strip()
print(run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv']))
print(run(['df','-h','/','/root/autodl-tmp']))
print(run(['pgrep','-af','topogpt_smoke|umpe_smoke|train.py|test.py|metric.py|setup.py']))
for name in ['topogpt_gpu_smoke_20261023.log','umpe_cuda_build_20261023.log','umpe_gpu_smoke_20261023.log','topogpt_full_eval_20261023.log','umpe_full_eval_20261023.log']:
    p=R/'logs'/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2200));s=f.read().decode(errors='replace')
        print('\n'+name+':\n'+'\n'.join(s.replace('\r','\n').splitlines()[-9:]))
for p in (R/'manifests').glob('*gpu_smoke_passed.json'):
    print(p.name+': '+p.read_text())
