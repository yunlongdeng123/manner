import json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
for name in ['topogpt_full_eval_20261023.log','umpe_full_eval_20261023.log','umpe_base_eval_20261023.log']:
    p=R/'logs'/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2500));s=f.read().decode(errors='replace')
        lines=s.replace('\r','\n').splitlines()
        progress=[x for x in lines if 'task/s' in x or 'Traceback' in x or 'Error' in x or 'Exception' in x]
        phase=[x.strip() for x in lines if x.strip() and 'task/s' not in x]
        parts=progress[-1:]+phase[-2:]
        print(name+': '+' | '.join(x[:240] for x in parts)[-700:])
for pattern in ['topogpt/*/metrics.json','umpe/*/predictions.metrics.json']:
    for p in (R/'results').glob(pattern):print(str(p)+': '+p.read_text())
tel=R/'logs/gpu_telemetry_20261023.jsonl'
if tel.exists():print(tel.read_text().splitlines()[-1])
