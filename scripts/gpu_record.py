import datetime,json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def run(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(exit_code=p.returncode,output=p.stdout)
record=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),gpu=run(['nvidia-smi','-q']),cgroup={n:pathlib.Path('/sys/fs/cgroup/'+n).read_text().strip() for n in ['cpu.max','memory.max']},disk=run(['df','-h','/','/root/autodl-tmp']),code=run(['git','-C',str(R),'rev-parse','HEAD']),scope='Published checkpoint validation; separate approximately 10-minute training/save/resume/evaluation engineering smokes. Full retraining not launched.')
p=R/'manifests/gpu_session_20261023.json'
assert not p.exists(),p
p.write_text(json.dumps(record,indent=2));print(str(p))
