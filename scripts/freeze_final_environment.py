"""Freeze runtime versions and verify published weights have not changed."""
import datetime,hashlib,json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
M=R/'manifests';out=M/'gpu_runtime';out.mkdir(exist_ok=True)
expected={
 'topogpt/pretrain.ckpt':'1c340ff9077d5449106689488042b04191e2f7844c2f7af1f9807e85f270be37',
 'topogpt/finetune.ckpt':'43cc8e36383793820f3d51ca8d2a244860fb6cb165d88909ab91c8b61345561a',
 'umpe/maptrv2_umpe.pth':'4c4cd40c262ae399595cabe71cf4846e312dd17bb5a78be17cb11ad2fa51d162',
 'umpe/maptrv2_umpe_ft.pth':'f934971dcae0ddb3397c08dca1d1503cc7d936606f5b1126e2b4d2a40238ae4f'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'weights':{}}
for name,h in expected.items():
    p=R/'weights'/name;actual=sha(p);assert actual==h,name
    record['weights'][name]={'bytes':p.stat().st_size,'sha256':actual,'unchanged':True}
for env in ['topogpt','umpe']:
    prefix='/root/miniconda3/envs/'+env
    for label,args in [('pip_freeze',[prefix+'/bin/python','-m','pip','freeze']),('pip_check',[prefix+'/bin/python','-m','pip','check']),('conda_explicit',['/root/miniconda3/bin/conda','list','-p',prefix,'--explicit'])]:
        result=subprocess.run(args,text=True,capture_output=True,check=True)
        (out/(env+'_'+label+'.txt')).write_text(result.stdout+result.stderr)
for label,args in [('cuda_toolkit',['/usr/local/cuda-11.8/bin/nvcc','--version']),('gpu_driver',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),('kernel',['uname','-a'])]:
    result=subprocess.run(args,text=True,capture_output=True,check=True)
    (out/(label+'.txt')).write_text(result.stdout+result.stderr)
(M/'published_weights_unchanged.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
