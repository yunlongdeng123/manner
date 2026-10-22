import json,pathlib,subprocess,concurrent.futures,time,hashlib
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
P='http://127.0.0.1:37767'
JOBS=[('maptrv2_umpe_ft.pth','1JZXLMF7GfI24JkkD0q1i4dxk3A6r9s_A'),('maptrv2_umpe.pth','1C2tnu9ZFl2_csYVC6Jd62oEVgcwatdAd')]
def run(job):
    name,id=job;dest=ROOT/'weights/umpe'/name
    url='https://drive.usercontent.google.com/download?id='+id+'&export=download&confirm=t'
    for proxy,label in [('', 'direct'),(P,'proxy')]:
        probe=subprocess.run(['curl','--http1.1','-fsSL','-x',proxy,'--connect-timeout','8','--max-time','20','--range','0-8388607','-o','/dev/null','-w','%{http_code} %{content_type} %{size_download} %{speed_download}',url],capture_output=True,text=True)
        print(json.dumps({'event':'probe','file':name,'source':label,'code':probe.returncode,'stats':probe.stdout}),flush=True)
        if not ('application/octet-stream' in probe.stdout or 'application/x-' in probe.stdout):continue
        start=time.monotonic()
        r=subprocess.run(['curl','--http1.1','-fL','-x',proxy,'--connect-timeout','15','--retry','3','--speed-limit','131072','--speed-time','30','--max-time','3600','-C','-','-o',str(dest)+'.part','-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
        print(json.dumps({'event':'download','file':name,'source':label,'code':r.returncode,'seconds':time.monotonic()-start,'stats':r.stdout,'error':r.stderr[-150:]}),flush=True)
        if r.returncode==0:
            part=pathlib.Path(str(dest)+'.part');h=hashlib.sha256()
            with part.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            part.rename(dest)
            print(json.dumps({'event':'saved','path':str(dest),'size':dest.stat().st_size,'sha256':h.hexdigest(),'validation':'local hash; no publisher hash available'}),flush=True)
            return True
    return False
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    if not all(pool.map(run,JOBS)):raise RuntimeError('UMPE checkpoint transfer incomplete')
