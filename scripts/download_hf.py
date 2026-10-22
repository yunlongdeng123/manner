import concurrent.futures, hashlib, json, os, pathlib, subprocess, time, urllib.request, urllib.error
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
PROXY='http://127.0.0.1:37767'
SOURCES=[('hf_direct','https://huggingface.co',''),('mirror_direct','https://hf-mirror.com',''),('hf_proxy','https://huggingface.co',PROXY)]
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None
def resolve_cdn(url):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY,'https':PROXY}),NoRedirect)
    try:
        opener.open(urllib.request.Request(url+'?download=true',method='HEAD'),timeout=20)
        return url
    except urllib.error.HTTPError as e:
        if e.code in (301,302,303,307,308):return e.headers['Location']
        raise
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def emit(record):
    print(json.dumps(record),flush=True)
def curl_base(proxy):
    return ['curl','--http1.1','-fsSL','--connect-timeout','12','-x',proxy]
def probe(source):
    name,host,proxy=source
    url=host+'/datasets/MatrixF/VectorMapPrior/resolve/main/ckpts/pretrain.ckpt'
    r=subprocess.run(curl_base(proxy)+['--range','0-8388607','--max-time','25','-o','/dev/null','-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
    vals=r.stdout.split()
    speed=float(vals[-1]) if len(vals)==3 and vals[0] in ('200','206') else 0
    emit({'event':'probe','source':name,'returncode':r.returncode,'stats':r.stdout,'error':r.stderr[-150:]})
    return speed,source
def download(job, ranked):
    dataset,item,folder=job
    dest=ROOT/folder/pathlib.Path(item['path']).name
    dest.parent.mkdir(parents=True,exist_ok=True)
    expected=item['lfs']['oid']
    if dest.exists() and dest.stat().st_size==item['size']:
        if sha256(dest)==expected:
            emit({'event':'already_verified','path':str(dest)});return True
    partial=dest.with_suffix(dest.suffix+'.part')
    for _,(name,host,proxy) in ranked:
        url=f'{host}/datasets/{dataset}/resolve/main/{item["path"]}'
        if name=='hf_cdn_direct':url=resolve_cdn(url)
        start=time.monotonic()
        emit({'event':'download_start','path':str(dest),'source':name,'size':item['size']})
        r=subprocess.run(curl_base(proxy)+['--retry','2','--retry-delay','2','--speed-limit','262144','--speed-time','30','--max-time','3600','-C','-','-o',str(partial),'-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
        emit({'event':'download_finish','path':str(dest),'source':name,'returncode':r.returncode,'seconds':round(time.monotonic()-start,1),'stats':r.stdout,'error':r.stderr[-250:]})
        if r.returncode==0 and partial.stat().st_size==item['size']:
            h=hashlib.sha256()
            with partial.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise RuntimeError('SHA256 mismatch '+str(partial))
            partial.rename(dest)
            emit({'event':'verified','path':str(dest),'sha256':h.hexdigest(),'size':dest.stat().st_size})
            return True
    return False
ranked=[(5403447,('hf_cdn_direct','https://huggingface.co','')),(1198824,SOURCES[2]),(819599,SOURCES[1])]
if not ranked:raise RuntimeError('No working source')
jobs=[]
for item in json.loads((ROOT/'manifests/topogpt_hf_ckpts.json').read_text()):
    jobs.append(('MatrixF/VectorMapPrior',item,'weights/topogpt'))
for item in json.loads((ROOT/'manifests/umpe_hf_root.json').read_text()):
    if item['path'].startswith('nuscenes_'):
        jobs.append(('rethinking123/UMPE-Prior',item,'data/umpe_priors'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    ok=list(pool.map(lambda job:download(job,ranked),jobs))
if not all(ok):raise RuntimeError('Incomplete downloads')
