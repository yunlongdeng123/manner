import concurrent.futures,json,subprocess,time,urllib.parse,urllib.request
P='http://127.0.0.1:37767'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':P,'https':P}),NoRedirect)
def redirect(url):
    try:
        opener.open(urllib.request.Request(url,method='HEAD'),timeout=15)
        return url
    except urllib.error.HTTPError as e:
        if e.code in (301,302,303,307,308):return e.headers['Location']
        raise
def probe(job):
    name,url,proxy=job
    r=subprocess.run(['curl','-fsSL','-x',proxy,'--connect-timeout','10','--max-time','25','--range','0-33554431','-o','/dev/null','-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
    print(json.dumps({'name':name,'host':urllib.parse.urlparse(url).netloc,'returncode':r.returncode,'stats':r.stdout,'error':r.stderr[-100:]}),flush=True)
jobs=[('pytorch_direct','https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl',''),('hf_mirror_32MB','https://hf-mirror.com/datasets/MatrixF/VectorMapPrior/resolve/main/ckpts/pretrain.ckpt','')]
url=redirect('https://huggingface.co/datasets/MatrixF/VectorMapPrior/resolve/main/ckpts/pretrain.ckpt?download=true')
jobs.append(('hf_cdn_direct',url,''))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(probe,jobs))
