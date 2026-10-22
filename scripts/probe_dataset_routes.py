import concurrent.futures,json,pathlib,subprocess,urllib.request,urllib.error,urllib.parse
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro');P='http://127.0.0.1:37767'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None
def cdn(url):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':P}),NoRedirect)
    try:opener.open(urllib.request.Request(url,method='HEAD'),timeout=20)
    except urllib.error.HTTPError as e:
        if e.code in (301,302,303,307,308):return e.headers['Location']
        raise
jobs=[]
for repo,path in [('AlayaNeW/OpenDriveLab___OpenLane-V2','raw/OpenLane-V2_subset_A_image_7.tar'),('rethinking123/UMPE-Prior','nuscenes_satellite_map.zip')]:
    orig=f'https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true'
    resolved=cdn(orig)
    for label,url,proxy in [('cdn_direct',resolved,''),('cdn_proxy',resolved,P),('hf_proxy',orig,P),('mirror_direct',orig.replace('huggingface.co','hf-mirror.com'),'')]:
        jobs.append((path,label,url,proxy))
def probe(j):
    path,label,url,proxy=j
    r=subprocess.run(['curl','--http1.1','-fsSL','-x',proxy,'--connect-timeout','10','--max-time','30','--range','0-16777215','-o','/dev/null','-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
    out={'path':path,'route':label,'host':urllib.parse.urlparse(url).netloc,'code':r.returncode,'stats':r.stdout}
    print(json.dumps(out),flush=True);return out
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(probe,jobs))
(ROOT/'manifests/dataset_route_probes.json').write_text(json.dumps(results,indent=2))
