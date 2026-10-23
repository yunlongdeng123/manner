"""Fetch the official MapTRv2 checkpoint for a bounded diagnostic control."""
import argparse,hashlib,json,pathlib,subprocess,time,zipfile
p=argparse.ArgumentParser();p.add_argument('--proxy',required=True);a=p.parse_args()
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'weights/diagnostic';D.mkdir(exist_ok=True)
dest=D/'maptrv2_nusc_r50_24e.pth';assert not dest.exists()
url='https://drive.usercontent.google.com/download?id=1AmQ3fT-J-MM4B8kh_9Gm2G5guM92Agww&export=download&confirm=t'
probes=[]
for proxy in ['',a.proxy]:
    r=subprocess.run(['curl','--http1.1','-fsSL','-x',proxy,'--connect-timeout','8','--max-time','15','--range','0-8388607','-o','/dev/null','-w','%{http_code} %{content_type} %{size_download} %{speed_download}',url],capture_output=True,text=True)
    row={'proxy':bool(proxy),'exit_code':r.returncode,'stats':r.stdout};probes.append(row);print(json.dumps(row),flush=True)
    if r.returncode==0 and 'application/octet-stream' in r.stdout and float(r.stdout.split()[-1])>262144:
        chosen=proxy;break
else:raise RuntimeError('No acceptably fast valid public route')
start=time.monotonic();part=dest.with_suffix('.part')
r=subprocess.run(['curl','--http1.1','-fL','-x',chosen,'--connect-timeout','15','--retry','3','--speed-limit','262144','--speed-time','30','--max-time','900','-C','-','-o',str(part),'-w','%{http_code} %{size_download} %{speed_download}',url],capture_output=True,text=True)
assert r.returncode==0,(r.returncode,r.stderr[-1000:])
with zipfile.ZipFile(part) as z:assert z.testzip() is None
h=hashlib.sha256()
with part.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
part.rename(dest)
record={'publisher':'https://github.com/hustvl/MapTR/blob/maptrv2/README.md','source':url,'file':str(dest),'bytes':dest.stat().st_size,'sha256':h.hexdigest(),'publisher_hash_available':False,'all_zip_member_crc_passed':True,'seconds':time.monotonic()-start,'curl_stats':r.stdout,'probes':probes,'purpose':'Engineering control to diagnose UMPE mAP gap; no method tuning'}
(R/'manifests/maptr_baseline_download.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
