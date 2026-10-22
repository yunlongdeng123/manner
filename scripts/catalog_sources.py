import concurrent.futures,json,pathlib,urllib.request,urllib.parse,subprocess
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
P='http://127.0.0.1:37767'
jobs=[
 ('hf_openlane_search.json','https://huggingface.co/api/datasets?search=openlane&limit=50'),
 ('openlane_current_data.md','https://raw.githubusercontent.com/OpenDriveLab/OpenLane-V2/master/data/README.md'),
 ('nuscenes_splits.py','https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/splits.py'),
 ('opendatalab_openlane.html','https://openxlab.org.cn/datasets/OpenDriveLab/OpenLane-V2'),
]
def fetch(job):
    name,url=job
    r=subprocess.run(['curl','-fsSL','-x',P,'--connect-timeout','12','--max-time','45','-o',str(ROOT/'manifests'/name),url],capture_output=True,text=True)
    print(name,r.returncode,r.stderr[-150:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,jobs))
