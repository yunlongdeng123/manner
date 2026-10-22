import concurrent.futures, json, os, pathlib, subprocess, time, urllib.request

ROOT = pathlib.Path('/root/autodl-tmp/manner-repro')
PROXY = 'http://127.0.0.1:37767'
def fetch(job):
    name, url = job
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '15', '--max-time', '120', '-x', PROXY, '-o', str(path), '-w', '%{http_code} %{size_download} %{speed_download}', url], capture_output=True, text=True)
    print(json.dumps({'path':str(path), 'url':url, 'returncode':result.returncode, 'seconds':round(time.monotonic()-start,2), 'transfer':result.stdout, 'error':result.stderr[-300:] if result.returncode else ''}), flush=True)
    return result.returncode

jobs = [
 ('papers/TopoGPT_2606.31814.pdf','https://arxiv.org/pdf/2606.31814'),
 ('papers/UMPE_2605.02762.pdf','https://arxiv.org/pdf/2605.02762'),
 ('manifests/topogpt_hf_ckpts.json','https://huggingface.co/api/datasets/MatrixF/VectorMapPrior/tree/main/ckpts'),
 ('manifests/umpe_hf_root.json','https://huggingface.co/api/datasets/rethinking123/UMPE-Prior/tree/main'),
 ('manifests/umpe_drive.html','https://drive.google.com/drive/folders/1G_OuDVUIDsJc_Ie-TNF0LC6M5CBMyNoK'),
 ('manifests/toponet_data.md','https://raw.githubusercontent.com/OpenDriveLab/TopoNet/main/docs/data.md'),
 ('manifests/umpe_env.md','https://raw.githubusercontent.com/Ethan-Zheng136/UMPE/main/docs/env.md'),
 ('manifests/umpe_dataset.md','https://raw.githubusercontent.com/Ethan-Zheng136/UMPE/main/docs/prepare_dataset.md'),
 ('manifests/umpe_map.md','https://raw.githubusercontent.com/Ethan-Zheng136/UMPE/main/docs/map.md'),
]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, jobs))
