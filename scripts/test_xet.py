import os,time,json,pathlib
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
os.environ.update(HTTPS_PROXY='http://127.0.0.1:37767',HTTP_PROXY='http://127.0.0.1:37767',HF_XET_HIGH_PERFORMANCE='1',HF_XET_NUM_CONCURRENT_RANGE_GETS='4',HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY='1',HF_HUB_DOWNLOAD_TIMEOUT='30',HF_HUB_ETAG_TIMEOUT='20')
from huggingface_hub import hf_hub_download
start=time.monotonic()
path=hf_hub_download(repo_id='AlayaNeW/OpenDriveLab___OpenLane-V2',filename='raw/OpenLane-V2_sample.tar',repo_type='dataset',revision='6c02353d42c777e2aab5a1056fbc80d1cc8fd798',local_dir=ROOT/'data/openlane_archives')
elapsed=time.monotonic()-start
print(json.dumps({'path':path,'seconds':elapsed,'MiB_s':pathlib.Path(path).stat().st_size/elapsed/1048576}),flush=True)
