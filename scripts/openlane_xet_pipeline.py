import os,json,pathlib,time,hashlib,tarfile,shutil,ctypes
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
os.environ.update(HTTPS_PROXY='http://127.0.0.1:37767',HTTP_PROXY='http://127.0.0.1:37767',HF_XET_HIGH_PERFORMANCE='1',HF_XET_NUM_CONCURRENT_RANGE_GETS='8',HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY='1',HF_HUB_DOWNLOAD_TIMEOUT='30',HF_HUB_ETAG_TIMEOUT='20',NO_PROXY='us.aws.cdn.hf.co,transfer.xethub.hf.co,cas-bridge.xethub.hf.co')
from huggingface_hub import hf_hub_download
REPO='AlayaNeW/OpenDriveLab___OpenLane-V2';REV='6c02353d42c777e2aab5a1056fbc80d1cc8fd798'
items={pathlib.Path(x['path']).name:x for x in json.loads((ROOT/'manifests/openlane_hf_raw.json').read_text())}
libc=ctypes.CDLL(None,use_errno=True);libc.fallocate.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong]
def run(name,kind,official_md5):
    ready=ROOT/'manifests'/('verified_'+name+'.json')
    if ready.exists():return
    item=items[name];start=time.monotonic()
    print(json.dumps({'event':'download_start','file':name,'bytes':item['size'],'transport':'HF Xet, eight ranges, CDN direct'}),flush=True)
    path=pathlib.Path(hf_hub_download(repo_id=REPO,repo_type='dataset',filename=item['path'],revision=REV,local_dir=ROOT/'data/openlane_archives'))
    sec=time.monotonic()-start
    print(json.dumps({'event':'download_complete','file':name,'seconds':sec,'MiB_s':path.stat().st_size/max(sec,0.001)/1048576}),flush=True)
    assert path.stat().st_size==item['size']
    sha=hashlib.sha256();md5=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block);md5.update(block)
    assert sha.hexdigest()==item['lfs']['oid'], 'Mirror SHA256 mismatch'
    assert md5.hexdigest()==official_md5, 'Official MD5 mismatch'
    verified={'file':name,'revision':REV,'size':item['size'],'sha256':sha.hexdigest(),'md5':md5.hexdigest()}
    (ROOT/'manifests'/('download_verified_'+name+'.json')).write_text(json.dumps(verified,indent=2))
    print(json.dumps({'event':'archive_hashes_verified',**verified}),flush=True)
    dest=ROOT/'data'/('openlane_smoke' if kind=='sample' else 'openlanev2');dest.mkdir(parents=True,exist_ok=True)
    count=0;written=0;punched=0;last=time.monotonic()
    # This is a newly downloaded, verified, disposable cache file. Reclaim consumed
    # source blocks only after they are extracted, to avoid two simultaneous 22 GB copies.
    reclaim=path.open('r+b') if kind!='sample' else None
    with tarfile.open(path,'r:') as tar:
        for member in tar:
            parts=pathlib.PurePosixPath(member.name).parts
            if member.name.startswith('/') or '..' in parts:raise RuntimeError('Unsafe tar path')
            selected=member.isfile() and not(kind=='info' and ('train' in parts or 'test' in parts))
            if selected:
                if parts and parts[0] in ('OpenLane-V2','OpenLane-V2_sample'):parts=parts[1:]
                target=dest.joinpath(*parts)
                if not target.resolve().is_relative_to(dest.resolve()):raise RuntimeError('Unsafe output')
                target.parent.mkdir(parents=True,exist_ok=True);temp=pathlib.Path(str(target)+'.part')
                with tar.extractfile(member) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
                assert temp.stat().st_size==member.size
                temp.replace(target);count+=1;written+=member.size
            if reclaim:
                safe_end=((member.offset_data+member.size)//(64*1024*1024))*(64*1024*1024)
                if safe_end>punched:
                    result=libc.fallocate(reclaim.fileno(),3,punched,safe_end-punched)
                    if result!=0:raise OSError(ctypes.get_errno(),'Cannot reclaim verified download cache blocks')
                    punched=safe_end
            now=time.monotonic()
            if now-last>30:
                print(json.dumps({'event':'extract_progress','file':name,'files':count,'written':written,'reclaimed':punched}),flush=True);last=now
    if reclaim:reclaim.close();path.unlink()
    verified.update(destination=str(dest),extracted_count=count,extracted_bytes=written,archive_cache_reclaimed=bool(reclaim))
    ready.write_text(json.dumps(verified,indent=2));print(json.dumps({'event':'extracted_verified',**verified}),flush=True)
for name,kind,md5 in [
 ('OpenLane-V2_sample.tar','sample','21c607fa5a1930275b7f1409b25042a0'),
 ('OpenLane-V2_subset_A_info.tar','info','95bf28ccf22583d20434d75800be065d'),
 ('OpenLane-V2_subset_A_image_7.tar','images','c73af4a7aef2692b96e4e00795120504')]:run(name,kind,md5)
