"""Relocate an inactive published weight, preserving its original path by symlink."""
import datetime,hashlib,json,os,pathlib,shutil
R=pathlib.Path('/root/autodl-tmp/manner-repro');src=R/'weights/topogpt/pretrain.ckpt'
expected='1c340ff9077d5449106689488042b04191e2f7844c2f7af1f9807e85f270be37'
assert src.is_file() and not src.is_symlink()
D=pathlib.Path('/root/manner-repro-weight-store/topogpt');D.mkdir(parents=True,exist_ok=True)
dst=D/'pretrain.ckpt';tmp=D/'pretrain.ckpt.part';link=src.with_name('pretrain.ckpt.storage-link')
assert not dst.exists() and not tmp.exists() and not link.exists()
assert shutil.disk_usage(D).free>src.stat().st_size+1024**3
for maps in pathlib.Path('/proc').glob('[0-9]*/maps'):
    try:assert str(src) not in maps.read_text(),'Pretrain file is currently memory mapped'
    except (FileNotFoundError,PermissionError):pass
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(src)==expected
shutil.copyfile(src,tmp)
assert sha(tmp)==expected
tmp.rename(dst)
link.symlink_to(dst);os.replace(link,src)
assert src.resolve()==dst and src.stat().st_size==dst.stat().st_size
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),logical_path=str(src),physical_path=str(dst),bytes=dst.stat().st_size,sha256=expected,both_source_and_destination_verified=True,active_finetune_weight_untouched=True)
(R/'manifests/pretrain_weight_storage.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
