"""Spread our expanded CAN-bus files across both disks without changing paths.

Each copied file is CRC-checked and atomically replaced with a symlink. Existing
open readers remain valid. Original public archives are never modified.
"""
import json,os,pathlib,shutil,time,zlib
R=pathlib.Path('/root/autodl-tmp/manner-repro')
src=R/'data/can_bus';dst=pathlib.Path('/root/manner-repro-download-cache/can_bus')
assert src.is_dir() and not src.is_symlink()
dst.mkdir(parents=True,exist_ok=True)
files=list(src.iterdir());total=0;count=0;last=time.monotonic()
assert shutil.disk_usage(dst).free>sum(p.stat().st_size for p in files if p.is_file() and not p.is_symlink())+2*1024**3
for p in files:
    if p.is_symlink():
        assert p.resolve().parent==dst
        continue
    assert p.is_file(),p
    target=dst/p.name;part=target.with_name(target.name+'.copying')
    crc=0;size=0
    with p.open('rb') as f,part.open('wb') as out:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            crc=zlib.crc32(block,crc);size+=len(block);out.write(block)
    check=0
    with part.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):check=zlib.crc32(block,check)
    assert crc==check and part.stat().st_size==p.stat().st_size==size,p
    part.replace(target)
    link=p.with_name(p.name+'.disklink')
    if link.is_symlink():link.unlink()
    link.symlink_to(target);os.replace(link,p)
    total+=size;count+=1
    if time.monotonic()-last>30:
        print(json.dumps(dict(files=count,relocated_bytes=total)),flush=True);last=time.monotonic()
report=dict(source=str(src),physical_storage=str(dst),files=count,bytes=total,atomic_symlinks=True,crc_verified=True)
(R/'manifests/storage_relocation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
