"""Preserve completed scoring intermediates in a verified archive on system disk."""
import datetime,hashlib,json,pathlib,tarfile,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
T=R/'repos/UMPE/Maptrv2/test/umpe_reproduce_cached'
dirs=['Sat_Oct_24_00_39_04_2026','Sat_Oct_24_00_57_44_2026']
files=[T/d/'pts_bbox'/name for d in dirs for name in ['cls_formatted.pkl','nuscmap_results.json']]
for p in files:assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(T.resolve()),str(p)
for name in ['finetuned_20261023T154019Z','base_parallel_20261023T155704Z']:
    folder=R/'results/umpe'/name
    c=json.loads((folder/'predictions.coverage.json').read_text())
    assert c['full_validation_coverage'] and (folder/'predictions.metrics.json').exists()
def sha_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
record={}
for p in files:
    with p.open('rb') as f:h=sha_stream(f)
    record[str(p.relative_to(T))]=dict(original=str(p),bytes=p.stat().st_size,sha256=h)
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc/'fd').iterdir():
            try:assert fd.resolve() not in files,'Cache is still open: '+str(fd)
            except (FileNotFoundError,PermissionError):pass
    except (FileNotFoundError,PermissionError):pass
D=pathlib.Path('/root/manner-repro-score-cache-archive');D.mkdir(exist_ok=True)
out=D/'umpe_completed_scoring_20261024.tgz';temp=out.with_suffix('.part')
assert not out.exists() and not temp.exists()
start=time.monotonic();print('Compressing completed score caches; originals remain until verification',flush=True)
with tarfile.open(temp,'w:gz',compresslevel=1) as t:
    for p in files:t.add(p,arcname=str(p.relative_to(T)),recursive=False)
with tarfile.open(temp,'r:gz') as t:
    for member in t:
        assert member.isfile() and member.name in record
        with t.extractfile(member) as f:assert sha_stream(f)==record[member.name]['sha256'],member.name
temp.rename(out)
with out.open('rb') as f:archive_sha=sha_stream(f)
for p in files:
    assert p.resolve().is_relative_to(T.resolve())
    p.unlink()
result=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(out),archive_bytes=out.stat().st_size,sha256=archive_sha,files=record,released_data_disk_bytes=sum(x['bytes'] for x in record.values()),seconds=time.monotonic()-start,all_member_hashes_verified=True,original_predictions_metrics_checkpoints_logs_untouched=True,rebuild='scripts/score_umpe_predictions.py can rebuild caches from predictions and ground truth')
(R/'manifests/score_cache_archive.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
