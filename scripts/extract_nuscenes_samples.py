import json,pathlib,tarfile,time,shutil
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro');DATA=ROOT/'data/nuscenes'
wanted=json.loads((ROOT/'manifests/nuscenes_required_files.json').read_text())['files']
archives=sorted(pathlib.Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval').glob('*_blobs.tgz'))
def extract(archive,stop_at_sweeps):
    marker=ROOT/'manifests'/('nuscenes_'+archive.name+('.samples' if stop_at_sweeps else '.full')+'.done.json')
    if marker.exists():return
    start=time.monotonic();count=0;total=0;last=start;offset=0
    print(json.dumps({'event':'scan_start','archive':str(archive),'stop_at_sweeps':stop_at_sweeps}),flush=True)
    with tarfile.open(archive,'r|gz',bufsize=1024*1024) as tar:
        for member in tar:
            name=member.name.removeprefix('./');offset=member.offset
            if stop_at_sweeps and name.startswith('sweeps/'):break
            if name in wanted and member.isfile():
                target=DATA/name
                if not(target.exists() and target.stat().st_size==member.size):
                    target.parent.mkdir(parents=True,exist_ok=True);temp=pathlib.Path(str(target)+'.part')
                    with tar.extractfile(member) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
                    temp.replace(target)
                count+=1;total+=member.size
            now=time.monotonic()
            if now-last>60:
                print(json.dumps({'event':'scan_progress','archive':archive.name,'found':count,'tar_offset':offset,'seconds':round(now-start)}),flush=True);last=now
    record={'source':str(archive),'files':count,'bytes':total,'seconds':round(time.monotonic()-start),'tar_offset':offset,'stop_at_sweeps':stop_at_sweeps}
    marker.write_text(json.dumps(record));print(json.dumps({'event':'scan_done',**record}),flush=True)
for archive in archives:extract(archive,True)
missing=[x for x in wanted if not (DATA/x).exists()]
if missing:
    print(json.dumps({'event':'fallback_full_scan','missing':len(missing)}),flush=True)
    for archive in archives:extract(archive,False)
missing=[x for x in wanted if not (DATA/x).exists()]
(ROOT/'manifests/nuscenes_coverage.json').write_text(json.dumps({'expected':len(wanted),'missing':missing,'val_samples':6019},indent=2))
if missing:raise RuntimeError(f'{len(missing)} files missing')
print('nuScenes validation + smoke input coverage COMPLETE',flush=True)
