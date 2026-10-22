import ast,ijson,json,pathlib,subprocess,tarfile,time,zipfile,shutil
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro'); DATA=ROOT/'data/nuscenes'; META=DATA/'v1.0-trainval'
def records(name):
    with (META/(name+'.json')).open('rb') as f:yield from ijson.items(f,'item')
tree=ast.parse((ROOT/'manifests/nuscenes_splits.py').read_text())
splits={}
for node in tree.body:
    if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ('train','val','train_detect','train_track'):
        try:splits[node.target.id]=ast.literal_eval(node.value)
        except Exception:pass
    if isinstance(node,ast.Assign):
        for t in node.targets:
            if isinstance(t,ast.Name) and t.id in ('train','val','train_detect','train_track'):
                try:splits[t.id]=ast.literal_eval(node.value)
                except Exception:pass
assert len(splits['val'])==150
val_names=set(splits['val']);scenes=list(records('scene'))
val_scenes={x['token'] for x in scenes if x['name'] in val_names}
smoke_scene=next(x for x in scenes if x['name'] not in val_names)
sample_tokens={};val_count=0
for x in records('sample'):
    if x['scene_token'] in val_scenes:
        sample_tokens[x['token']]='val';val_count+=1
    elif x['scene_token']==smoke_scene['token']:sample_tokens[x['token']]='train_smoke'
assert val_count==6019, val_count
wanted={};kinds={}
for x in records('sample_data'):
    name=x['filename']
    if x['sample_token'] in sample_tokens and x['is_key_frame'] and (name.startswith('samples/CAM_') or name.startswith('samples/LIDAR_TOP/')):
        wanted[name]=x['sample_token'];kinds[name]=sample_tokens[x['sample_token']]
manifest={'val_scenes':150,'val_samples':val_count,'train_smoke_scene':smoke_scene['name'],'train_smoke_samples':sum(x=='train_smoke' for x in sample_tokens.values()),'files':wanted,'split':kinds}
(ROOT/'manifests/nuscenes_required_files.json').write_text(json.dumps(manifest))
print(json.dumps({'event':'selected','val_samples':val_count,'selected_files':len(wanted),'train_smoke_scene':smoke_scene['name']}),flush=True)
for zp,subdir in [('/root/autodl-pub/nuScenes/Mapexpansion/nuScenes-map-expansion-v1.3.zip','data/nuscenes/maps'),('/root/autodl-pub/nuScenes/CANbusexpansion/can_bus.zip','data')]:
    dest=ROOT/subdir
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            target=(dest/info.filename).resolve()
            if not target.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe zip path')
        z.extractall(dest)
    print(json.dumps({'event':'public_archive_extracted','source':zp,'destination':str(dest)}),flush=True)
for archive in sorted(pathlib.Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval').glob('*_blobs.tgz')):
    marker=ROOT/'manifests'/('nuscenes_'+archive.name+'.done.json')
    if marker.exists():continue
    start=time.monotonic();count=0;total=0;last=start
    print(json.dumps({'event':'scan_start','archive':str(archive)}),flush=True)
    with tarfile.open(archive,'r|gz',bufsize=1024*1024) as tar:
        for member in tar:
            name=member.name.removeprefix('./')
            if name in wanted and member.isfile():
                target=DATA/name
                if not(target.exists() and target.stat().st_size==member.size):
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temp=pathlib.Path(str(target)+'.part')
                    with tar.extractfile(member) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
                    temp.replace(target)
                count+=1;total+=member.size
            now=time.monotonic()
            if now-last>60:
                print(json.dumps({'event':'scan_progress','archive':archive.name,'found':count,'tar_offset':member.offset,'seconds':round(now-start)}),flush=True);last=now
    record={'source':str(archive),'files':count,'bytes':total,'seconds':round(time.monotonic()-start)}
    marker.write_text(json.dumps(record));print(json.dumps({'event':'scan_done',**record}),flush=True)
missing=[x for x in wanted if not (DATA/x).exists()]
(ROOT/'manifests/nuscenes_coverage.json').write_text(json.dumps({'expected':len(wanted),'missing':missing,'val_samples':val_count},indent=2))
if missing:raise RuntimeError(f'{len(missing)} files missing')
print('nuScenes selected validation and training smoke files are complete',flush=True)
