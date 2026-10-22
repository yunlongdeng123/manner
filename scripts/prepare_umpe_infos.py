"""Use UMPE's unchanged map-conversion functions with bounded metadata loading.

The converter discards all 3D boxes from get_sample_data. This adapter returns
only the paths/intrinsics that it consumes. All coordinate transforms, map
geometry, CAN-bus handling and sweep metadata remain in the official functions.
The full original JSON tables are preserved. Each complete scene is resumable.
"""
import ast, collections, functools, gc, hashlib, json, os, pathlib, pickle, sqlite3, sys, time
import ijson
import numpy as np
R=pathlib.Path('/root/autodl-tmp/manner-repro')
N=R/'data/nuscenes'
SRC=R/'repos/UMPE/Maptrv2/tools/maptrv2/custom_nusc_map_converter.py'

def records(table):
    with (N/'v1.0-trainval'/(table+'.json')).open('rb') as f:
        yield from ijson.items(f,'item',use_float=True)

class MetadataView:
    def __init__(self, selected):
        cache=R/'data/umpe_metadata.sqlite'
        fingerprint={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in (N/'v1.0-trainval').glob('*.json')}
        if cache.exists():
            self.db=sqlite3.connect(str(cache))
            saved=pickle.loads(self.db.execute('select payload from state').fetchone()[0])
            assert saved['fingerprint']==fingerprint and saved['selected']==selected
            self.tables=saved['tables'];self.samples=self.tables['sample'];self.sample=[]
            self.db.execute('pragma cache_size=-8192')
            print('Loaded disk-backed metadata index',flush=True)
            return
        part=cache.with_suffix('.part')
        if part.exists():part.unlink()  # Our interrupted disposable index only.
        self.db=sqlite3.connect(str(part))
        self.db.executescript('pragma cache_size=-8192; pragma journal_mode=OFF; create table records (tab text, token text, value text, primary key(tab,token)) without rowid; create table state(payload blob);')
        self.tables={}
        for name in ('scene','log','calibrated_sensor','sensor'):
            self.tables[name]={x['token']:x for x in records(name)}
        self.samples={x['token']:x for x in records('sample') if x['token'] in selected}
        needed_pose=set(); count=0;scanned=0
        for s in self.samples.values():s['data']={}
        print('Indexing selected sample_data records on disk',flush=True)
        for x in records('sample_data'):
            scanned+=1
            if scanned%250000==0:print(json.dumps(dict(event='sample_data_scan',scanned=scanned,selected=count)),flush=True)
            if x['sample_token'] not in selected:continue
            channel=pathlib.PurePosixPath(x['filename']).parts[1]
            if channel!='LIDAR_TOP' and not (channel.startswith('CAM_') and x['is_key_frame']):continue
            self.db.execute('insert into records values (?,?,?)',('sample_data',x['token'],json.dumps(x)))
            count+=1;needed_pose.add(x['ego_pose_token'])
            if x['is_key_frame']:self.samples[x['sample_token']]['data'][channel]=x['token']
        self.tables['sample']=self.samples
        print('Indexing referenced ego poses on disk',flush=True)
        for x in records('ego_pose'):
            if x['token'] in needed_pose:self.db.execute('insert into records values (?,?,?)',('ego_pose',x['token'],json.dumps(x)))
        self.sample=[]
        for s in self.samples.values():assert len(s['data'])==7,(s['token'],s['data'])
        self.db.execute('insert into state values (?)',(pickle.dumps(dict(fingerprint=fingerprint,selected=selected,tables=self.tables)),))
        self.db.commit();self.db.close();part.replace(cache)
        self.db=sqlite3.connect(str(cache));self.db.execute('pragma cache_size=-8192')
        print(json.dumps({'event':'metadata_loaded','samples':len(self.samples),'sample_data':count,'ego_pose':len(needed_pose)}),flush=True)
    @functools.lru_cache(maxsize=2048)
    def get(self, table, token):
        if table in self.tables:return self.tables[table][token]
        row=self.db.execute('select value from records where tab=? and token=?',(table,token)).fetchone()
        if row is None:raise KeyError((table,token))
        return json.loads(row[0])
    def get_sample_data_path(self, token):return str(N/self.get('sample_data',token)['filename'])
    def get_sample_data(self, token):
        sd=self.get('sample_data',token)
        cs=self.get('calibrated_sensor',sd['calibrated_sensor_token'])
        sensor=self.get('sensor',cs['sensor_token'])
        intrinsic=np.array(cs['camera_intrinsic']) if sensor['modality']=='camera' else None
        return self.get_sample_data_path(token),[],intrinsic

def official_module():
    tree=ast.parse(SRC.read_text())
    body=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):
            # These two unused imports trigger CUDA-extension loading unnecessarily.
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('mmdet3d'):continue
            body.append(node)
    ns={'__name__':'official_umpe_converter','__file__':str(SRC)}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(SRC),'exec'),ns)
    return ns

def cache_geometry(m):
    # Map polygons/lines are immutable inputs; intersection/rotation return new
    # geometries. Cache only their repeated extraction, retaining official math.
    for name in ('extract_polygon','extract_line'):
        setattr(m,name,functools.lru_cache(maxsize=8192)(getattr(m,name)))

def main():
    probe='--probe' in sys.argv
    if not probe:
        coverage=json.loads((R/'manifests/nuscenes_coverage.json').read_text())
        assert not coverage['missing']
    required=json.loads((R/'manifests/nuscenes_required_files.json').read_text())
    chosen=set(required['files'].values())
    os.chdir(R/'repos/UMPE/Maptrv2')
    ns=official_module()
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    nusc=MetadataView(chosen)
    canbus=NuScenesCanBus(dataroot=str(R/'data'))
    if probe:
        sample=next(s for s in nusc.samples.values() if pathlib.Path(nusc.get_sample_data_path(s['data']['LIDAR_TOP'])).exists())
        scene=nusc.get('scene',sample['scene_token'])
        loc=nusc.get('log',scene['log_token'])['location']
        m=ns['NuScenesMap'](dataroot=str(N),map_name=loc)
        nusc.sample=[sample]
        train,val=ns['_fill_trainval_infos'](nusc,canbus,{loc:m},{loc:ns['CNuScenesMapExplorer'](m)},set(),{scene['token']},test=False,max_sweeps=10)
        report=dict(token=sample['token'],camera_count=len(val[0]['cams']),annotation_counts={k:len(v) for k,v in val[0]['annotation'].items()})
        cache_geometry(m)
        _,cached=ns['_fill_trainval_infos'](nusc,canbus,{loc:m},{loc:ns['CNuScenesMapExplorer'](m)},set(),{scene['token']},test=False,max_sweeps=10)
        for key,lines in val[0]['annotation'].items():
            assert len(lines)==len(cached[0]['annotation'][key])
            assert all(np.array_equal(a,b) for a,b in zip(lines,cached[0]['annotation'][key]))
        report['geometry_cache_exact_match']=True
        (R/'manifests/umpe_converter_probe.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return
    grouped=collections.defaultdict(list)
    for s in nusc.samples.values():grouped[s['scene_token']].append(s)
    work=R/'data/umpe_info_scenes';work.mkdir(exist_ok=True)
    source_sha=hashlib.sha256(SRC.read_bytes()).hexdigest()
    maps={};explorers={}
    train=[];val=[];start=time.monotonic()
    order=sorted(grouped,key=lambda t:nusc.get('log',nusc.get('scene',t)['log_token'])['location'])
    for token in order:
        samples=grouped[token]
        scene=nusc.get('scene',token);name=scene['name']
        split='train_smoke' if name==required['train_smoke_scene'] else 'val'
        cache=work/(name+'.pkl')
        if cache.exists():
            result=pickle.loads(cache.read_bytes())
            assert result['source_sha256']==source_sha,'Converter changed: use a new cache directory'
            infos=result['infos']
        else:
            assert samples[0]['prev']=='' and samples[-1]['next']=='',name
            assert all(a['next']==b['token'] for a,b in zip(samples,samples[1:])),name
            loc=nusc.get('log',scene['log_token'])['location']
            if loc not in maps:
                maps.clear();explorers.clear()
                gc.collect()
                maps[loc]=ns['NuScenesMap'](dataroot=str(N),map_name=loc)
                cache_geometry(maps[loc])
                explorers[loc]=ns['CNuScenesMapExplorer'](maps[loc])
            nusc.sample=samples
            tr,va=ns['_fill_trainval_infos'](nusc,canbus,maps,explorers,{token} if split=='train_smoke' else set(),{token} if split=='val' else set(),test=False,max_sweeps=10)
            infos=tr+va
            assert len(infos)==len(samples)
            tmp=cache.with_suffix('.part')
            tmp.write_bytes(pickle.dumps(dict(source_sha256=source_sha,infos=infos)))
            tmp.replace(cache)
        (train if split=='train_smoke' else val).extend(infos)
        print(json.dumps(dict(event='scene_ready',scene=name,split=split,frames=len(infos),elapsed=round(time.monotonic()-start))),flush=True)
    assert len(val)==6019 and len(train)==required['train_smoke_samples']
    assert set(i['token'] for i in train+val)==chosen
    for split,infos in [('train_smoke',train),('val',val),('val_smoke',val[:8])]:
        path=N/('nuscenes_map_infos_temporal_'+split+'.pkl')
        path.write_bytes(pickle.dumps(dict(infos=infos,metadata=dict(version='v1.0-trainval'))))
    report=dict(val_frames=len(val),train_smoke_frames=len(train),val_scenes=150,converter_sha256=source_sha,full_train_prepared=False,adapter='paths and intrinsics only; discarded 3D boxes omitted; official geometric functions retained')
    (R/'manifests/umpe_infos_ready.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
