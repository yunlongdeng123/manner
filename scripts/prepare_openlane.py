"""Memory-bounded equivalent of OpenLane-V2 v1.1.0 collect(point_interval=1)."""
import json,os,pathlib,pickle,subprocess,sys,numpy as np
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro')
def collect(raw,selection,output):
    meta={};missing=[];images=set()
    for split,segs in selection.items():
        for segment,timestamps in segs.items():
            for stamp in timestamps:
                stamp=stamp.split('.')[0];path=raw/split/segment/'info'/(stamp+'.json')
                if not path.exists():missing.append(str(path));continue
                frame=json.loads(path.read_text())
                for k,v in frame['pose'].items():frame['pose'][k]=np.array(v,dtype=np.float64)
                for camera in frame['sensor'].values():
                    images.add(camera['image_path'])
                    for para in ('intrinsic','extrinsic'):
                        for k,v in camera[para].items():camera[para][k]=np.array(v,dtype=np.float64)
                if 'annotation' in frame:
                    anno=frame['annotation']
                    for lane in anno['lane_centerline']:lane['points']=np.array(lane['points'],dtype=np.float32)
                    for obj in anno['traffic_element']:obj['points']=np.array(obj['points'],dtype=np.float32)
                    for key in ('topology_lclc','topology_lcte'):anno[key]=np.array(anno[key],dtype=np.int8)
                meta[(split,segment,stamp)]=frame
    if missing:raise RuntimeError(f'{len(missing)} annotations missing; first: {missing[0]}')
    missing_images=[str(raw/x) for x in images if not (raw/x).exists()]
    if missing_images:raise RuntimeError(f'{len(missing_images)} images missing; first: {missing_images[0]}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as f:pickle.dump(meta,f)
    return {'frames':len(meta),'images':len(images),'missing_images':0,'annotation_file':str(output)}
def link(src,dst):
    if not dst.exists():dst.symlink_to(src,target_is_directory=True)
def convert(anno,output):
    cmd=[sys.executable,str(ROOT/'repos/TopoGPT/MapPrior/datasets/openlanev2/openlane_map_gen.py'),'--source-data-info',str(anno),'--save-path',str(output)]
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'repos/TopoGPT')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run(cmd,cwd=ROOT/'repos/TopoGPT',check=True,env=env)
def collect_full():
    raw=ROOT/'data/openlanev2';out=ROOT/'data/topogpt/openlane_subA'
    official=ROOT/'repos/OpenLane-V2/data/OpenLane-V2/data_dict_subset_A.json'
    if not (raw/'data_dict_subset_A.json').exists():
        (raw/'data_dict_subset_A.json').write_bytes(official.read_bytes())
    assert json.loads((raw/'data_dict_subset_A.json').read_text())==json.loads(official.read_text()),'Split differs from official v1.1.0'
    split=json.loads((raw/'data_dict_subset_A.json').read_text())
    out.mkdir(parents=True,exist_ok=True);link(raw/'val',out/'val')
    anno=out/'data_dict_subset_A_val.pkl'
    if not anno.exists():collect(raw,{'val':split['val']},anno)
    print('OpenLane annotations serialized; collection process can exit before map conversion',flush=True)

def verify_full():
    raw=ROOT/'data/openlanev2';out=ROOT/'data/topogpt/openlane_subA'
    split=json.loads((raw/'data_dict_subset_A.json').read_text())['val']
    expected={('val',seg,stamp.split('.')[0]) for seg,stamps in split.items() for stamp in stamps}
    with (out/'data_dict_subset_A_val.pkl').open('rb') as f:meta=pickle.load(f)
    assert set(meta)==expected
    images={cam['image_path'] for frame in meta.values() for cam in frame['sensor'].values()}
    missing=[x for x in images if not (raw/x).is_file()]
    assert not missing,(len(missing),missing[:3])
    with (out/'val.pkl').open('rb') as f:maps=pickle.load(f)
    mapkeys={('val',str(x['log_token']),str(x['timestamp'])) for x in maps}
    assert mapkeys==expected and len(maps)==len(expected)
    assert all(np.isfinite(x['map_vec']).all() for x in maps)
    report=dict(frames=len(meta),images=len(images),missing_images=0,converted_frames=len(maps),annotation_file=str(out/'data_dict_subset_A_val.pkl'),official_split_exact_match=True)
    (ROOT/'manifests/openlane_coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

def prep_full():
    # Each large phase exits before the next begins. Keeping the collection
    # process alive retains its allocator heap and exceeds the no-GPU RAM cap.
    subprocess.run([sys.executable,__file__,'--collect-only'],check=True)
    out=ROOT/'data/topogpt/openlane_subA'
    if not (out/'val.pkl').exists():convert(out/'data_dict_subset_A_val.pkl',out/'val.pkl')
    subprocess.run([sys.executable,__file__,'--verify-full'],check=True)
def prep_smoke():
    out=ROOT/'data/topogpt_smoke/openlane_subA';out.mkdir(parents=True,exist_ok=True)
    report={}
    for split in ('train','val'):
        raw=ROOT/'data'/('openlane_smoke' if split=='train' else 'openlanev2')
        selection={}
        if split=='train':
            for p in sorted((raw/split).glob('*/info/*.json')):
                selection.setdefault(p.parent.parent.name,[]).append(p.stem)
        else:
            official=json.loads((raw/'data_dict_subset_A.json').read_text())['val']
            for segment,timestamps in official.items():
                selection[segment]=timestamps[:2]
                if sum(map(len,selection.values()))>=8:break
        if not selection:raise RuntimeError(f'No {split} annotations')
        link(raw/split,out/split)
        anno=out/f'data_dict_subset_A_{split}.pkl'
        report[split]=collect(raw,{split:selection},anno)
        convert(anno,out/f'{split}.pkl')
    (ROOT/'manifests/openlane_smoke_coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    if '--collect-only' in sys.argv:collect_full()
    elif '--verify-full' in sys.argv:verify_full()
    elif '--smoke' in sys.argv:prep_smoke()
    else:prep_full()
