"""Lossless frame-key selection from publisher priors; source files preserved."""
import gc,hashlib,json,pathlib,pickle,time
import numpy as np
import torch
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'data/nuscenes'
groups={}
for split,name in [('val','val'),('train_smoke','train_smoke'),('val_smoke','val_smoke_diverse')]:
    with (D/('nuscenes_map_infos_temporal_'+name+'.pkl')).open('rb') as f: info=pickle.load(f)
    groups[split]=[i['token'] for i in info['infos']]
outdir=R/'data/umpe_priors/selected';outdir.mkdir(exist_ok=True)
def digest(obj):
    h=hashlib.sha256();count=0
    def visit(x):
        nonlocal count
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();assert torch.isfinite(y).all()
            h.update(str((tuple(y.shape),str(y.dtype))).encode());h.update(y.numpy().tobytes());count+=1
        elif isinstance(x,np.ndarray):
            assert np.isfinite(x).all();h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes());count+=1
        elif isinstance(x,dict):
            for k in sorted(x):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return dict(sha256=h.hexdigest(),arrays=count)
report={};start=time.monotonic()
for prior,name in [('hd','nuscenes_trainval_hdmap.pkl'),('sd','nuscenes_trainval_sdmap.pkl')]:
    print('Loading '+name,flush=True)
    with (R/'data/umpe_priors'/name).open('rb') as f:full=pickle.load(f)
    for split,keys in groups.items():
        subset={k:full[k] for k in keys};before=digest(subset)
        path=outdir/(prior+'_'+split+'.pkl')
        assert not path.exists(),path
        with path.open('wb') as f:pickle.dump(subset,f,protocol=4)
        with path.open('rb') as f:restored=pickle.load(f)
        assert digest(restored)==before
        report[prior+'_'+split]=dict(frames=len(keys),file=str(path),bytes=path.stat().st_size,verified=before)
        del subset,restored;gc.collect();print(json.dumps(report[prior+'_'+split]),flush=True)
    del full;gc.collect()
report['seconds']=time.monotonic()-start
(R/'manifests/umpe_selected_priors_verified.json').write_text(json.dumps(report,indent=2))
print('All selected priors verified',flush=True)
