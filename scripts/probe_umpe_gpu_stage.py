"""Strict CPU checkpoint loading and real pipeline audit after CUDA build."""
import gc,json,os,pathlib,time
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin
R=pathlib.Path('/root/autodl-tmp/manner-repro')
os.chdir(R/'repos/UMPE/Maptrv2')
cfg=Config.fromfile(str(R/'configs/umpe_reproduce.py'))
start=time.monotonic();model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
record=dict(checkpoints={},device='cpu',cuda_numerics='separate GPU smoke required')
for name in ['maptrv2_umpe.pth','maptrv2_umpe_ft.pth']:
    ck=torch.load(R/'weights/umpe'/name,map_location='cpu')
    result=model.load_state_dict(ck['state_dict'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    record['checkpoints'][name]=dict(strict=True,epoch=ck['meta']['epoch'],iter=ck['meta']['iter'])
    del ck;gc.collect();print(name+' strict load passed',flush=True)
del model;gc.collect()
cfg.data.test.test_mode=True
ds=build_dataset(cfg.data.test)
assert len(ds)==6019
tokens={i['token'] for i in ds.data_infos}
assert tokens<=ds.vectorized_sdmap.keys() and tokens<=ds.vectorized_hdmap.keys()
selected=json.loads((R/'manifests/umpe_smoke_selection.json').read_text())
print('Selection manifest:',selected,flush=True)
def describe(x):
    if isinstance(x,DataContainer):return describe(x.data)
    if torch.is_tensor(x):
        assert torch.isfinite(x).all(), 'nonfinite tensor'
        return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,np.ndarray):
        if np.issubdtype(x.dtype,np.number):assert np.isfinite(x).all()
        return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,dict):return {k:describe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [describe(v) for v in x]
    return str(type(x).__name__)
record.update(frames=len(ds),sd_prior_keys=len(ds.vectorized_sdmap),hd_prior_keys=len(ds.vectorized_hdmap),first_pipeline=describe(ds[0]),seconds=time.monotonic()-start)
(R/'manifests/umpe_real_loader_and_strict_weights.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
