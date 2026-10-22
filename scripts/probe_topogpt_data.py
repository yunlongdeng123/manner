"""CPU data/augmentation wiring check only; does not replace a GPU smoke."""
import json,os,pathlib
import cv2,hydra,numpy as np,torch
from hydra import compose,initialize_config_dir
R=pathlib.Path('/root/autodl-tmp/manner-repro')
os.environ['PROJECT_ROOT']=str(R/'repos/TopoGPT');cv2.setNumThreads(1)
with initialize_config_dir(version_base='1.3',config_dir=str(R/'repos/TopoGPT/MapPrior/configs')):
    cfg=compose(config_name='train',overrides=['model=finetune','data=finetune_subA','exp_name=cpu_data_probe'])
cfg.paths.data_dir=str(R/'data/topogpt_smoke')
cfg.data.batch_size=1;cfg.data.num_workers=0
dm=hydra.utils.instantiate(cfg.data);dm.setup('fit')
def describe(x):
    if torch.is_tensor(x):
        if x.is_floating_point():assert torch.isfinite(x).all().item()
        return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,dict):return {k:describe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [describe(v) for v in x]
    return str(x)
report={}
for name,loader in [('train',dm.train_dataloader()),('val',dm.val_dataloader()[0])]:
    batch=next(iter(loader))
    assert 'img' in batch and 'vec_idx' in batch and 'vec_mask' in batch
    assert batch['vec_idx'].min()>=0
    assert batch['vec_idx'].max()<64**2+4
    report[name]=describe(batch)
report['scope']='Actual CPU image/target loading and augmentation; no model forward or GPU validation'
(R/'manifests/topogpt_loader_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
