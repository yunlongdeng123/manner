import json,pathlib,torch
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
R=pathlib.Path('/root/autodl-tmp/manner-repro')
with initialize_config_dir(version_base='1.3',config_dir=str(R/'repos/TopoGPT/MapPrior/configs')):
    cfg=compose(config_name='train',overrides=['model=finetune','data=finetune_subA','exp_name=audit'])
current=OmegaConf.to_container(cfg.model,resolve=True)
ck=torch.load(str(R/'weights/topogpt/finetune.ckpt'),map_location='cpu',mmap=True)
saved=ck['hyper_parameters']
report={}
for key in ('optimizer_cfg','scheduler_cfg','loss_cfg','eval_cfg','sample_cfg'):
    a=current[key];b=saved[key]
    if OmegaConf.is_config(b):b=OmegaConf.to_container(b,resolve=True)
    report[key]=dict(repo=a,checkpoint=b,equal=a==b)
(R/'manifests/topogpt_config_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if not v['equal']},indent=2))
