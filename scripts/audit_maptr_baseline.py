"""Check official baseline state/config before using it as a diagnostic control."""
import copy,json,pathlib,torch
from mmcv import Config
from mmdet3d.models import build_model
import projects.mmdet3d_plugin
R=pathlib.Path('/root/autodl-tmp/manner-repro')
c=Config.fromfile(str(R/'configs/maptr_baseline_control.py'))
ck=torch.load(str(R/'weights/diagnostic/maptrv2_nusc_r50_24e.pth'),map_location='cpu')
m=build_model(copy.deepcopy(c.model),train_cfg=c.get('train_cfg'),test_cfg=c.get('test_cfg'))
loaded=m.load_state_dict(ck['state_dict'],strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
assert not c.model.get('map_encoder')
src=ck['meta'].get('config','')
(R/'manifests/maptr_baseline_embedded_config.txt').write_text(src)
saved=Config.fromstring(src,'.py')
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for k in sorted(a.keys()|b.keys()) for d in diff(a.get(k),b.get(k),path+'/'+str(k))]
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b):
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in diff(x,y,path+'/'+str(i))]
    return [] if a==b else [dict(path=path,local=repr(a),checkpoint=repr(b))]
report=dict(strict_load=True,epoch=ck['meta'].get('epoch'),iter=ck['meta'].get('iter'),model_differences=diff(c.model,saved.model),test_pipeline_differences=diff(c.test_pipeline,saved.test_pipeline),test_dataset_differences=diff(c.data.test,saved.data.test),parameters=sum(p.numel() for p in m.parameters()),uses_map_priors_in_model=False)
(R/'manifests/maptr_baseline_config_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
assert not report['model_differences'] and not report['test_pipeline_differences']
