import cv2,hashlib,json,os,pathlib,time
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin
R=pathlib.Path('/root/autodl-tmp/manner-repro');os.chdir(R/'repos/UMPE/Maptrv2')
cfg=Config.fromfile(str(R/'results/smoke/umpe_recovery_20261023T153416Z/eval_config.py'))
cfg.data.test.test_mode=True;ds=build_dataset(cfg.data.test)
def digest(obj):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,DataContainer):visit(x.data)
        elif torch.is_tensor(x):visit(x.detach().cpu().numpy())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):h.update(k.encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif isinstance(x,(str,int,float,bool,type(None))):h.update(repr(x).encode())
        else:h.update(str(type(x)).encode())
    visit(obj);return h.hexdigest()
rows=[]
for n in [112,1,1,112]:
    cv2.setNumThreads(n)
    # Warm metadata and code paths; source image pages are already cached.
    ds[0]
    start=time.monotonic();data=[ds[i] for i in range(len(ds))];seconds=time.monotonic()-start
    hashes=[digest(x) for x in data];del data
    rows.append(dict(opencv_threads=n,frames=len(ds),seconds=seconds,hashes=hashes))
    print(json.dumps({k:v for k,v in rows[-1].items() if k!='hashes'}),flush=True)
assert all(row['hashes']==rows[0]['hashes'] for row in rows)
record=dict(all_inputs_bitwise_equal=True,runs=rows)
(R/'manifests/opencv_pipeline_equivalence.json').write_text(json.dumps(record,indent=2))
print('Bitwise pipeline equivalence passed',flush=True)
