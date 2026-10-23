"""CPU-only replay of the pinned official UMPE metric on saved predictions.

No image, map prior, dataset constructor or model is loaded. The official
formatting and scoring methods receive just their required metadata fields.
"""
import argparse,json,pathlib,pickle,sys,types
p=argparse.ArgumentParser()
p.add_argument('--repo',default='/root/autodl-tmp/manner-repro/repos/UMPE/Maptrv2')
p.add_argument('--pred',required=True);p.add_argument('--infos',required=True)
p.add_argument('--gt',required=True);p.add_argument('--out-dir',required=True)
p.add_argument('--reference')
a=p.parse_args();repo=pathlib.Path(a.repo).resolve()
sys.path[:0]=[str(repo),str(repo/'mmdetection3d')]
import mmcv,numpy as np,torch
from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import CustomNuScenesLocalMapDataset
out=pathlib.Path(a.out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
cfg=mmcv.Config.fromfile(str(repo/'projects/configs/maptrv2/maptrv2_nusc_r50_24ep_umpe.py'))
with open(a.infos,'rb') as f:infos=sorted(pickle.load(f)['infos'],key=lambda x:x['timestamp'])
with open(a.pred,'rb') as f:pred=pickle.load(f)
gt_path=pathlib.Path(a.gt).resolve();gt=json.loads(gt_path.read_text())
assert len(pred)==len(infos)==len(gt['GTs'])
assert [x['token'] for x in infos]==[x['sample_token'] for x in gt['GTs']],'GT frame order mismatch'
metadata=types.SimpleNamespace(data_infos=infos,MAPCLASSES=cfg.map_classes,
    NUM_MAPCLASSES=len(cfg.map_classes),map_ann_file=str(gt_path),
    modality=cfg.input_modality,fixed_num=cfg.fixed_ptsnum_per_gt_line,
    eval_use_same_gt_sample_num_flag=cfg.eval_use_same_gt_sample_num_flag,
    pc_range=cfg.point_cloud_range)
formatted=CustomNuScenesLocalMapDataset._format_bbox(metadata,[x['pts_bbox'] for x in pred],str(out/'formatted'))
metrics=CustomNuScenesLocalMapDataset._evaluate_single(metadata,formatted,metric='chamfer')
metrics={k:float(v) for k,v in metrics.items()}
assert all(np.isfinite(v) for v in metrics.values())
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
audit=dict(frames=len(pred),cuda_initialized=torch.cuda.is_initialized(),official_methods=True,
    images_or_priors_loaded=False,model_instantiated=False,reference_match=None)
assert not audit['cuda_initialized']
if a.reference:
    reference=json.loads(pathlib.Path(a.reference).read_text())
    assert set(metrics)==set(reference)
    diffs={k:abs(metrics[k]-reference[k]) for k in metrics}
    assert max(diffs.values())<=1e-12,diffs
    audit.update(reference_match=True,max_absolute_difference=max(diffs.values()))
(out/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'metrics':metrics,'audit':audit},indent=2))
