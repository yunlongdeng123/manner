import json,pathlib,pickle,sys
R=pathlib.Path('/root/autodl-tmp/manner-repro');out=pathlib.Path(sys.argv[1])
train=json.loads((out/'train_audit.json').read_text())
resume=json.loads((out/'resumed_audit.json').read_text())
restored=json.loads((out/'resume_verified.json').read_text())
metrics=json.loads((out/'metrics.json').read_text())
assert train['seconds']>=450 and train['end_step']>=5 and train['backward_steps']>=5
assert resume['start_step']==train['end_step'] and resume['end_step']>=train['end_step']+3
assert all(restored[k] for k in ('model','optimizer','scheduler','rng'))
assert {'pr','topo','graph'}<=metrics.keys()
from MapPrior.tools.eval.metric import load_gt,load_ar_pred
gt=load_gt(str(R/'data/topogpt_smoke/openlane_subA/data_dict_subset_A_val.pkl'))
pred=load_ar_pred(str(out/'predictions.pkl'),region_crop=True,ret_topo=False,filter_score=0)
assert gt.keys()==pred.keys(),f'Smoke prediction coverage mismatch: {len(gt)} vs {len(pred)}'
record=dict(train=train,resume=resume,restored=restored,evaluation_frames=len(gt),metrics_path=str(out/'metrics.json'),directory=str(out),formal_evaluation_completed=False)
(R/'manifests/topogpt_gpu_smoke_passed.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
