import json,pathlib,pickle,sys
import numpy as np
R=pathlib.Path('/root/autodl-tmp/manner-repro');out=pathlib.Path(sys.argv[1])
train=json.loads((out/'train_audit.json').read_text());resumed=json.loads((out/'resumed_audit.json').read_text())
restore=json.loads((out/'resume_verified.json').read_text())
metrics=json.loads((out/'predictions.metrics.json').read_text())
with (out/'predictions.pkl').open('rb') as f:pred=pickle.load(f)
assert len(pred)==8,len(pred)
assert train['seconds']>=450 and train['backward_steps']>=5
assert resumed['start_iter']==train['end_iter'] and resumed['end_iter']>resumed['start_iter']
assert resumed['start_epoch']==train['end_epoch']
assert all(restore[k] for k in ('model','optimizer','fp16_scaler','rng','next_lr'))
assert np.isfinite(metrics['NuscMap_chamfer/mAP'])
record=dict(train=train,resumed=resumed,restore=restore,evaluation_frames=len(pred),directory=str(out),metrics=metrics,formal_evaluation_completed=False)
(R/'manifests/umpe_gpu_smoke_passed.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
