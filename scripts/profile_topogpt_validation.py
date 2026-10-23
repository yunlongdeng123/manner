"""Time official loss/decode calls on eight real frames; no running-job mutation."""
import json,os,pathlib,time
import cv2,hydra,torch
from hydra import compose,initialize_config_dir
from lightning.fabric.utilities.apply_func import move_data_to_device
R=pathlib.Path('/root/autodl-tmp/manner-repro')
os.environ['PROJECT_ROOT']=str(R/'repos/TopoGPT')
cv2.setNumThreads(1)
with initialize_config_dir(version_base='1.3',config_dir=str(R/'repos/TopoGPT/MapPrior/configs')):
    cfg=compose(config_name='train',overrides=['model=finetune','data=finetune_subA','exp_name=validation_profile'])
cfg.paths.data_dir=str(R/'data/topogpt_smoke');cfg.data.batch_size=1;cfg.data.num_workers=0
dm=hydra.utils.instantiate(cfg.data);dm.setup('fit')
model=hydra.utils.instantiate(cfg.model)
ck=torch.load(str(R/'weights/topogpt/finetune.ckpt'),map_location='cpu',mmap=True)
model.load_state_dict(ck['state_dict'],strict=True);del ck
model=model.cuda().eval()
# This diagnostic has no Lightning Trainer; suppress only scalar logging.
model._log_losses=lambda *args,**kwargs:None
rows=[]
with torch.inference_mode():
    for i,batch in enumerate(dm.val_dataloader()[0]):
        batch=move_data_to_device(batch,torch.device('cuda'))
        torch.cuda.synchronize();start=time.monotonic()
        logits,loss=model._calculate_loss(batch,'val','profile')
        torch.cuda.synchronize();t1=time.monotonic()
        assert torch.isfinite(loss['total_loss']).item()
        del logits,loss
        lanes,probs,seq,seq_probs=model.pred_lanes_from_bev_feats(batch)
        torch.cuda.synchronize();t2=time.monotonic()
        model.gpt_net.unsetup_caches()
        row=dict(frame=i,loss_seconds=t1-start,decode_seconds=t2-t1,tokens=int(seq.shape[-1]))
        rows.append(row);print(json.dumps(row),flush=True)
record=dict(frames=rows,shared_gpu=True,diagnostic_only=True)
record['loss_fraction']=sum(x['loss_seconds'] for x in rows)/sum(x['loss_seconds']+x['decode_seconds'] for x in rows)
(R/'manifests/topogpt_validation_profile.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
