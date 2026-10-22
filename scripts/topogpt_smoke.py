"""GPU-only training/save/resume audit. Run phases in separate processes.

This is a separate engineering smoke budget. It never overwrites official
weights and does not set the full-evaluation readiness marker by itself.
"""
import argparse, hashlib, json, os, pathlib, random, time
import numpy as np
import torch
import hydra
import lightning as L
from hydra import compose, initialize_config_dir
from lightning.pytorch import Callback, Trainer
R=pathlib.Path('/root/autodl-tmp/manner-repro')

def digest(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().contiguous().cpu()
            h.update(str((tuple(y.shape),y.dtype)).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return h.hexdigest()

class Audit(Callback):
    def __init__(self, seconds, resume, out):
        self.seconds=seconds;self.resume=resume;self.out=out
        self.losses=[];self.backward=0;self.initial_step=0;self.rng=None
    def on_save_checkpoint(self,trainer,model,checkpoint):
        checkpoint['repro_rng']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def on_load_checkpoint(self,trainer,model,checkpoint):self.rng=checkpoint.get('repro_rng')
    def on_train_start(self,trainer,model):
        self.initial_step=trainer.global_step
        if self.resume:
            ck=torch.load(self.resume,map_location='cpu',mmap=True)
            assert trainer.global_step==ck['global_step']>0
            assert digest(model.state_dict())==digest(ck['state_dict']),'Model restore mismatch'
            assert digest(trainer.optimizers[0].state_dict())==digest(ck['optimizer_states'][0]),'Optimizer restore mismatch'
            assert digest(trainer.lr_scheduler_configs[0].scheduler.state_dict())==digest(ck['lr_schedulers'][0]),'Scheduler restore mismatch'
            assert self.rng is not None,'Missing RNG state'
            random.setstate(self.rng['python']);np.random.set_state(self.rng['numpy'])
            torch.set_rng_state(self.rng['torch']);torch.cuda.set_rng_state_all(self.rng['cuda'])
            (self.out/'resume_verified.json').write_text(json.dumps(dict(global_step=trainer.global_step,model=True,optimizer=True,scheduler=True,rng=True,scope='Saved state restored; bitwise data-order replay not asserted'),indent=2))
            del ck
        torch.cuda.reset_peak_memory_stats();self.start=time.monotonic()
    def on_after_backward(self,trainer,model):
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads),'Invalid gradients'
        assert any(torch.count_nonzero(g).item()>0 for g in grads),'All gradients zero'
        self.backward+=1
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        loss=outputs['loss'] if isinstance(outputs,dict) else outputs
        value=float(loss.detach().cpu());assert np.isfinite(value),value
        self.losses.append(value)
        if self.resume:
            stop=trainer.global_step>=self.initial_step+3
        else:
            stop=time.monotonic()-self.start>=self.seconds and trainer.global_step>=5
        if stop:trainer.should_stop=True
    def on_train_end(self,trainer,model):
        assert self.backward>0 and self.losses
        name='resumed' if self.resume else 'train'
        trainer.save_checkpoint(str(self.out/(name+'.ckpt')))
        report=dict(phase=name,start_step=self.initial_step,end_step=trainer.global_step,backward_steps=self.backward,seconds=time.monotonic()-self.start,loss_first=self.losses[0],loss_last=self.losses[-1],all_losses_finite=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name())
        (self.out/(name+'_audit.json')).write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--seconds',type=int,default=480);args=p.parse_args()
    assert torch.cuda.is_available(),'Requires real GPU; CPU execution is not a smoke substitute'
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    os.environ['PROJECT_ROOT']=str(R/'repos/TopoGPT')
    os.environ['TORCH_HOME']=str(R/'weights/torch')
    L.seed_everything(3407,workers=True)
    with initialize_config_dir(version_base='1.3',config_dir=str(R/'repos/TopoGPT/MapPrior/configs')):
        cfg=compose(config_name='train',overrides=['model=finetune','data=finetune_subA','exp_name=gpu_smoke'])
    cfg.paths.data_dir=str(R/'data/topogpt_smoke');cfg.paths.output_dir=str(out);cfg.paths.log_dir=str(out);cfg.paths.work_dir=str(R/'repos/TopoGPT')
    cfg.data.batch_size=1;cfg.data.num_workers=0
    # Exercise the published map-prior -> image-finetuning constructor path.
    cfg.model.pretrain_ckpt=str(R/'weights/topogpt/pretrain.ckpt') if not args.resume else None
    dm=hydra.utils.instantiate(cfg.data)
    model=hydra.utils.instantiate(cfg.model)
    if not args.resume:
        ck=torch.load(str(R/'weights/topogpt/finetune.ckpt'),map_location='cpu',mmap=True)
        model.load_state_dict(ck['state_dict'],strict=True);del ck
    audit=Audit(args.seconds,args.resume,out)
    # Same scheduler horizon in both phases; only this smoke is time-bounded.
    trainer=Trainer(accelerator='gpu',devices=1,precision=32,max_epochs=10000,max_steps=100000,min_epochs=0,gradient_clip_val=1.0,gradient_clip_algorithm='norm',num_sanity_val_steps=0,limit_val_batches=0,logger=False,enable_checkpointing=False,enable_model_summary=False,callbacks=[audit],default_root_dir=str(out))
    trainer.fit(model,datamodule=dm,ckpt_path=args.resume)

if __name__=='__main__':main()
