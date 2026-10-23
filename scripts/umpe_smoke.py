"""Real-GPU UMPE smoke through the official training CLI and runner.

Stops at an epoch boundary, saves full state, then resumes in a new process.
The audit is a separate engineering budget; formal training is not launched.
"""
import argparse,copy,hashlib,json,os,pathlib,random,runpy,sys,time
import numpy as np
import torch
from mmcv.runner import HOOKS,Hook
R=pathlib.Path('/root/autodl-tmp/manner-repro')

def digest(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().contiguous().cpu()
            h.update(str((tuple(y.shape),y.dtype)).encode())
            h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return h.hexdigest()

class SmokeComplete(Exception):pass

@HOOKS.register_module()
class ReproductionMetadataJSONGuard(Hook):
    def before_run(self,runner):
        # TextLoggerHook serializes runner.meta before the audit runs. RNG
        # tensors stay intact in the checkpoint and are read directly below.
        # Remove only their runtime logging copy, then regenerate on save.
        runner.meta.pop('repro_rng',None)

@HOOKS.register_module()
class ReproductionSmokeAudit(Hook):
    def __init__(self,out,seconds=480,resume=None):
        self.out=pathlib.Path(out);self.seconds=seconds;self.resume=resume
        self.losses=[];self.backward_steps=0
        self.expected_lr=None
    def before_run(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        self.start_iter=runner.iter;self.start_epoch=runner.epoch
        if self.resume:
            ck=torch.load(self.resume,map_location='cpu')
            assert runner.iter==ck['meta']['iter']>0 and runner.epoch==ck['meta']['epoch']
            assert digest(model.state_dict())==digest(ck['state_dict']),'Model restore mismatch'
            assert digest(runner.optimizer.state_dict())==digest(ck['optimizer']),'Optimizer restore mismatch'
            fp16=[h.loss_scaler.state_dict() for h in runner.hooks if hasattr(h,'loss_scaler')]
            assert fp16 and digest(fp16[0])==digest(ck['meta']['fp16']['loss_scaler']),'FP16 scaler restore mismatch'
            rng=ck['meta']['repro_rng']
            self.expected_lr=ck['meta']['repro_next_lr']
            random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
            report=dict(iter=runner.iter,epoch=runner.epoch,model=True,optimizer=True,lr_and_schedule='optimizer groups restored; next applied learning rate pending independent saved expectation',fp16_scaler=True,rng=True,scope='Epoch-boundary state restore; bitwise cross-process trajectory not asserted')
            (self.out/'resume_verified.json').write_text(json.dumps(report,indent=2))
            del ck
        else:
            ck=torch.load(str(R/'weights/umpe/maptrv2_umpe.pth'),map_location='cpu')
            model.load_state_dict(ck['state_dict'],strict=True);del ck
        torch.cuda.reset_peak_memory_stats();self.start=time.monotonic()
    def before_train_iter(self,runner):
        if self.resume and runner.iter==self.start_iter:
            actual=[g['lr'] for g in runner.optimizer.param_groups]
            assert digest(actual)==digest(self.expected_lr),'Next-step LR restore mismatch'
            p=self.out/'resume_verified.json';report=json.loads(p.read_text())
            report.update(next_lr=True,lr_and_schedule='First resumed step matches next-step cosine/warmup LR saved by previous process',next_lr_min=min(actual),next_lr_max=max(actual))
            p.write_text(json.dumps(report,indent=2))
    def after_train_iter(self,runner):
        loss=float(runner.outputs['loss'].detach().cpu());assert np.isfinite(loss),loss
        grads=[p.grad for p in runner.model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads),'Invalid gradients'
        assert any(torch.count_nonzero(g).item()>0 for g in grads),'All gradients zero'
        self.losses.append(loss);self.backward_steps+=1
    def after_train_epoch(self,runner):
        elapsed=time.monotonic()-self.start
        if not self.resume and elapsed<self.seconds:return
        assert self.losses and self.backward_steps>0
        runner.meta['repro_rng']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        # Compute the next epoch's first LR without modifying the live runner.
        proxy=copy.copy(runner);proxy._epoch=runner.epoch+1
        hooks=[h for h in runner.hooks if hasattr(h,'get_regular_lr')]
        assert len(hooks)==1
        lr_hook=copy.copy(hooks[0]);lr_hook.regular_lr=lr_hook.get_regular_lr(proxy)
        next_lr=lr_hook.regular_lr
        if lr_hook.warmup is not None and proxy.iter<lr_hook.warmup_iters:
            next_lr=lr_hook.get_warmup_lr(proxy.iter)
        runner.meta['repro_next_lr']=next_lr
        phase='resumed' if self.resume else 'train'
        runner.save_checkpoint(str(self.out),filename_tmpl=phase+'.pth',save_optimizer=True,create_symlink=False)
        record=dict(phase=phase,start_iter=self.start_iter,end_iter=runner.iter,start_epoch=self.start_epoch,end_epoch=runner.epoch+1,backward_steps=self.backward_steps,seconds=elapsed,all_losses_finite=True,loss_first=self.losses[0],loss_last=self.losses[-1],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name())
        (self.out/(phase+'_audit.json')).write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        raise SmokeComplete()

def make_config(out,seconds,resume):
    # Preserve the checkpoint's optimizer/schedule settings; only the data and
    # logging/checkpoint workload are reduced for this separate smoke.
    cfg=(
        "_base_ = '/root/autodl-tmp/manner-repro/configs/umpe_reproduce.py'\n"
        "data = dict(samples_per_gpu=1, workers_per_gpu=0,\n"
        " train=dict(ann_file='data/nuscenes/nuscenes_map_infos_temporal_train_smoke.pkl'),\n"
        " val=dict(ann_file='data/nuscenes/nuscenes_map_infos_temporal_val_smoke_diverse.pkl', map_ann_file="+repr(str(out/'gt_smoke.json'))+"),\n"
        " test=dict(ann_file='data/nuscenes/nuscenes_map_infos_temporal_val_smoke_diverse.pkl', map_ann_file="+repr(str(out/'gt_smoke.json'))+"))\n"
        "checkpoint_config = dict(interval=100000, max_keep_ckpts=1)\n"
        "log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])\n"
        "custom_hooks = [\n"
        " dict(type='ReproductionMetadataJSONGuard',priority='HIGH'),\n"
        " dict(type='AlphaScheduleHook', start_iter=0,end_iter=20000,start_alpha=0.0,target_alpha=0.6,schedule_type='cosine'),\n"
        " dict(type='FusionAlphaMonitorHook',log_interval=100),\n"
        " dict(type='ReproductionSmokeAudit',out="+repr(str(out))+",seconds="+str(seconds)+",resume="+repr(resume)+",priority='LOWEST')]\n"
        "load_from = None\n"
    )
    if (R/'manifests/umpe_selected_priors_verified.json').exists():
        for role,split in [('train','train_smoke'),('val','val_smoke'),('test','val_smoke')]:
            base=R/'data/umpe_priors/selected'
            cfg+="data["+repr(role)+"].update(sdmap_pkl="+repr(str(base/('sd_'+split+'.pkl')))+",hdmap_pkl="+repr(str(base/('hd_'+split+'.pkl')))+")\n"
    path=out/'smoke_config.py'
    if path.exists():
        (out/('smoke_config.before_'+('resume' if resume else 'train')+'.py')).write_bytes(path.read_bytes())
    path.write_text(cfg);return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--seconds',type=int,default=480);args=p.parse_args()
    assert torch.cuda.is_available(),'Requires a real GPU'
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    os.chdir(R/'repos/UMPE/Maptrv2')
    config=make_config(out,args.seconds,args.resume)
    sys.argv=['tools/train.py',str(config),'--work-dir',str(out),'--gpus','1','--seed','0','--no-validate']
    if args.resume:sys.argv+=['--resume-from',args.resume]
    try:runpy.run_path('tools/train.py',run_name='__main__')
    except SmokeComplete:pass
    phase='resumed' if args.resume else 'train'
    assert (out/(phase+'_audit.json')).exists(),'Smoke did not reach its stop/checkpoint boundary'

if __name__=='__main__':main()
