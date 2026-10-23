"""Deterministic qualitative checks; display filtering never affects metrics."""
import argparse,json,pathlib,pickle,random,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=pathlib.Path('/root/autodl-tmp/manner-repro')
sys.path.insert(0,str(R/'repos/TopoGPT'))
from MapPrior.datasets.map_dataset import get_ar_topo_pred
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
out=R/'reports'/('smoke_figures' if a.smoke else 'figures');out.mkdir(parents=True,exist_ok=True)
def read(path):
    with pathlib.Path(path).open('rb') as f:return pickle.load(f)
def style(ax,xlim,ylim,title):
    ax.set(xlim=xlim,ylim=ylim,aspect='equal',title=title)
    ax.grid(alpha=.16);ax.scatter([0],[0],s=14,c='#111827',marker='^',zorder=5)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():spine.set_color('#cbd5e1')
    ax.set_xlabel('x (m)',fontsize=8);ax.set_ylabel('y (m)',fontsize=8)
def line(ax,points,color):
    pts=np.asarray(points)
    if len(pts)>1:ax.plot(pts[:,0],pts[:,1],color=color,lw=1.15,alpha=.9)
def graph(ax,obj,color):
    lanes=obj['lane_centerline']
    for lane in lanes:line(ax,lane['points'],color)
    adj=np.asarray(obj['topology_lclc']) if 'topology_lclc' in obj else get_ar_topo_pred([x['points'] for x in lanes],dist_thre=.5,ret_hard=True)
    for i,j in zip(*np.where(adj>.5)):
        u=np.asarray(lanes[i]['points'])[-1,:2];v=np.asarray(lanes[j]['points'])[0,:2]
        ax.annotate('',xy=v,xytext=u,arrowprops=dict(arrowstyle='->',lw=.6,color='#64748b',alpha=.65,linestyle='dashed'))
if a.smoke:
    ts=json.loads((R/'manifests/topogpt_gpu_smoke_passed.json').read_text())
    us=json.loads((R/'manifests/umpe_gpu_smoke_passed.json').read_text())
    tp=pathlib.Path(ts['directory'])/'predictions.pkl';up=pathlib.Path(us['directory'])/'predictions.pkl'
    tg=R/'data/topogpt_smoke/openlane_subA/data_dict_subset_A_val.pkl'
    ug=R/'data/nuscenes/nuscenes_map_infos_temporal_val_smoke_diverse.pkl'
else:
    summary=json.loads((R/'reports/results.json').read_text())
    tp=pathlib.Path(summary['runs']['topogpt']['directory'])/'predictions.pkl'
    up=pathlib.Path(summary['runs']['umpe_finetuned']['directory'])/'predictions.pkl'
    tg=R/'data/topogpt/openlane_subA/data_dict_subset_A_val.pkl'
    ug=R/'data/nuscenes/nuscenes_map_infos_temporal_val.pkl'
pred=read(tp);gt={('val',str(k[1]),str(k[2])):v['annotation'] for k,v in read(tg).items()}
segments=sorted({k[1] for k in pred});chosen=random.Random(3407).sample(segments,min(6,len(segments)))
keys=[sorted(k for k in pred if k[1]==s)[0] for s in chosen]
for k in sorted(pred):
    if len(keys)>=6:break
    if k not in keys:keys.append(k)
no_eos=[k for k,v in pred.items() if 2 not in v['lane_token']]
if no_eos and not a.smoke:keys[-1]=no_eos[0]
fig,axes=plt.subplots(3,4,figsize=(16,9),constrained_layout=True)
for n,key in enumerate(keys):
    left,right=axes[n//2,(n%2)*2:(n%2)*2+2]
    ident=key[1][:8]+' / '+key[2][-7:]
    graph(left,gt[key],'#16803c');graph(right,pred[key],'#2563eb')
    style(left,(-51.2,51.2),(-25.6,25.6),'GT '+ident)
    suffix=' (no EOS)' if key in no_eos else ''
    style(right,(-51.2,51.2),(-25.6,25.6),'TopoGPT'+suffix)
fig.suptitle('TopoGPT: ground truth and published-checkpoint predictions\nDashed arrows: lane connectivity; black triangle: ego origin',fontsize=13)
fig.savefig(out/'topogpt_examples.png',dpi=170);fig.savefig(out/'topogpt_examples.svg');plt.close(fig)
del pred,gt
pred=read(up);infos=sorted(read(ug)['infos'],key=lambda x:x['timestamp'])
assert len(pred)==len(infos)
indices=np.linspace(0,len(infos)-1,6,dtype=int).tolist()
colors=['#2563eb','#dc2626','#16803c'];classes=['divider','ped_crossing','boundary']
fig,axes=plt.subplots(2,6,figsize=(16,11),constrained_layout=True)
for n,idx in enumerate(indices):
    left,right=axes[n//3,(n%3)*2:(n%3)*2+2]
    info=infos[idx]
    for cls,color in zip(classes,colors):
        for pts in info['annotation'][cls]:line(left,pts,color)
    p=pred[idx]['pts_bbox']
    for pts,score,label in zip(p['pts_3d'],p['scores_3d'],p['labels_3d']):
        if float(score)>=.3:line(right,pts.cpu().numpy(),colors[int(label)])
    style(left,(-15,15),(-30,30),'GT '+info['token'][:7])
    style(right,(-15,15),(-30,30),'UMPE ft')
fig.suptitle('UMPE: ground truth and published finetuned checkpoint\nDisplay score >= 0.3 only; all scores are retained for official evaluation; black triangle: ego origin',fontsize=13)
fig.legend([Line2D([0],[0],color=c) for c in colors],classes,loc='outside lower center',ncol=3) if tuple(int(x) for x in matplotlib.__version__.split('.')[:2])>=(3,7) else fig.legend([Line2D([0],[0],color=c) for c in colors],classes,loc='lower center',ncol=3)
fig.savefig(out/'umpe_examples.png',dpi=170);fig.savefig(out/'umpe_examples.svg');plt.close(fig)
(out/'selection.json').write_text(json.dumps(dict(smoke=a.smoke,topogpt_frames=keys,topogpt_selection='Seed 3407 samples segments, first timestamp per segment; last replaced by first no-EOS frame if present',umpe_indices=indices,umpe_tokens=[infos[i]['token'] for i in indices],umpe_display_score_threshold=.3,official_metrics_unchanged=True),indent=2))
print(str(out))
