"""Audit UMPE full-validation labels and inspect deterministic examples."""
import collections,json,pathlib,pickle
import numpy as np,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=pathlib.Path('/root/autodl-tmp/manner-repro');O=R/'reports/diagnostics';O.mkdir(exist_ok=True)
P=R/'results/umpe/finetuned_20261023T154019Z/predictions.pkl'
with P.open('rb') as f:pred=pickle.load(f)
with (R/'data/nuscenes/nuscenes_map_infos_temporal_val.pkl').open('rb') as f:infos=sorted(pickle.load(f)['infos'],key=lambda x:x['timestamp'])
gt=json.loads((R/'data/nuscenes/nuscenes_map_anns_val.json').read_text())['GTs']
classes=['divider','ped_crossing','boundary'];colors=['#2563eb','#dc2626','#16803c']
counts=collections.Counter();scores={c:[] for c in classes};arrays=0
assert len(pred)==len(infos)==len(gt)==6019
for result,info,truth in zip(pred,infos,gt):
    assert info['token']==truth['sample_token']
    by_class={c:[v for v in truth['vectors'] if v['cls_name']==c] for c in classes}
    for c in classes:
        expected=info['annotation'][c];actual=by_class[c]
        assert len(expected)==len(actual),(info['token'],c)
        for pts,v in zip(expected,actual):
            assert v['type']==classes.index(c)
            assert np.array_equal(np.asarray(pts),np.asarray(v['pts'])),(info['token'],c)
            arrays+=1
        counts[c]+=len(actual)
    det=result['pts_bbox']
    for score,label in zip(det['scores_3d'],det['labels_3d']):scores[classes[int(label)]].append(float(score))
checkpoint_details={}
for name in ['maptrv2_umpe','maptrv2_umpe_ft']:
    ck=torch.load(str(R/'weights/umpe'/(name+'.pth')),map_location='cpu',mmap=True)
    checkpoint_details[name]={'epoch':ck['meta']['epoch'],'iter':ck['meta']['iter'],'training_history_not_this_evaluation':ck['meta'].get('hook_msgs'),'alpha_parameters':{k:v.tolist() for k,v in ck['state_dict'].items() if 'alpha' in k and v.numel()<5}}
    del ck
report=dict(frames=6019,all_gt_tokens_in_official_dataset_order=True,all_gt_arrays_equal_converter_annotations=True,arrays_checked=arrays,ground_truth_counts=dict(counts),prediction_scores={c:{'count':len(s),'above_0_3':int(np.sum(np.array(s)>=.3)),'quantiles':np.quantile(s,[0,.25,.5,.75,1]).tolist()} for c,s in scores.items()},checkpoint_details=checkpoint_details)
(O/'umpe_label_audit.json').write_text(json.dumps(report,indent=2))
indices=np.linspace(0,6018,6,dtype=int).tolist()
fig,axes=plt.subplots(2,6,figsize=(16,11),constrained_layout=True)
for n,i in enumerate(indices):
    left,right=axes[n//3,(n%3)*2:(n%3)*2+2]
    for v in gt[i]['vectors']:
        pts=np.array(v['pts']);left.plot(pts[:,0],pts[:,1],color=colors[v['type']],lw=1.2)
    p=pred[i]['pts_bbox']
    for pts,s,label in zip(p['pts_3d'],p['scores_3d'],p['labels_3d']):
        if float(s)>=.3:
            xy=pts.numpy();right.plot(xy[:,0],xy[:,1],color=colors[int(label)],lw=1.2)
    for ax,title in [(left,'GT '+infos[i]['token'][:7]),(right,'UMPE ft')]:
        ax.set(xlim=(-15,15),ylim=(-30,30),aspect='equal',title=title,xlabel='x (m)',ylabel='y (m)')
        ax.scatter([0],[0],c='black',marker='^',s=16);ax.grid(alpha=.15)
fig.suptitle('Full validation: six equally spaced examples\nDisplay score >= 0.3; official AP uses every prediction',fontsize=14)
fig.legend([Line2D([0],[0],color=c) for c in colors],classes,loc='outside lower center',ncol=3)
fig.savefig(O/'umpe_examples.png',dpi=170)
(O/'umpe_examples_selection.json').write_text(json.dumps({'indices':indices,'tokens':[infos[i]['token'] for i in indices]},indent=2))
print(json.dumps(report,indent=2))
