import pathlib,sys
out=pathlib.Path(sys.argv[1])
cfg="_base_ = '/root/autodl-tmp/manner-repro/configs/umpe_reproduce.py'\n"
cfg+="data = dict(workers_per_gpu=0, test=dict(ann_file='data/nuscenes/nuscenes_map_infos_temporal_val_smoke_diverse.pkl', map_ann_file="+repr(str(out/'gt_smoke.json'))+"))\n"
root=pathlib.Path('/root/autodl-tmp/manner-repro')
if (root/'manifests/umpe_selected_priors_verified.json').exists():
    base=root/'data/umpe_priors/selected'
    cfg+="data['test'].update(sdmap_pkl="+repr(str(base/'sd_val_smoke.pkl'))+",hdmap_pkl="+repr(str(base/'hd_val_smoke.pkl'))+")\n"
(out/'eval_config.py').write_text(cfg)
