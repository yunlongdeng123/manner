"""Official image-only MapTRv2 control; priors only satisfy dataset initialization."""
_base_ = '/root/autodl-tmp/manner-repro/repos/UMPE/Maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py'
data = dict(workers_per_gpu=0,
    val=dict(sdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/sd_val.pkl',hdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/hd_val.pkl'),
    test=dict(sdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/sd_val.pkl',hdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/hd_val.pkl'))
