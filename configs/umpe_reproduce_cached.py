"""Same validation frames and tensors, using verified selected-prior caches."""
_base_ = '/root/autodl-tmp/manner-repro/configs/umpe_reproduce.py'
data = dict(
    val=dict(
        sdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/sd_val.pkl',
        hdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/hd_val.pkl'),
    test=dict(
        sdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/sd_val.pkl',
        hdmap_pkl='/root/autodl-tmp/manner-repro/data/umpe_priors/selected/hd_val.pkl'))
