"""Match the configuration embedded in the authors' released checkpoints.

Model and train/test pipelines are identical to the repository example.
These two training settings differ in both released checkpoint configurations.
"""
_base_ = '/root/autodl-tmp/manner-repro/repos/UMPE/Maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_umpe.py'
optimizer = dict(lr=6e-5)
custom_hooks = [
    dict(type='AlphaScheduleHook',start_iter=0,end_iter=20000,start_alpha=0.0,target_alpha=0.6,schedule_type='cosine'),
    dict(type='FusionAlphaMonitorHook',log_interval=100),
]
