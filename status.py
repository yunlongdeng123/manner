"""Small, read-only preparation status; no torch imports or GPU allocation."""
import json,pathlib,shutil
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def read(name):
    p=R/'manifests'/name
    return json.loads(p.read_text()) if p.exists() else None
report={
 'root':str(R),
 'environments':{n:(R/'manifests'/(n+'_deps_ready')).exists() for n in ('topogpt','umpe')},
 'checkpoint_inventory':list((read('checkpoints_inventory.json') or {}).keys()),
 'openlane_archives_verified':{n:bool(read('verified_'+n+'.json')) for n in ('OpenLane-V2_sample.tar','OpenLane-V2_subset_A_info.tar','OpenLane-V2_subset_A_image_7.tar')},
 'topogpt_data':read('openlane_coverage.json'),
 'topogpt_smoke_data':read('openlane_smoke_coverage.json'),
 'nuscenes_inputs':read('nuscenes_coverage.json'),
 'umpe_scenes_prepared':len(list((R/'data/umpe_info_scenes').glob('*.pkl'))),
 'umpe_scenes_expected':151,
 'umpe_infos':read('umpe_infos_ready.json'),
 'prior_keys':read('umpe_prior_keys_verified.json'),
 'cpu_preparation':read('cpu_prepare_state.json'),
 'gpu_smoke_passed':{n:bool(read(n+'_gpu_smoke_passed.json')) for n in ('topogpt','umpe')},
 'free_GiB':{p:round(shutil.disk_usage(p).free/1024**3,2) for p in ('/root',str(R))},
 'formal_training_started':False,
}
print(json.dumps(report,indent=2,ensure_ascii=False))
