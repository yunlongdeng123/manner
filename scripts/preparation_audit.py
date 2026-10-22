"""Final audit of the no-GPU preparation stage, with explicit GPU exclusions."""
import hashlib,json,pathlib,subprocess,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def read(name):return json.loads((R/'manifests'/name).read_text())
for name in ('topogpt','umpe'):
    assert (R/'manifests'/(name+'_deps_ready')).exists()
    assert 'No broken requirements found.' in (R/'manifests'/(name+'_pip_check.txt')).read_text()
assert (R/'manifests/umpe_sdk_ready').exists()
inventory=read('checkpoints_inventory.json');assert len(inventory)==4
for row in inventory.values():assert row['cpu_mmap_load'] and row['full_state']
assert all(row['zip_crc_verified'] for row in read('umpe_checkpoint_zip_crc.json').values())
for name,size in {'topogpt/pretrain.ckpt':1257462106,'topogpt/finetune.ckpt':1596822246,'umpe/maptrv2_umpe.pth':664671899,'umpe/maptrv2_umpe_ft.pth':664671899}.items():
    assert (R/'weights'/name).stat().st_size==size
for name in ('OpenLane-V2_sample.tar','OpenLane-V2_subset_A_info.tar','OpenLane-V2_subset_A_image_7.tar'):
    assert read('verified_'+name+'.json')['md5']
topo=read('openlane_coverage.json');assert topo['frames']==4806 and topo['images']==33642 and topo['missing_images']==0 and topo['official_split_exact_match']
smoke=read('openlane_smoke_coverage.json');assert smoke['train']['frames']==64 and smoke['val']['frames']==8
assert read('collect_equivalence.json')['all_keys_types_arrays_equal']
assert read('topogpt_loader_probe.json')['train'] and read('topogpt_loader_probe.json')['val']
nusc=read('nuscenes_coverage.json');assert nusc['val_samples']==6019 and not nusc['missing']
umpe=read('umpe_infos_ready.json');assert umpe['val_frames']==6019 and umpe['train_smoke_frames']==40 and umpe['val_scenes']==150
assert read('umpe_converter_probe.json')['geometry_cache_exact_match']
assert all(not v['missing'] for v in read('umpe_prior_keys_verified.json').values())
assert read('umpe_smoke_selection.json')['real_missing_raster_frames']==4
assert len(list((R/'data/umpe_info_scenes').glob('*.pkl')))==151
for name,row in read('umpe_config_audit.json').items():assert not row['model'] and not row['test_pipeline']
sources={}
for repo in ('TopoGPT','UMPE','OpenLane-V2'):
    path=R/'repos'/repo
    commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    diff=subprocess.check_output(['git','-C',str(path),'diff','--stat'],text=True).strip()
    sources[repo]=dict(commit=commit,tracked_changes=diff)
papers={}
for p in sorted((R/'papers').glob('*.pdf')):
    with p.open('rb') as f:assert f.read(5)==b'%PDF-'
    papers[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert len(papers)==2
gpu=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True)
report=dict(no_gpu_preparation_ready=True,time=time.strftime('%Y-%m-%d %H:%M:%S %z'),sources=sources,papers=papers,topogpt_validation_frames=4806,umpe_validation_frames=6019,full_training_prepared=False,formal_training_started=False,gpu_smoke_executed=False,cuda_extensions_verified=False,nvidia_smi=gpu.stdout.strip() or gpu.stderr.strip(),pending=['UMPE CUDA-extension build and actual CUDA-operator execution','Actual model strict loading and approximately ten-minute GPU smokes','Official full-validation inference, metrics comparison and final experiment report'])
(R/'manifests/no_gpu_preparation_ready.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
