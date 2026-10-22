"""Serialize memory-heavy CPU preparation under the 2 GiB no-GPU limit."""
import json,os,pathlib,signal,subprocess,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
STATE=R/'manifests/cpu_prepare_state.json'
def state(phase,**kw):
    data=dict(phase=phase,time=time.strftime('%Y-%m-%d %H:%M:%S %z'),**kw)
    STATE.write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
def full_umpe_pids():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if str(R/'scripts/prepare_umpe_infos.py').encode() in args and b'--probe' not in args:found.append(int(p.name))
    return found
def map_generator_running():
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if str(R/'repos/TopoGPT/MapPrior/datasets/openlanev2/openlane_map_gen.py').encode() in args:return True
    return False
def resume_umpe():
    if (R/'manifests/umpe_infos_ready.json').exists() or full_umpe_pids():return
    with (R/'logs/umpe_infos_after_openlane.log').open('ab') as log:
        subprocess.Popen(['/root/miniconda3/envs/umpe/bin/python',str(R/'scripts/prepare_umpe_infos.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})

state('waiting_for_verified_openlane_images')
while not (R/'manifests/verified_OpenLane-V2_subset_A_image_7.tar.json').exists():time.sleep(10)
while not (R/'manifests/umpe_prior_keys_verified.json').exists():time.sleep(10)
pids=full_umpe_pids()
state('preparing_topogpt',paused_umpe_pids=pids)
for pid in pids:os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+30
while full_umpe_pids() and time.monotonic()<deadline:time.sleep(0.5)
assert not full_umpe_pids(),'UMPE worker did not stop; refusing overlapping large loads'
while map_generator_running():time.sleep(5)
success=False
try:
    for flag,marker in [([],R/'manifests/openlane_coverage.json'),(['--smoke'],R/'manifests/openlane_smoke_coverage.json')]:
        if marker.exists():continue
        subprocess.run(['/root/miniconda3/envs/topogpt/bin/python',str(R/'scripts/prepare_openlane.py'),*flag],check=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    subprocess.run(['/root/miniconda3/envs/topogpt/bin/python',str(R/'scripts/probe_topogpt_data.py')],check=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(R/'repos/TopoGPT')})
    state('topogpt_data_ready')
    success=True
finally:
    if success:resume_umpe()
    else:state('topogpt_preparation_failed_preserved_for_recovery')
state('topogpt_ready_umpe_continuing')
