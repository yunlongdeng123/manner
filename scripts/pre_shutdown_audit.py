"""Check result retrieval and absence of experiment jobs; this script never shuts down."""
import datetime,json,os,pathlib,shutil,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro');S=pathlib.Path('/root/manner-repro-shutdown-state');S.mkdir(exist_ok=True)
assert not (S/'cache_transfer_pending.json').exists(),'Scoring-cache retrieval is still pending'
receipt=json.loads((S/'local_delivery_receipt.json').read_text())
head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
assert receipt['commit']==head
for key in ['artifact_hashes','source_archive_hashes','full_prediction_hashes','evaluation_input_hashes','git_fsck']:assert receipt[key],key
ready=json.loads((R/'manifests/delivery_ready.json').read_text())
assert set(ready['runs'])=={'topogpt','umpe_base','umpe_finetuned','maptrv2_control'}
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU compute processes remain: '+gpu
active=[];supervisors=[]
for d in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        pid=int(d.name);comm=(d/'comm').read_text().strip()
        if pid==os.getpid():continue
        args=(d/'cmdline').read_bytes().decode(errors='replace').split('\0')
        cwd=(d/'cwd').resolve()
        if comm=='supervisord':supervisors.append(dict(pid=pid,command=' '.join(args).strip()))
        if comm.startswith('python'):
            active.append(dict(pid=pid,comm=comm,executable=args[0],script=next((x for x in args[1:] if x.endswith('.py')),None),cwd=str(cwd)))
        elif comm in ['bash','sh'] and cwd.is_relative_to(R) and not any('pre_shutdown_audit.py' in x for x in args):
            script=next((x for x in args[1:] if x.endswith('.sh')),None)
            if script:active.append(dict(pid=pid,comm=comm,script=script,cwd=str(cwd)))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert not active,'Other Python or experiment shell processes require inspection: '+json.dumps(active)
assert len(supervisors)==1 and '/init/supervisor/supervisor.ini' in supervisors[0]['command']
schedules={}
for name,args in [('crontab',['crontab','-l']),('atq',['atq'])]:
    if shutil.which(name):
        p=subprocess.run(args,capture_output=True,text=True)
        schedules[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        assert not p.stdout.strip(),'Pending scheduled entries require inspection: '+name
    else:schedules[name]=dict(installed=False)
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=head,local_delivery_verified=True,gpu_mode_previously_confirmed='RTX 3090 actual GPU smokes and full inference',active_gpu_processes=[],other_experiment_processes=active,schedules=schedules,supervisor=supervisors[0],ready_for_authorized_shutdown=True,action='SIGTERM to the verified AutoDL supervisord, matching the platform shutdown mechanism without Trash deletion')
(S/'pre_shutdown_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
