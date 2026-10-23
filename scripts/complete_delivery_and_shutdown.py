"""Finish the authorized existing run, verify local delivery, then stop AutoDL."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys,time
HOST='manner-3090-1023'
ROOT='/root/autodl-tmp/manner-repro'
STATE='/root/manner-repro-shutdown-state'
REMOTE_DELIVERY='/root/manner-repro-gpu-delivery'
HERE=pathlib.Path(__file__).resolve().parent
DEST=pathlib.Path('C:/Users/dengyunlong/Desktop/工作文件整理/nio/MANNER_TopoGPT_UMPE_复现结果_20261024')
DEST.mkdir(exist_ok=True)

def event(stage,**details):
    row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**details)
    with (DEST/'closure_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)

def remote(command,timeout=60,check=True):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command],capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if check and p.returncode:raise RuntimeError(f'SSH command failed ({p.returncode}): {p.stdout}\n{p.stderr}')
    return p

def remote_python(source,timeout=60,check=True):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'/root/miniconda3/bin/python','-'],input=source,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if check and p.returncode:raise RuntimeError(f'Remote Python failed ({p.returncode}): {p.stdout}\n{p.stderr}')
    return p

def scp(source,target):subprocess.run(['scp','-o','BatchMode=yes',str(source),str(target)],check=True,timeout=600)

event('waiting_for_existing_evaluation')
while True:
    p=remote(f'test -f {ROOT}/manifests/final_reporting_complete.json',check=False)
    if p.returncode==0:break
    if p.returncode!=1:raise RuntimeError('Could not check remote reporting status: '+p.stderr)
    # Do not silently wait if the report controller has already failed.
    alive=remote('test -d /proc/23121',check=False)
    if alive.returncode!=0:raise RuntimeError('Report controller stopped; inspect logs/final_reporting.log')
    time.sleep(30)

event('report_ready')
report=remote(f'cat {ROOT}/reports/reproduction_report.md').stdout
(DEST/'report_preview.md').write_text(report,encoding='utf-8')
summary=json.loads(remote(f'cat {ROOT}/reports/results.json').stdout)
event('final_metrics',metrics={k:v['metrics'] for k,v in summary['runs'].items()})

# The verified report/controller only creates these known final artifacts.
paths=['README.md','manifests/final_status.json','manifests/final_status_no_gpu_snapshot.json','manifests/evaluation_inputs.json','manifests/delivery_ready.json','manifests/final_reporting_complete.json','reports','scripts/complete_delivery_and_shutdown.py']
command=f'cd {ROOT} && git diff --check && git add '+ ' '.join(shlex.quote(x) for x in paths)+" && (git diff --cached --quiet || git commit -m 'Finalize complete validation results and verified delivery evidence') && git diff --exit-code && git diff --cached --exit-code"
event('git_commit',output=remote(command).stdout)
event('packaging')
p=remote(f'cd {ROOT} && /root/miniconda3/bin/python scripts/package_gpu_results.py',timeout=900)
event('package_created',output=p.stdout)
scp(f'{HOST}:{REMOTE_DELIVERY}/delivery_manifest.json',DEST/'delivery_manifest.json')
manifest=json.loads((DEST/'delivery_manifest.json').read_text())
for name,entry in manifest['files'].items():
    path=(DEST/name).resolve();assert path.is_relative_to(DEST.resolve())
    if entry.get('prestaged'):
        assert path.is_file(),name
        continue  # Full hashes are checked below with the other delivery files.
    event('retrieving',file=name,bytes=entry['bytes'])
    if name.endswith('.tgz'):
        subprocess.run([sys.executable,'-X','utf8',str(HERE/'parallel_ssh_fetch.py'),'--remote',REMOTE_DELIVERY+'/'+name,'--output',str(path),'--bytes',str(entry['bytes']),'--sha256',entry['sha256'],'--streams','4'],check=True)
    else:scp(f'{HOST}:{REMOTE_DELIVERY}/{name}',path)
event('verifying_local_delivery')
subprocess.run([sys.executable,'-X','utf8',str(HERE/'unpack_verify_gpu_results.py'),str(DEST)],check=True)
receipt=json.loads((DEST/'verification.json').read_text())
assert receipt['commit']==manifest['commit']
event('local_delivery_verified',commit=receipt['commit'])
remote(f'mkdir -p {STATE}')
scp(DEST/'verification.json',f'{HOST}:{STATE}/local_delivery_receipt.json')
event('pre_shutdown_audit')
audit=remote(f'cd {ROOT} && /root/miniconda3/bin/python scripts/pre_shutdown_audit.py').stdout
(DEST/'pre_shutdown_audit.json').write_text(audit,encoding='utf-8')
audit=json.loads(audit);assert audit['ready_for_authorized_shutdown']
pid=audit['supervisor']['pid'];expected=audit['supervisor']['command']
record=dict(target=HOST,requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=receipt['commit'],local_delivery_verified=True,pre_shutdown_audit_passed=True,mechanism='Platform supervisord SIGTERM, without deleting user Trash',supervisor_pid=pid)
(DEST/'shutdown_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
event('requesting_shutdown',supervisor_pid=pid)
source=f'''import datetime,json,os,pathlib,signal,subprocess
pid={pid!r}
args=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().split('\\0')
assert ' '.join(args).strip()=={expected!r}
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,gpu
record={record!r}
pathlib.Path({STATE!r}+'/shutdown_requested.json').write_text(json.dumps(record,indent=2))
message='['+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'] user execute shutdown command in container!\\n'
with open('/proc/1/fd/1','w') as f:f.write(message);f.flush()
os.kill(pid,signal.SIGTERM)
print('SHUTDOWN_SIGNAL_SENT',flush=True)
'''
p=remote_python(source,check=False)
record.update(command_returncode=p.returncode,command_stdout=p.stdout,command_stderr=p.stderr,signal_sent='SHUTDOWN_SIGNAL_SENT' in p.stdout)
probes=[]
for _ in range(3):
    time.sleep(5)
    probe=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ConnectionAttempts=1',HOST,'true'],capture_output=True,text=True,encoding='utf-8',timeout=15)
    probes.append(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=probe.returncode,stderr=probe.stderr))
record.update(ssh_probes=probes,ssh_unreachable=all(x['returncode']!=0 for x in probes),billing_control_plane_verified=False)
(DEST/'shutdown_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
assert record['signal_sent'] and record['ssh_unreachable'],record
tm=summary['runs']['topogpt']['metrics'];um=summary['runs']['umpe_finetuned']['metrics'];base=summary['runs']['umpe_base']['metrics']
readme=f'''# TopoGPT 与 UMPE 复现交付

已通过 AutoDL 实例提供的关机机制发送 SIGTERM，随后三次 SSH 检查均不可达。详细记录见 [关机记录](shutdown_record.json)。未通过账单控制台独立核验计费状态。

[完整复现报告](records/reports/reproduction_report.md) · [本地完整性核验](verification.json) · [原始结果汇总](records/reports/results.json)

![架构组件与数据流](records/reports/figures/architecture_components.png)

| 项目 | 实测 |
|---|---:|
| TopoGPT M-P / M-R / M-T | {tm['pr']['PR_mean'][0]*100:.2f} / {tm['pr']['PR_mean'][1]*100:.2f} / {tm['topo']['Topo mAP']*100:.2f} |
| UMPE 基础 / 微调 mAP | {base['NuscMap_chamfer/mAP']*100:.2f} / {um['NuscMap_chamfer/mAP']*100:.2f} |
| MapTRv2 诊断对照 mAP | {summary['diagnostic_control']['metrics']['NuscMap_chamfer/mAP']*100:.2f} |

UMPE 尚未达到论文参考的 67.4 mAP。本次完成公开权重的完整验证和训练、保存、恢复 smoke，未进行全量重新训练。

`records/` 保存论文、代码归档、预测、评分标注、指标、日志与报告；`deployment/` 是经 Git fsck 验证的部署代码克隆，commit `{receipt['commit']}`。远端无配置 Git remote，使用已验证的 bundle 与本地克隆备份。

原始图像、地图先验、模型权重、Python 环境和训练恢复现场仍在远端 `/root/autodl-tmp/manner-repro` 及报告所列系统盘路径，关机未删除这些内容。
'''
(DEST/'README.md').write_text(readme,encoding='utf-8')
event('complete',shutdown_signal_sent=True,ssh_unreachable=True,report=str(DEST/'records/reports/reproduction_report.md'))
