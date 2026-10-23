"""Mark completion from verified outputs, preserving the no-GPU snapshot."""
import datetime,hashlib,json,pathlib,shutil
R=pathlib.Path('/root/autodl-tmp/manner-repro');M=R/'manifests'
summary=json.loads((R/'reports/results.json').read_text())
assert set(summary['runs'])=={'topogpt','umpe_finetuned','umpe_base'}
for row in list(summary['runs'].values())+[summary['diagnostic_control']]:
    assert row['coverage']['full_validation_coverage']
    assert row['coverage']['all_numeric_outputs_finite']
old=M/'final_status.json';backup=M/'final_status_no_gpu_snapshot.json'
if not backup.exists():shutil.copy2(old,backup)
state=json.loads(old.read_text())
state.update(phase='published_checkpoint_validation_complete',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_smoke_passed={'topogpt':True,'umpe':True},full_validation={name:{'directory':row['directory'],'frames':row['coverage']['frames'],'sha256':row['coverage']['sha256']} for name,row in summary['runs'].items()},formal_training_started=False,report='/root/autodl-tmp/manner-repro/reports/reproduction_report.md')
state['free_GiB']={p:round(shutil.disk_usage(p).free/2**30,2) for p in ['/root',str(R)]}
state['diagnostic_control']=summary['diagnostic_control']
state['cpu_preparation']['phase']='complete'
old.write_text(json.dumps(state,ensure_ascii=False,indent=2))
readme=R/'README.md';text=readme.read_text()
text=text.replace('2026-10-23 的 GPU 阶段已完成两套 smoke，目前正在运行完整验证；最终分数以结果文件及覆盖核验为准。','2026-10-23 启动的 GPU 阶段已完成两套 smoke，以及 TopoGPT 和 UMPE 两份权重的完整验证。实际分数与口径见 [复现结果报告](reports/reproduction_report.md)，逐帧覆盖、哈希和原始指标均已保存。')
text=text.replace('全量预测和指标仍需完成后核对，不将 smoke 分数当作完整验证结果。','全量预测和指标已核对完成；smoke 分数仅用于入口验收。正式 UMPE 两份权重在本轮采用独立进程并行评测，重跑入口采用顺序运行，每次日志保存在各自独立结果目录。')
readme.write_text(text,encoding='utf-8')
inputs={}
for label,relative in [('topogpt_val.pkl','data/topogpt/openlane_subA/data_dict_subset_A_val.pkl'),('umpe_val.pkl','data/nuscenes/nuscenes_map_infos_temporal_val.pkl'),('umpe_map_anns_val.json','data/nuscenes/nuscenes_map_anns_val.json'),('openlane_split.json','data/openlanev2/data_dict_subset_A.json')]:
    path=R/relative;h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    inputs[label]={'source':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
(M/'evaluation_inputs.json').write_text(json.dumps(inputs,indent=2))
print(json.dumps(state['full_validation'],indent=2))
