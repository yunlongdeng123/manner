"""Build a report only after all three complete validation outputs exist."""
import datetime,hashlib,json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def read(p):return json.loads(p.read_text())
def latest(pattern):
    paths=sorted((R/'results').glob(pattern),key=lambda p:p.stat().st_mtime)
    assert paths,pattern
    return paths[-1]
top=latest('topogpt/*/metrics.json')
ft=latest('umpe/finetuned_*/predictions.metrics.json')
base=latest('umpe/base_*/predictions.metrics.json')
items={}
for name,path in [('topogpt',top),('umpe_finetuned',ft),('umpe_base',base)]:
    coverage=read(path.parent/'predictions.coverage.json')
    assert coverage['full_validation_coverage'] and coverage['all_numeric_outputs_finite']
    items[name]=dict(directory=str(path.parent),metrics=read(path),coverage=coverage)
tm=items['topogpt']['metrics']
toprows=[('M-P',tm['pr']['PR_mean'][0]*100,45.6),('M-R',tm['pr']['PR_mean'][1]*100,42.6),('M-T',tm['topo']['Topo mAP']*100,24.4)]
for label,key,ref in [('G-IoU','Graph IoU',54.4),('SDA','SDA',26.9),('G-F1','GEO F1',60.3),('T-F1','TOPO F1',47.4),('JT-F1','JTOPO F1',35.7),('APLS','APLS',28.7)]:
    assert tm['graph'][key] is not None,key
    toprows.append((label,tm['graph'][key]*100,ref))
umrows=[]
for label,key,ref in [('人行横道','ped_crossing_AP',66.6),('分隔线','divider_AP',67.2),('道路边界','boundary_AP',68.2),('mAP','mAP',67.4)]:
    umrows.append((label,items['umpe_base']['metrics']['NuscMap_chamfer/'+key]*100,items['umpe_finetuned']['metrics']['NuscMap_chamfer/'+key]*100,ref))
smokes={k:read(R/'manifests'/(k+'_gpu_smoke_passed.json')) for k in ['topogpt','umpe']}
summary=dict(completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),scope='Published checkpoint full validation, plus separately budgeted real-GPU engineering smokes; not from-scratch training reproduction',runs=items,smokes=smokes,topogpt_comparison=toprows,umpe_comparison=umrows,git_commit_before_report=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip())
report=R/'reports';report.mkdir(exist_ok=True)
(report/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
md=['# TopoGPT 与 UMPE：3090 复现结果','',f"完成时间：{summary['completed_at']}。主机 `manner-3090-1023`；工作目录 `/root/autodl-tmp/manner-repro`。",'',
'已完成作者发布权重在完整验证集上的运行、官方指标计算及结果覆盖检查。两套训练 smoke 均覆盖真实数据、前后向、有限 loss/非零梯度、checkpoint 保存、独立进程完整状态恢复和评价入口。未进行从头预训练或全量重新微调。','',
'## 结构与输入','','![架构组件与数据流](figures/architecture_components.png)','',
'TopoGPT 使用图像推理，地图先验已学入权重；本次 UMPE 配置额外读取四类地图先验。两个任务的输入、输出及指标不同，分数不能相互排名。','',
'TopoGPT 的官方评测将生成的中心线按默认 0.5 m 端点距离构建有向连接；连接指标包含生成几何与这一构图步骤的共同效果。UMPE 本次配置 `queue_length=1`、`video_test_mode=False`。本轮未测试长时序在线地图跟踪、SLAM/VIO 或定位与地图联合优化。','',
'## TopoGPT：OpenLane-V2 subset A','',
'完整 4,806 帧、7 路图像。预测键与官方划分逐一相等；使用作者提供的 PR、topology、graph 指标。下表均乘以 100，差值为百分点。参考来自 [论文表 1](https://arxiv.org/html/2606.31814v1)。','',
'| 指标 | 本次实测 | 论文参考 | 差值 |','|---|---:|---:|---:|']
md += [f'| {n} | {v:.2f} | {ref:.1f} | {v-ref:+.2f} |' for n,v,ref in toprows]
tc=items['topogpt']['coverage']
md += ['',f"诊断：{len(tc.get('no_eos_frames',[]))} 帧在默认生成上限内未输出 EOS；{len(tc.get('empty_lane_frames',[]))} 帧最终中心线为空。保留作者默认生成上限和原始预测，不因这些情况筛除验证帧。对应帧键见 `predictions.coverage.json`。"]
md += ['', '## UMPE：nuScenes MapTRv2 + 四类先验','',
'两份 checkpoint 均完成官方 150 个验证场景、6,019 帧，采用原测试代码的单 rank、无打乱顺序。Chamfer AP 取 0.5 / 1.0 / 1.5 m 阈值平均。参考来自 [论文 MapTRv2 主表](https://arxiv.org/html/2605.02762v1)。','',
'| 指标 | 基础权重 | 微调权重 | 论文参考 |','|---|---:|---:|---:|']
md += [f'| {n} | {b:.2f} | {f:.2f} | {ref:.1f} |' for n,b,f,ref in umrows]
md += ['',
'论文主表标注 24 epochs；发布的基础/微调 checkpoint 元数据分别为 epoch 29/44，内嵌训练配置为 72 epochs。因此本次是公开权重的实测复现，不能声称复现了论文的严格训练预算。原始栅格 SD 包缺少验证集 16 帧，保留官方白图回退，未删除这些帧或补造先验。','',
'## 真实 GPU 验收','',
'| 项目 | TopoGPT | UMPE |','|---|---|---|',
f"| 训练 smoke | {smokes['topogpt']['train']['end_step']} 步 / {smokes['topogpt']['train']['seconds']:.1f} 秒 | {smokes['umpe']['train']['end_iter']} 步 / {smokes['umpe']['train']['seconds']:.1f} 秒 |",
'| 独立进程恢复 | 440 → 443 步 | 320 → 360 步 |',
'| 恢复校验 | 模型、优化器、调度器、步数、RNG | 模型、优化器、epoch/iter、FP16 scaler、RNG、下一步学习率 |',
'| 小规模评价 | 8 帧，三组官方指标 | 8 帧，其中 4 帧缺失栅格先验 |',
'', 'RNG 已恢复，但未断言跨进程数据顺序和整段训练轨迹逐位一致。smoke 权重不用于完整验证。GKT 与 MMCV 注意力 CUDA 算子的前向、反向对照误差均小于 1e-6。','',
'## 工程处理与可复查证据','',
'- TopoGPT 官方源码未修改，checkpoint 内保存的采样/损失/优化器/调度配置与仓库一致。',
'- UMPE 源码差异仅为结果持久化、旧依赖声明和 Numba 异常类导入，见 `patches/umpe_engineering.patch`。',
'- 修复了 smoke 审计中 RNG 元数据 JSON 序列化和旧 PyTorch 字节视图兼容问题；从原始第 320 步 checkpoint 恢复，未重复该训练段。失败现场与日志保留。',
'- 全量评测最初遇到旧数据集 `dict_keys` 无法经 spawn 序列化；使用与 smoke 相同的 `workers_per_gpu=0` 后运行。模型、输入及评分未更改。',
'- 先验按帧键无损缓存。验证集 21,050 个 HD 数组、945,554 个 SD 数组保存前后哈希一致；原始作者文件保留。',
'- 两套任务部分时间共享 GPU，记录了资源遥测；耗时不作为单模型 FPS 基准。',
'- UMPE 两份权重最终并行评价，等待队列父进程已停止，未中断任何模型进程或重复执行已完成帧。OpenCV 单线程在 8 个真实样本上与默认 112 线程输出逐字节一致，数据管线约快 28%；考虑已运行进度，本轮未重启或更改正式进程，证据见 `manifests/opencv_pipeline_equivalence.json`。',
'- TopoGPT 在 8 个真实样本上的额外诊断显示，教师强制损失约占计算时间 14%；本轮保留原始验证入口。该测量与 CPU 线程对比均为额外工程开销，未计入正式指标样本。',
'- `scripts/score_umpe_predictions.py` 直接调用官方格式化与评分函数，读取保存的预测、帧元数据和真值。8 帧 CPU 回放全部指标与原官方输出完全一致，未初始化 CUDA、加载模型、图像或先验。',
'', '## 重新运行','',
'```bash','cd /root/autodl-tmp/manner-repro','bash scripts/topogpt_eval.sh','bash scripts/umpe_eval_pair.sh','/root/miniconda3/bin/python scripts/runtime_status.py','```','',
'环境分别为 `topogpt`（Python 3.10 / torch 2.1.0+cu118）和 `umpe`（Python 3.8 / torch 1.9.1+cu111 / MMCV 1.4.0）。入口脚本自动选择环境与权重，重复运行使用独立结果目录。','',
'完整结果路径：','']
md += [f"- `{name}`：`{row['directory']}`，预测 SHA-256 `{row['coverage']['sha256']}`。" for name,row in items.items()]
md += ['', '只重算 UMPE 指标，无需 GPU 推理：','', '```bash','CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\', '/root/miniconda3/envs/umpe/bin/python scripts/score_umpe_predictions.py \\', f"  --pred {ft.parent}/predictions.pkl \\", '  --infos data/nuscenes/nuscenes_map_infos_temporal_val.pkl \\', '  --gt data/nuscenes/nuscenes_map_anns_val.json \\', '  --out-dir results/metric_replay/umpe_finetuned_cpu \\', f'  --reference {ft}', '```','', '为保护旧结果，CPU 回放的 `--out-dir` 必须是尚不存在的目录。离线交付包在 `evaluation_inputs/` 中包含验证标注与帧元数据；原始图像、先验、权重和环境仍保留于远端。']
md += ['', '后续需要独立准备/预算的项目：全量重新训练、TopoGPT 3.3M 地图预训练、OpenLane subset B、UMPE 的 AV2/MapQR/VAD 与先验消融。上述项目未作为本次已完成结果。','']
md += ['## 定性核对','',
'TopoGPT 固定随机种子选场景、每场景取首帧；若存在未生成 EOS 的帧，最后一例展示其中一帧。UMPE 在完整验证序列上等间隔取 6 帧，仅为显示清晰采用 0.3 分数阈值；正式指标保留全部预测。精确帧键及选择规则在 `figures/selection.json`，这些例子不替代总体统计。','',
'![TopoGPT 真值与预测](figures/topogpt_examples.png)','',
'![UMPE 真值与预测](figures/umpe_examples.png)','']
(report/'reproduction_report.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(dict(report=str(report/'reproduction_report.md'),results=str(report/'results.json')),ensure_ascii=False))
