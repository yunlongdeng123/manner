# GPU 实例复现与实验边界

2026-09-24 在 AutoDL 新实例上已完成官方 TopoLogic checkpoint 的真实 GPU 推理、冻结特征导出和轻量头试训。下述路径是该实例的数据位置；仓库只保存程序与配置，不保存图像、权重或特征缓存。

## 实例与依赖

| 组件 | 已验证版本 |
| --- | --- |
| GPU | NVIDIA RTX 4080 SUPER，32 GB 显存 |
| Python / CUDA PyTorch | 3.10.8 / `torch==2.1.2+cu118`、`torchvision==0.16.2+cu118` |
| MMCV | `mmcv-full==1.7.2`，使用 OpenMMLab `cu118/torch2.1.0` 预编译 wheel；CUDA NMS 算子已实测 |
| 检测框架 | `mmdet==2.26.0`、`mmsegmentation==0.29.1`、`mmdet3d==1.0.0rc6`、`mmcls==0.24.1` |
| 数据与评测库 | OpenLane-V2 devkit `v2.1.0`、`scipy==1.11.4`、`shapely==2.0.7`、`ortools==9.3.10497`、`similaritymeasures==0.6.0` 等 |

GPU 环境位于 `/root/autodl-tmp/envs/topologic-gpu`，用 `venv --system-site-packages` 继承实例镜像里的 GPU PyTorch；CPU 环境 `/root/autodl-tmp/envs/scenariotopo` 保持独立。在同一 AutoDL 基础镜像上可运行 `bash tools/bootstrap_topologic_gpu.sh`，额外包版本由 `requirements-topologic-gpu.txt` 锁定。TopoLogic 官方锁定的 Python 3.8 / PyTorch 1.9.1 / CUDA 11.1 栈不适合这张 Ada GPU。这里采用已实测推理的兼容栈，但不把它等同于官方原环境复现。

仅在隔离 venv 内做了两处兼容补丁，工具为 `tools/patch_topologic_gpu_compat.py`：允许 `mmdet3d` 的版本检查接受 MMCV 1.7.2；为旧 MMCV scatter 调用传入 PyTorch 2.1 要求的 `torch.device`。原文件保留 `.orig`，TopoLogic 源码和 checkpoint 均未修改。其余 Python 包包括 `opencv-python-headless`、`pycocotools`、`pyquaternion`、`numba`、`scikit-learn`、`scikit-image`、`pandas`、`nuscenes-devkit`、`lyft-dataset-sdk`、`iso3166`。旧版 `ortools==9.2.9972` 无当前 Python 3.10 可用包，因此使用仍提供 `ortools.graph.pywrapgraph` 的 9.3.10497。

## 数据与运行

| 资源 | 实例路径 |
| --- | --- |
| OpenLane-V2 subset A 标注与已提取 val 图像 | `/root/autodl-tmp/datasets/openlanev2/OpenLane-V2` |
| TopoLogic 源码 | `/root/autodl-tmp/third_party/TopoLogic` |
| OpenLane-V2 devkit | `/root/autodl-tmp/third_party/OpenLane-V2` |
| 官方权重 | `/root/autodl-tmp/checkpoints/topologic_r50_8x1_24e_olv2_subset_A.pth` |
| 地理隔离划分 | `/root/autodl-tmp/ScenarioTopo/results/openlanev2_geo_split.jsonl` |

原始 val 共 4,806 帧、33,642 张相机图像；其中项目地理隔离划分的验证集为 **15 段、480 帧**。训练图像仍在 AutoDL 只读 AV2 tar 分片中，尚未整批导出训练缓存。当前轻量头试验使用原始 val 内地理隔离为 `train` 的 **121 段、122 帧**（每段按 32 帧间隔抽样）；验证场景不参与训练。

从仓库目录运行下列命令。若尚未有官方 pickle，先调用官方 `collect()` 包装器：

```bash
cd /root/autodl-tmp/ScenarioTopo
export PYTHONPATH=/root/autodl-tmp/ScenarioTopo/src:/root/autodl-tmp/third_party/OpenLane-V2:/root/autodl-tmp/third_party/TopoLogic
GPU_PY=/root/autodl-tmp/envs/topologic-gpu/bin/python
DATA=/root/autodl-tmp/datasets/openlanev2/OpenLane-V2
TOPO=/root/autodl-tmp/third_party/TopoLogic

$GPU_PY tools/prepare_topologic_annotations.py \
  --openlane-source /root/autodl-tmp/third_party/OpenLane-V2 \
  --data-root "$DATA" --split val --output-stem data_dict_subset_A_val

$GPU_PY tools/export_topologic_cache.py \
  --topologic-root "$TOPO" \
  --config "$TOPO/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py" \
  --checkpoint /root/autodl-tmp/checkpoints/topologic_r50_8x1_24e_olv2_subset_A.pth \
  --data-root "$DATA" --annotation "$DATA/data_dict_subset_A_val.pkl" \
  --manifest results/openlanev2_geo_split.jsonl \
  --output results/topologic_val_full --model-split val
```

`prepare_topologic_annotations.py` 拒绝覆盖已有 pkl。`export_topologic_cache.py` 支持 `--sample-every N` 做跨场景抽样，但完整评测不要抽样。每帧缓存保留 200×256 查询特征、200×11×3 预测中心线、置信度、官方 Hungarian 一对一 GT 匹配、GT 端点/拓扑和两种未阈值化拓扑分数。`semantic_topology_scores` 是 sigmoid 后的纯语义分数，范围 0–1；`base_topology_scores` 是 TopoLogic 原始语义＋几何求和分数，**可能超过 1**，因此不是概率。两者可用 `tools/evaluate_cached_heads.py --score-key ...` 分开评估。

```bash
$GPU_PY tools/train_cached_heads.py \
  --config configs/full.yaml \
  --index results/topologic_train_cross_scene/index.jsonl \
  --output results/checkpoints/full_pilot.pth \
  --device cuda --epochs 5 --batch-size 4

$GPU_PY tools/evaluate_cached_heads.py \
  --checkpoint results/checkpoints/full_pilot.pth \
  --index results/topologic_val_full/index.jsonl \
  --split val --device cuda --threshold 0.9 --output results/full_val.json
```

比较不同拓扑输出前，先用 `tools/calibrate_topology_threshold.py` 在 `model_split=train` 缓存上选阈值。当前 122 帧训练样本得到纯语义 0.20、TopoLogic 融合 0.75、学习型关系头 0.90。校准报告和完整 480 帧评测 JSON 均留在实例的 `results/`，不入 Git。已核对的主要数字见仓库 README；可用 `--score-key semantic_topology_scores` 与融合分数做配对比较。

这些指标只针对 **GT 匹配查询**，不是 OpenLane-V2 官方 `TOP_ll`，也不是完整检测召回。端点“过渡区侵入”仍是用切向量和连接段/分合流标签构造的代理指标；不能当作公司 P1 区域标注的精确复现。GT 端点替换实验只能作为 oracle，不能加入实际模型推理。
