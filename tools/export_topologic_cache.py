#!/usr/bin/env python3
"""Export frozen TopoLogic query tensors from OpenLane-V2 subset A.

Run from any directory with the TopoLogic repository supplied explicitly. The
official checkpoint and upstream model remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scenariotopo.data.cache_writer import FeatureCacheWriter


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologic-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True, help="Official collect() pickle")
    parser.add_argument("--manifest", type=Path, required=True, help="Geographic split JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-split", choices=("train", "val", "test"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 exports every selected frame")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Keep every Nth frame within each segment for a cross-scene pilot")
    parser.add_argument("--resume", action="store_true", help="Skip frames already recorded in output index")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["frame_key"]] = record
    return records


def frame_key(info: dict, split: str) -> str:
    return f"{split}/{info['segment_id']}/{info['timestamp']}"


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("TopoLogic checkpoint export requires a CUDA GPU")
    repo = args.topologic_root.resolve()
    sys.path.insert(0, str(repo))

    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from projects.topologic.core.lane.util import fix_pts_interpolate

    cfg = mmcv.Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.data.test.data_root = str(args.data_root.resolve())
    cfg.data.test.ann_file = str(args.annotation.resolve())
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    manifest = load_manifest(args.manifest)
    if args.model_split:
        dataset.data_infos = [
            info for info in dataset.data_infos
            if manifest.get(frame_key(info, dataset.split), {}).get("model_split") == args.model_split
        ]
    if args.start < 0 or args.start > len(dataset.data_infos):
        raise ValueError("--start is outside selected dataset")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be positive")
    if args.sample_every > 1:
        seen = {}
        sampled = []
        for info in dataset.data_infos:
            segment = info["segment_id"]
            ordinal = seen.get(segment, 0)
            if ordinal % args.sample_every == 0:
                sampled.append(info)
            seen[segment] = ordinal + 1
        dataset.data_infos = sampled
    dataset.data_infos = dataset.data_infos[args.start:]
    if args.limit:
        dataset.data_infos = dataset.data_infos[:args.limit]
    selected_count = len(dataset.data_infos)
    existing_index = args.output / "index.jsonl"
    if existing_index.exists() and existing_index.stat().st_size:
        if not args.resume:
            raise FileExistsError(f"{existing_index} exists; use --resume to skip completed frames")
        previous = load_manifest(existing_index)
        for record in previous.values():
            if not (args.output / record["cache_path"]).is_file():
                raise FileNotFoundError(f"Index points to missing cache: {record['cache_path']}")
        dataset.data_infos = [
            info for info in dataset.data_infos
            if frame_key(info, dataset.split) not in previous
        ]
    if not dataset.data_infos:
        if args.resume and selected_count and existing_index.exists():
            print(json.dumps({"written": 0, "index": str(existing_index), "already_complete": True}))
            return
        raise ValueError("No frames selected; check annotation and manifest")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.cuda().eval()
    captured = {}

    def capture_head(_module, _inputs, output):
        captured["output"] = output

    hook = model.pts_bbox_head.register_forward_hook(capture_head)
    parallel_model = MMDataParallel(model, device_ids=[0])
    writer = FeatureCacheWriter(args.output)
    written = 0
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                captured.clear()
                predictions = parallel_model(return_loss=False, rescale=True, **batch)
                if len(predictions) != 1 or "output" not in captured:
                    raise RuntimeError("Expected one prediction and one lane head output per frame")
                prediction = predictions[0]
                head = captured.pop("output")
                info = dataset.data_infos[index]
                key = frame_key(info, dataset.split)
                lanes = np.asarray(prediction["lane_results"][0], dtype=np.float32).reshape(-1, 11, 3)
                confidence = np.asarray(prediction["lane_results"][1], dtype=np.float32)
                query_features = head["history_states"][-1, 0].detach().cpu().numpy()
                semantic_scores = head["all_lclc_preds"][-1, 0, :, :, 0].sigmoid().detach().cpu().numpy()
                fused_scores = np.asarray(prediction["lclc_results"], dtype=np.float32)

                ann = dataset.get_ann_info(index)
                gt_lanes = ann["gt_lanes_3d"]
                gt_matrix = np.stack([
                    fix_pts_interpolate(lane, lanes.shape[1]).reshape(-1)
                    for lane in gt_lanes
                ]).astype(np.float32) if gt_lanes else np.empty((0, lanes.shape[1] * 3), dtype=np.float32)
                predicted_lanes = head["all_lanes_preds"][-1, 0]
                predicted_logits = head["all_cls_scores"][-1, 0]
                assigned = model.pts_bbox_head.assigner.assign(
                    predicted_lanes, predicted_logits,
                    torch.as_tensor(gt_matrix, device=predicted_lanes.device),
                    torch.zeros(len(gt_lanes), dtype=torch.long, device=predicted_lanes.device),
                )
                query_to_gt = assigned.gt_inds.detach().cpu().numpy().astype(np.int64) - 1
                connectors = [
                    bool(lane.get("is_intersection_or_connector", False))
                    for lane in info["annotation"]["lane_centerline"]
                ]
                record = manifest.get(key, {})
                writer.write(
                    frame_key=key,
                    query_features=query_features,
                    lanes=lanes,
                    confidence=confidence,
                    query_to_gt=query_to_gt,
                    gt_lanes=gt_lanes,
                    gt_adjacency=ann["gt_lane_adj"],
                    gt_is_intersection_or_connector=connectors,
                    base_topology_scores=fused_scores,
                    semantic_topology_scores=semantic_scores,
                    tags=record.get("tags", []),
                    split=dataset.split,
                    metadata={"model_split": record.get("model_split", dataset.split)},
                )
                written += 1
                print(json.dumps({"frame_key": key, "queries": len(lanes), "matched": int((query_to_gt >= 0).sum())}), flush=True)
    finally:
        hook.remove()
    print(json.dumps({"written": written, "index": str(writer.index_path)}))


if __name__ == "__main__":
    main()
