"""Read actual official checkpoint structures without allocating all tensor storage.
This checks CPU deserialization and tensor metadata, not GPU numerical correctness.
"""
import collections, gc, json, pathlib, zipfile
import torch
R=pathlib.Path('/root/autodl-tmp/manner-repro')
report={}
for rel in ('topogpt/pretrain.ckpt','topogpt/finetune.ckpt','umpe/maptrv2_umpe.pth','umpe/maptrv2_umpe_ft.pth'):
    path=R/'weights'/rel
    assert zipfile.is_zipfile(path),f'Expected PyTorch ZIP checkpoint: {path}'
    ckpt=torch.load(str(path),map_location='cpu',mmap=True)
    state=ckpt['state_dict']
    assert state and all(isinstance(k,str) for k in state)
    shapes={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items() if torch.is_tensor(v)}
    row=dict(file=str(path),keys=list(ckpt),tensor_count=len(shapes),tensor_elements=sum(v.numel() for v in state.values() if torch.is_tensor(v)),prefixes=dict(collections.Counter(k.split('.')[0] for k in shapes)),cpu_mmap_load=True,full_state=bool(ckpt.get('optimizer_states') or ckpt.get('optimizer')))
    if 'meta' in ckpt:
        meta=ckpt['meta'];row['meta_keys']=list(meta)
        row['epoch']=meta.get('epoch');row['iter']=meta.get('iter')
        if 'config' in meta:
            (R/'manifests'/(path.stem+'_embedded_config.txt')).write_text(str(meta['config']))
    if 'global_step' in ckpt:row['global_step']=ckpt['global_step']
    (R/'manifests'/(path.stem+'_tensor_shapes.json')).write_text(json.dumps(shapes,indent=2))
    report[rel]=row
    print(json.dumps(row),flush=True)
    del ckpt,state,shapes;gc.collect()
(R/'manifests/checkpoints_inventory.json').write_text(json.dumps(report,indent=2))
