"""Materialize official priors and backbone caches; never fabricate missing priors."""
import hashlib, json, pathlib, subprocess, time, zipfile
R = pathlib.Path('/root/autodl-tmp/manner-repro')
P = R/'data/umpe_priors'
N = R/'data/nuscenes'

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        assert dst.resolve() == src.resolve(), (src, dst)
    elif dst.exists():
        raise RuntimeError(f'Unexpected existing destination: {dst}')
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())

report = {}
required = json.loads((R/'manifests/nuscenes_required_files.json').read_text())
tokens = set(required['files'].values())
meta=N/'v1.0-trainval'
logs={x['token']:x['location'] for x in json.loads((meta/'log.json').read_text())}
scenes={x['token']:logs[x['log_token']] for x in json.loads((meta/'scene.json').read_text())}
satellite_keys={str(x['timestamp'])+'_'+scenes[x['scene_token']] for x in json.loads((meta/'sample.json').read_text()) if x['token'] in tokens}
for name, rootname, dest in [
    ('nuscenes_satellite_map.zip', 'nuscenes_satellite_map', N/'satellite_map_dataset/rotated_trainval'),
    ('nuscenes_rasterized_sdmap.zip', 'nuscenes_rasterized_sdmap', N/'sd_map_reraster2_vis'),
]:
    with zipfile.ZipFile(P/name) as z:
        for item in z.infolist():
            target = (P/item.filename).resolve()
            assert target.is_relative_to(P.resolve())
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # ZipFile.read checks member CRC32.
            if not target.exists() or target.stat().st_size != item.file_size:
                target.write_bytes(z.read(item))
    link(P/rootname, dest)
    satellite=name=='nuscenes_satellite_map.zip'
    present = {('_'.join(f.stem.split('_')[:2]) if satellite else f.stem) for f in dest.iterdir() if f.is_file()}
    missing = sorted((satellite_keys if satellite else tokens)-present)
    report[name] = dict(files=len(present), selected_frames=len(tokens), missing=missing)
    print(json.dumps(dict(file=name, files=len(present), selected_frames=len(tokens), missing=len(missing), examples=missing[:5])), flush=True)
    if satellite:
        assert not missing, f'Missing {len(missing)} satellite priors'
    else:
        report[name]['missing_policy']='Unmodified official LoadSDRasterFromFile supplies a white raster; no files fabricated'
        token_split={required['files'][f]:s for f,s in required['split'].items()}
        report[name]['missing_by_split']={s:sum(token_split[t]==s for t in missing) for s in ('val','train_smoke')}
for src, dst in [
    ('nuscenes_trainval_sdmap.pkl', 'inter_nuscenes_trainval_sdmap_new.pkl'),
    ('nuscenes_trainval_hdmap.pkl', 'interpolated_nuscenes_trainval_hdmap.pkl'),
]:
    link(P/src, N/dst)
link(N, R/'repos/UMPE/Maptrv2/data/nuscenes')
link(R/'data/can_bus', R/'repos/UMPE/Maptrv2/data/can_bus')
cache = R/'weights/torch/hub/checkpoints'
cache.mkdir(parents=True, exist_ok=True)
for name in ('resnet50-19c8e357.pth', 'resnet18-f37072fd.pth', 'resnet50-11ad3fa6.pth'):
    path = cache/name
    start = time.monotonic()
    if not path.exists():
        subprocess.run(['curl','-L','-x','http://127.0.0.1:37767','--fail','--retry','4','--speed-time','45','--speed-limit','200000','-C','-','-o',str(path)+'.part','https://download.pytorch.org/models/'+name], check=True)
        pathlib.Path(str(path)+'.part').replace(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h.startswith(name.split('-')[1].split('.')[0]), (name,h)
    report[name] = dict(bytes=path.stat().st_size, sha256=h, seconds=time.monotonic()-start)
    if name != 'resnet50-11ad3fa6.pth':
        link(path, R/'repos/UMPE/Maptrv2/ckpts'/name)
(R/'manifests/auxiliary_assets.json').write_text(json.dumps(report, indent=2))
print('All auxiliary assets verified', flush=True)
