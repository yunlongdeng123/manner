"""Include real publisher-missing raster cases in the separate GPU smoke set."""
import json,pathlib,pickle
R=pathlib.Path('/root/autodl-tmp/manner-repro');N=R/'data/nuscenes'
assert (R/'manifests/umpe_infos_ready.json').exists()
with (N/'nuscenes_map_infos_temporal_val.pkl').open('rb') as f:data=pickle.load(f)
missing=set(json.loads((R/'manifests/auxiliary_assets.json').read_text())['nuscenes_rasterized_sdmap.zip']['missing'])
infos=data['infos'];chosen=[x for x in infos if x['token'] not in missing][:4]+[x for x in infos if x['token'] in missing][:4]
assert len(chosen)==8 and len({x['token'] for x in chosen})==8
out=N/'nuscenes_map_infos_temporal_val_smoke_diverse.pkl'
with out.open('wb') as f:pickle.dump(dict(infos=chosen,metadata=data['metadata']),f)
record=dict(frames=8,real_missing_raster_frames=sum(x['token'] in missing for x in chosen),tokens=[x['token'] for x in chosen],annotation_file=str(out),formal_validation_unchanged=True)
(R/'manifests/umpe_smoke_selection.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
