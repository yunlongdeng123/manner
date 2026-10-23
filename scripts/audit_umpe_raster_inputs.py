"""Check image-prior association independently of the GPU inference path."""
import collections,json,pathlib,pickle
import cv2,numpy as np
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'data/nuscenes'
cv2.setNumThreads(1)
with (D/'nuscenes_map_infos_temporal_val.pkl').open('rb') as f:infos=sorted(pickle.load(f)['infos'],key=lambda x:x['timestamp'])
sat=D/'satellite_map_dataset/rotated_trainval';raster=D/'sd_map_reraster2_vis'
index=collections.defaultdict(list)
for p in sat.glob('*.jpg'):
    parts=p.stem.split('_');index[(parts[0],parts[1])].append(p)
ambiguous=[];pose_errors=[];missing=[];examples=[]
selected=set(np.linspace(0,len(infos)-1,32,dtype=int).tolist())
for n,i in enumerate(infos):
    matches=index[(str(i['timestamp']),i['map_location'])]
    if len(matches)!=1:ambiguous.append(dict(token=i['token'],matches=[str(p) for p in matches]));continue
    parts=matches[0].stem.split('_')
    xy=np.asarray([float(parts[2]),float(parts[3])]);pose=np.asarray(i['ego2global_translation'][:2])
    pose_errors.append(float(np.abs(xy-pose).max()))
    rp=raster/(i['token']+'.png')
    if not rp.exists():missing.append(i['token'])
    if n in selected:
        row=dict(index=n,token=i['token'],satellite=str(matches[0]),raster_missing=not rp.exists(),modalities={})
        for name,path in [('satellite',matches[0]),('sd_raster',rp)]:
            if not path.exists():continue
            img=cv2.imread(str(path));assert img is not None,str(path)
            row['modalities'][name]=dict(shape=list(img.shape),min=int(img.min()),max=int(img.max()),mean=float(img.mean()),std=float(img.std()),nonwhite_fraction=float(np.mean(np.any(img!=255,axis=-1))))
        examples.append(row)
record=dict(frames=len(infos),ambiguous_or_missing_satellite_matches=ambiguous,satellite_pose_filename_max_error_m=max(pose_errors),raster_missing_count=len(missing),raster_missing_tokens=missing,pixel_audit_examples=examples,method='Independent filename index by timestamp/location, x/y compared against ego global pose; 32 equally spaced decoded image samples',limitation='Does not independently establish geographic correctness or rotation of the published imagery')
out=R/'reports/diagnostics/umpe_raster_input_audit.json';out.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k not in ['pixel_audit_examples','raster_missing_tokens']},indent=2))
assert not ambiguous and max(pose_errors)<1e-8 and len(missing)==16
