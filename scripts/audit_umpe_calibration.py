"""Independently recompose camera-to-LiDAR transforms from source metadata."""
import json,pathlib,pickle,sqlite3
import numpy as np
from pyquaternion import Quaternion
R=pathlib.Path('/root/autodl-tmp/manner-repro');N=R/'data/nuscenes'
db=sqlite3.connect('file:'+str(R/'data/umpe_metadata.sqlite')+'?mode=ro',uri=True)
state=pickle.loads(db.execute('select payload from state').fetchone()[0]);tables=state['tables']
with (N/'nuscenes_map_infos_temporal_val.pkl').open('rb') as f:infos=sorted(pickle.load(f)['infos'],key=lambda x:x['timestamp'])
def get(table,token):
    if table in tables:return tables[table][token]
    return json.loads(db.execute('select value from records where tab=? and token=?',(table,token)).fetchone()[0])
def transform(record):
    t=np.eye(4);t[:3,:3]=Quaternion(record['rotation']).rotation_matrix;t[:3,3]=record['translation'];return t
rows=[]
for i in np.linspace(0,len(infos)-1,16,dtype=int):
    info=infos[i];sample=get('sample',info['token']);lidar=get('sample_data',sample['data']['LIDAR_TOP'])
    cs_l=get('calibrated_sensor',lidar['calibrated_sensor_token']);pose_l=get('ego_pose',lidar['ego_pose_token'])
    lidar_to_global=transform(pose_l)@transform(cs_l)
    assert np.array_equal(info['ego2global_translation'],pose_l['translation'])
    assert np.array_equal(info['ego2global_rotation'],pose_l['rotation'])
    assert np.array_equal(info['lidar2ego_translation'],cs_l['translation'])
    assert np.array_equal(info['lidar2ego_rotation'],cs_l['rotation'])
    errors=[]
    for cam,stored in info['cams'].items():
        sd=get('sample_data',sample['data'][cam]);cs=get('calibrated_sensor',sd['calibrated_sensor_token']);pose=get('ego_pose',sd['ego_pose_token'])
        assert sd['sample_token']==info['token'] and sd['is_key_frame']
        assert sd['timestamp']==stored['timestamp']
        assert np.array_equal(np.array(cs['camera_intrinsic']),stored['cam_intrinsic'])
        path=pathlib.Path(stored['data_path'])
        if not path.is_absolute():path=R/'repos/UMPE/Maptrv2'/path
        assert path.resolve()==(N/sd['filename']).resolve()
        expected=np.linalg.inv(lidar_to_global)@transform(pose)@transform(cs)
        actual=np.eye(4);actual[:3,:3]=stored['sensor2lidar_rotation'];actual[:3,3]=stored['sensor2lidar_translation']
        error=float(np.max(np.abs(expected-actual)));assert error<1e-8,(info['token'],cam,error)
        errors.append(error)
    rows.append({'token':info['token'],'map_location':info['map_location'],'camera_count':len(errors),'max_matrix_error':max(errors)})
record={'frames':len(rows),'cameras':sum(x['camera_count'] for x in rows),'max_matrix_error':max(x['max_matrix_error'] for x in rows),'formula':'inverse(lidar_to_global) @ camera_ego_to_global @ camera_to_ego','intrinsics_timestamps_paths_and_keyframes_match':True,'records':rows}
(R/'reports/diagnostics/umpe_calibration_audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
