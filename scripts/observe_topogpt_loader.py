"""Read-only inotify observation of data loading; no model/process injection.

This reports the loaded frame position, which includes up to a few prefetched
batches. It is NOT a completed prediction count or an evaluation coverage check.
"""
import ctypes,datetime,json,os,pathlib,pickle,select,struct,time
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'data/topogpt/openlane_subA'
with (D/'val.pkl').open('rb') as f:frames=pickle.load(f)
paths={}
for i,frame in enumerate(frames):
    for cam in frame['sensor'].values():
        p=(D/cam['image_path']).resolve();paths[str(p)]=i
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_NONBLOCK)
assert fd>=0
watches={}
for directory in {str(pathlib.Path(p).parent) for p in paths}:
    wd=lib.inotify_add_watch(fd,directory.encode(),0x20) # IN_OPEN only
    assert wd>=0,(directory,ctypes.get_errno())
    watches[wd]=directory
seen=[];deadline=time.monotonic()+12
while time.monotonic()<deadline:
    if not select.select([fd],[],[],1)[0]:continue
    data=os.read(fd,1024*1024);offset=0
    while offset+16<=len(data):
        wd,mask,cookie,size=struct.unpack_from('iIII',data,offset);offset+=16
        name=data[offset:offset+size].split(b'\0',1)[0].decode();offset+=size
        path=watches.get(wd,'')+'/'+name
        if path in paths:seen.append(paths[path])
os.close(fd)
record=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),frames_total=len(frames),observed_loader_position=max(seen)+1 if seen else None,observed_reads=len(seen),includes_prefetch=True,prediction_completion_not_asserted=True)
(R/'manifests/topogpt_loader_activity.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
