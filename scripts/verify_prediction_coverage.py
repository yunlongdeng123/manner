"""Reject incomplete inference outputs before claiming full-validation results."""
import hashlib,json,pathlib,pickle,sys
import numpy as np
R=pathlib.Path('/root/autodl-tmp/manner-repro')
kind,path=sys.argv[1],pathlib.Path(sys.argv[2])
with path.open('rb') as f:pred=pickle.load(f)
if kind=='topogpt':
    split=json.loads((R/'data/openlanev2/data_dict_subset_A.json').read_text())['val']
    expected={('val',seg,t.split('.')[0]) for seg,stamps in split.items() for t in stamps}
    actual=set(pred)
    assert actual==expected,(len(expected-actual),len(actual-expected))
    count=len(expected)
    basis='Exact equality of all official frame keys'
elif kind=='umpe':
    # The official single-rank nonshuffling sampler preserves annotation order.
    count=6019
    assert isinstance(pred,list) and len(pred)==count,(type(pred),len(pred))
    basis='6019 outputs from official single-rank nonshuffling sampler in annotation order'
else:raise ValueError(kind)
leaves=[0]
def check_finite(obj):
    if hasattr(obj,'detach'):
        obj=obj.detach().cpu().numpy()
    if isinstance(obj,np.ndarray):
        if np.issubdtype(obj.dtype,np.number):
            assert np.isfinite(obj).all(),'Nonfinite prediction array'
            leaves[0]+=1
    elif isinstance(obj,dict):
        for v in obj.values():check_finite(v)
    elif isinstance(obj,(list,tuple)):
        for v in obj:check_finite(v)
    elif isinstance(obj,(float,np.floating)):
        assert np.isfinite(obj),'Nonfinite prediction scalar'
check_finite(pred)
h=hashlib.sha256()
with path.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
record=dict(kind=kind,prediction_file=str(path),frames=count,full_validation_coverage=True,coverage_basis=basis,all_numeric_outputs_finite=True,array_leaves_checked=leaves[0],sha256=h.hexdigest(),bytes=path.stat().st_size)
if kind=='topogpt':
    # EOS=2 is defined in the pinned author's SpecialTokenID enum.
    no_eos=[list(k) for k,v in pred.items() if 2 not in v['lane_token']]
    empty=[list(k) for k,v in pred.items() if not v['lane_centerline']]
    record.update(no_eos_frames=no_eos,empty_lane_frames=empty)
path.with_suffix('.coverage.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
