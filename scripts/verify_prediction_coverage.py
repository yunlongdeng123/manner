"""Reject incomplete inference outputs before claiming full-validation results."""
import json,pathlib,pickle,sys
R=pathlib.Path('/root/autodl-tmp/manner-repro')
kind,path=sys.argv[1],pathlib.Path(sys.argv[2])
with path.open('rb') as f:pred=pickle.load(f)
if kind=='topogpt':
    split=json.loads((R/'data/openlanev2/data_dict_subset_A.json').read_text())['val']
    expected={('val',seg,t.split('.')[0]) for seg,stamps in split.items() for t in stamps}
    actual=set(pred)
    assert actual==expected,(len(expected-actual),len(actual-expected))
    count=len(expected)
elif kind=='umpe':
    # The official single-rank nonshuffling sampler preserves annotation order.
    count=6019
    assert isinstance(pred,list) and len(pred)==count,(type(pred),len(pred))
else:raise ValueError(kind)
record=dict(kind=kind,prediction_file=str(path),frames=count,full_validation_coverage=True)
path.with_suffix('.coverage.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
