"""Exact output comparison against the pinned OpenLane-V2 collect function."""
import ast, json, pathlib, pickle, tempfile, types
import numpy as np
from prepare_openlane import collect, ROOT
raw=ROOT/'data/openlane_smoke'
paths=sorted((raw/'train').glob('*/info/*.json'))[:3]
assert len(paths)==3
selection={'train':{}}
for p in paths:selection['train'].setdefault(p.parent.parent.name,[]).append(p.stem)
source=ROOT/'repos/OpenLane-V2/openlanev2/preprocessing/collect.py'
tree=ast.parse(source.read_text())
tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!='io']
stored={}
io=types.SimpleNamespace(json_load=lambda path:json.loads(pathlib.Path(path).read_text()),pickle_dump=lambda path,obj:stored.update(result=obj))
ns={'io':io}
exec(compile(tree,str(source),'exec'),ns)
ns['collect'](str(raw),selection,'comparison',point_interval=1)
with tempfile.TemporaryDirectory(prefix='manner-collect-') as tmp:
    path=pathlib.Path(tmp)/'our.pkl'
    collect(raw,selection,path)
    actual=pickle.loads(path.read_bytes())
def equal(a,b,path='root'):
    assert type(a)==type(b),(path,type(a),type(b))
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and np.array_equal(a,b),(path,a.dtype,b.dtype)
    else:assert a==b,path
equal(actual,stored['result'])
report=dict(frames=3,all_keys_types_arrays_equal=True,point_interval=1,official_devkit='v1.1.0')
(ROOT/'manifests/collect_equivalence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
