"""Compare released checkpoint configuration against the repository example."""
import ast,json,pathlib
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def config(path):
    src=path.read_text();tree=ast.parse(src)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)) for n in ast.walk(tree))
    ns={'__builtins__':{'dict':dict,'range':range,'len':len}}
    exec(compile(src,str(path),'exec'),ns)
    return ns
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:result.append({'path':path+'/'+k,'repo':repr(a.get(k)),'checkpoint':repr(b.get(k))})
            else:result.extend(diff(a[k],b[k],path+'/'+k))
        return result
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b):
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in diff(x,y,path+'/'+str(i))]
    return [] if a==b else [{'path':path,'repo':repr(a),'checkpoint':repr(b)}]
repo=config(R/'repos/UMPE/Maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_umpe.py')
report={}
for name in ('maptrv2_umpe','maptrv2_umpe_ft'):
    saved=config(R/'manifests'/(name+'_embedded_config.txt'))
    row={k:diff(repo[k],saved[k],k) for k in ('model','test_pipeline','train_pipeline','optimizer','lr_config','custom_hooks')}
    row['test_dataset_differences']=diff(repo['data']['test'],saved['data']['test'],'data/test')
    assert not row['model'] and not row['test_pipeline'],'Published model/test-pipeline differs from checkpoint'
    report[name]=row
    print(json.dumps({name:row}),flush=True)
(R/'manifests/umpe_config_audit.json').write_text(json.dumps(report,indent=2))
