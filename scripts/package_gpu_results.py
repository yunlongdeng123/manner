"""Package final evidence after reporting and Git commit, without raw images/weights."""
import hashlib,json,pathlib,subprocess,tarfile
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'delivery';D.mkdir(exist_ok=True)
summary=json.loads((R/'reports/results.json').read_text())
assert len(summary['runs'])==3
for name in ['topogpt_examples.png','umpe_examples.png','architecture_components.png','selection.json']:
    assert (R/'reports/figures'/name).exists(),name
head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
bundle=D/('reproduction_'+head[:12]+'.bundle')
assert not bundle.exists(),bundle
subprocess.run(['git','-C',str(R),'bundle','create',str(bundle),'--all'],check=True)
subprocess.run(['git','-C',str(R),'bundle','verify',str(bundle)],check=True)
archive=D/('repro-results_'+head[:12]+'.tgz')
def keep(info):
    parts=pathlib.PurePosixPath(info.name).parts
    if '__pycache__' in parts or info.name.endswith(('.pth','.ckpt','.pyc')):return None
    if info.issym() or info.islnk():return None
    return info
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
    for name in ['README.md','scripts','configs','patches','manifests','reports','logs','source_artifacts','results','papers']:
        t.add(R/name,arcname=name,filter=keep)
    topo=pathlib.Path(json.loads((R/'manifests/topogpt_gpu_smoke_passed.json').read_text())['directory'])
    t.add(topo,arcname='gpu_smoke/topogpt',filter=keep)
    inputs=json.loads((R/'manifests/evaluation_inputs.json').read_text())
    for label,entry in inputs.items():
        t.add(entry['source'],arcname='evaluation_inputs/'+label,filter=keep)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
record=dict(commit=head,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [bundle,archive]},scope='Reports, complete validation predictions/metrics/annotations, logs, scripts, exact source archives/patches and papers. Raw images, raw priors, model/checkpoint weights and Python environments remain on remote.')
(D/'delivery_manifest.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
