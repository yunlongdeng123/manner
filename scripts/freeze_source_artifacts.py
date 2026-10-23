import hashlib,json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro');D=R/'source_artifacts';D.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
records={}
for name in ['TopoGPT','UMPE','OpenLane-V2']:
    repo=R/'repos'/name
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    archive=D/(name+'_'+commit[:12]+'.tgz')
    if not archive.exists():subprocess.run(['git','-C',str(repo),'archive','--format=tar.gz','--prefix='+name+'/', '--output='+str(archive),'HEAD'],check=True)
    diff=subprocess.check_output(['git','-C',str(repo),'diff','--binary'])
    patch=D/(name+'_local.patch');patch.write_bytes(diff)
    records[name]=dict(commit=commit,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),patch=str(patch),patch_sha256=sha(patch),patch_bytes=len(diff))
    print(json.dumps(records[name]),flush=True)
(R/'manifests/source_archives.json').write_text(json.dumps(records,indent=2))
