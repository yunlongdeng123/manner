"""Verify transferred results and source archives, then materialize a Git clone."""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys,tarfile
parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
D=pathlib.Path(args.directory).resolve()
manifest=json.loads((D/'delivery_manifest.json').read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for name,entry in manifest['files'].items():
    path=D/name
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
archive=next(D.glob('repro-results_*.tgz'));bundle=next(D.glob('reproduction_*.bundle'))
records=D/'records'
if not args.verify_only:
    assert not records.exists();records.mkdir()
    with tarfile.open(archive,'r:gz') as t:
        for member in t.getmembers():
            target=(records/member.name).resolve()
            assert target.is_relative_to(records),member.name
            assert member.isfile() or member.isdir(),member.name
        t.extractall(records)
    for source in (D/'source_artifacts').glob('*.tgz'):
        target=records/'source_artifacts'/source.name
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
else:assert records.is_dir()
sources=json.loads((records/'manifests/source_archives.json').read_text(encoding='utf-8'))
for source in sources.values():
    assert sha(records/'source_artifacts'/pathlib.PurePosixPath(source['archive']).name)==source['sha256']
    assert sha(records/'source_artifacts'/pathlib.PurePosixPath(source['patch']).name)==source['patch_sha256']
summary=json.loads((records/'reports/results.json').read_text(encoding='utf-8'))
inputs=json.loads((records/'manifests/evaluation_inputs.json').read_text(encoding='utf-8'))
for label,entry in inputs.items():
    path=records/'evaluation_inputs'/label
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],label
for name,row in list(summary['runs'].items())+[('maptrv2_control',summary['diagnostic_control'])]:
    rel=pathlib.PurePosixPath(row['directory']).relative_to('/root/autodl-tmp/manner-repro')
    assert sha(records/pathlib.Path(str(rel))/'predictions.pkl')==row['coverage']['sha256'],name
deployment=D/'deployment'
if not args.verify_only:
    subprocess.run(['git','-c','core.autocrlf=false','clone','--quiet',str(bundle),str(deployment)],check=True)
    subprocess.run(['git','-C',str(deployment),'config','core.autocrlf','false'],check=True)
subprocess.run(['git','-C',str(deployment),'fsck','--full'],check=True)
head=subprocess.check_output(['git','-C',str(deployment),'rev-parse','HEAD'],text=True).strip()
assert head==manifest['commit']
assert not subprocess.check_output(['git','-C',str(deployment),'status','--porcelain'],text=True).strip()
verification=dict(artifact_hashes=True,source_archive_hashes=True,full_prediction_hashes=True,evaluation_input_hashes=True,git_fsck=True,commit=head,report=str(records/'reports/reproduction_report.md'))
if not args.verify_only:(D/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
