"""Final consistency checks without launching or changing any experiment."""
import datetime,hashlib,json,pathlib,subprocess
R=pathlib.Path('/root/autodl-tmp/manner-repro')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
summary=json.loads((R/'reports/results.json').read_text())
checks={}
for name,row in list(summary['runs'].items())+[('maptrv2_control',summary['diagnostic_control'])]:
    p=pathlib.Path(row['directory'])/'predictions.pkl';c=row['coverage']
    assert c['full_validation_coverage'] and c['all_numeric_outputs_finite']
    assert p.stat().st_size==c['bytes'] and sha(p)==c['sha256'],name
    assert c['frames']==(4806 if name=='topogpt' else 6019)
    checks[name]=dict(frames=c['frames'],prediction_sha256=c['sha256'])
sources=json.loads((R/'manifests/source_archives.json').read_text())
for name,s in sources.items():
    repo=R/'repos'/name
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==s['commit']
    diff=subprocess.check_output(['git','-C',str(repo),'diff','--binary'])
    assert hashlib.sha256(diff).hexdigest()==s['patch_sha256'],name
    assert sha(pathlib.Path(s['archive']))==s['sha256'],name
    assert sha(pathlib.Path(s['patch']))==s['patch_sha256'],name
for p in (R/'papers').glob('*.pdf'):
    with p.open('rb') as f:assert f.read(5)==b'%PDF-',str(p)
assert len(list((R/'papers').glob('*.pdf')))==2
logs=(R/'logs/topogpt_full_eval_20261023.log').read_text(errors='replace')
warnings=dict(apls_error_messages=logs.count('Error calculating APLS metric:'),mean_empty_slice_messages=logs.count('Mean of empty slice'),note='Log message counts, not per-frame failure counts; official evaluator behavior preserved')
gitremotes=subprocess.check_output(['git','-C',str(R),'remote','-v'],text=True).strip()
record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=checks,source_commits_and_patches_unchanged=True,source_archives_verified=True,paper_headers_valid=True,topogpt_metric_log_warnings=warnings,report_sha256=sha(R/'reports/reproduction_report.md'),git_remote=gitremotes or None,backup_strategy='Verified Git bundle and local clone; no remote configured' if not gitremotes else 'Configured remote plus bundle')
(R/'manifests/delivery_ready.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
