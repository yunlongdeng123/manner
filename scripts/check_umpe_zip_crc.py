import json,pathlib,time,zipfile
R=pathlib.Path('/root/autodl-tmp/manner-repro');report={}
for p in sorted((R/'weights/umpe').glob('maptrv2_umpe*.pth')):
    start=time.monotonic()
    with zipfile.ZipFile(p) as z:
        bad=z.testzip()
        assert bad is None,(p,bad)
        report[p.name]=dict(zip_crc_verified=True,members=len(z.infolist()),seconds=time.monotonic()-start)
    print(json.dumps({p.name:report[p.name]}),flush=True)
(R/'manifests/umpe_checkpoint_zip_crc.json').write_text(json.dumps(report,indent=2))
