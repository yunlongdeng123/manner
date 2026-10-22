import ctypes,json,pathlib,tempfile
r=pathlib.Path('/root/autodl-tmp/manner-repro')
lib=ctypes.CDLL(None,use_errno=True)
lib.fallocate.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong]
with tempfile.TemporaryFile(dir=r/'data') as f:
    f.write(b'A'*65536+b'B'*65536);f.flush()
    assert lib.fallocate(f.fileno(),3,0,65536)==0,ctypes.get_errno()
    f.seek(0);assert f.read(65536)==b'\0'*65536
    assert f.read()==b'B'*65536
report={'hole_punch_supported':True,'unconsumed_tail_intact':True}
(r/'manifests/cache_reclaim_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
