"""Resume a local prefix and fetch independent byte ranges over SSH, then hash."""
import argparse,concurrent.futures,hashlib,json,math,pathlib,shlex,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--remote',required=True);p.add_argument('--output',required=True);p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True);p.add_argument('--streams',type=int,default=4);a=p.parse_args()
out=pathlib.Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
manifest=out.with_name(out.name+'.transfer.json');start=time.monotonic()
if manifest.exists():state=json.loads(manifest.read_text())
else:
    prefix=out.stat().st_size if out.exists() else 0
    assert prefix<a.bytes
    old=out.with_name(out.name+'.prefix.part')
    if prefix:assert not old.exists();out.rename(old)
    size=math.ceil((a.bytes-prefix)/a.streams)
    chunks=[dict(start=i,end=min(i+size,a.bytes),file=str(out.with_name(out.name+f'.chunk{n}.part'))) for n,i in enumerate(range(prefix,a.bytes,size))]
    state=dict(remote=a.remote,output=str(out),bytes=a.bytes,sha256=a.sha256,prefix_bytes=prefix,prefix_file=str(old),chunks=chunks)
    manifest.write_text(json.dumps(state,indent=2))
assert state['remote']==a.remote and state['sha256']==a.sha256
def fetch(row):
    path=pathlib.Path(row['file']);length=row['end']-row['start'];have=path.stat().st_size if path.exists() else 0
    assert have<=length
    if have<length:
        cmd=f'dd if={shlex.quote(a.remote)} iflag=skip_bytes,count_bytes skip={row["start"]+have} count={length-have} status=none'
        with path.open('ab') as f:subprocess.run(['ssh','-o','BatchMode=yes','manner-3090-1023',cmd],stdout=f,check=True)
    assert path.stat().st_size==length
    print(json.dumps(dict(chunk=path.name,bytes=length,seconds=time.monotonic()-start)),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.streams) as pool:list(pool.map(fetch,state['chunks']))
tmp=out.with_name(out.name+'.assembled.part');assert not tmp.exists()
h=hashlib.sha256()
with tmp.open('wb') as dest:
    paths=([pathlib.Path(state['prefix_file'])] if state['prefix_bytes'] else [])+[pathlib.Path(x['file']) for x in state['chunks']]
    for path in paths:
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):dest.write(b);h.update(b)
assert tmp.stat().st_size==a.bytes and h.hexdigest()==a.sha256
tmp.rename(out)
state.update(verified=True,seconds=time.monotonic()-start,average_MiB_s=(a.bytes-state['prefix_bytes'])/2**20/(time.monotonic()-start))
manifest.write_text(json.dumps(state,indent=2))
# Keep only the verified final file; exact chunk paths must remain in its parent.
for path in paths:
    assert path.resolve().parent==out.parent and path.name.endswith('.part')
    path.unlink()
print(json.dumps(state),flush=True)
