"""Verified tar extraction with reconnects; archives never occupy a second copy."""
import hashlib, io, json, pathlib, subprocess, tarfile, time, urllib.request, urllib.error
ROOT=pathlib.Path('/root/autodl-tmp/manner-repro'); P='http://127.0.0.1:37767'
DATASET='AlayaNeW/OpenDriveLab___OpenLane-V2'
REV='6c02353d42c777e2aab5a1056fbc80d1cc8fd798'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None
class Reader:
    def __init__(self,item):
        self.item=item;self.pos=0;self.resp=None;self.sha=hashlib.sha256();self.md5=hashlib.md5();self.last=time.monotonic();self.lastpos=0
    def connect(self):
        if self.resp:self.resp.close()
        url=f'https://huggingface.co/datasets/{DATASET}/resolve/{REV}/{self.item["path"]}?download=true'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':P,'https':P}),NoRedirect)
        try:opener.open(urllib.request.Request(url,method='HEAD'),timeout=30)
        except urllib.error.HTTPError as e:
            if e.code not in (301,302,303,307,308):raise
            url=e.headers['Location']
        direct=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req=urllib.request.Request(url,headers={'Range':f'bytes={self.pos}-'})
        self.resp=direct.open(req,timeout=45)
        if self.pos and not self.resp.headers.get('Content-Range','').startswith(f'bytes {self.pos}-'):raise RuntimeError('Invalid resume response')
    def read(self,n=-1):
        if self.pos>=self.item['size']:return b''
        n=min(n if n>0 else 1024*1024,self.item['size']-self.pos)
        for attempt in range(8):
            try:
                if self.resp is None:self.connect()
                data=self.resp.read(n)
                if not data:raise IOError('Unexpected EOF')
                self.pos+=len(data);self.sha.update(data);self.md5.update(data)
                now=time.monotonic()
                if now-self.last>30:
                    print(json.dumps({'event':'progress','file':self.item['path'],'bytes':self.pos,'total':self.item['size'],'MiB_s':round((self.pos-self.lastpos)/(now-self.last)/1048576,2)}),flush=True)
                    self.last=now;self.lastpos=self.pos
                return data
            except Exception as e:
                if self.resp:self.resp.close()
                self.resp=None
                print(json.dumps({'event':'reconnect','byte':self.pos,'error':str(e),'attempt':attempt}),flush=True)
                time.sleep(min(2**attempt,30))
        raise RuntimeError('Repeated download failure')

def extract(item,kind,official_md5):
    dest=ROOT/'data'/('openlane_smoke' if kind=='sample' else 'openlanev2')
    ready=ROOT/'manifests'/('verified_'+pathlib.Path(item['path']).name+'.json')
    if ready.exists():print('Already verified '+str(ready),flush=True);return
    dest.mkdir(parents=True,exist_ok=True)
    reader=Reader(item);count=0;written=0
    print(json.dumps({'event':'start','file':item['path'],'dest':str(dest)}),flush=True)
    with tarfile.open(fileobj=reader,mode='r|',bufsize=1024*1024) as archive:
        for member in archive:
            parts=pathlib.PurePosixPath(member.name).parts
            if member.name.startswith('/') or '..' in parts:raise RuntimeError('Unsafe tar path')
            if not member.isfile():continue
            if kind=='info' and ('train' in parts or 'test' in parts):continue
            # Strip the publisher's common wrapper directory only.
            if parts and parts[0] in ('OpenLane-V2','OpenLane-V2_sample'):parts=parts[1:]
            target=dest.joinpath(*parts)
            if not target.resolve().is_relative_to(dest.resolve()):raise RuntimeError('Unsafe target')
            count+=1
            if target.exists() and target.stat().st_size==member.size:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=pathlib.Path(str(target)+'.part')
            with archive.extractfile(member) as src,temp.open('wb') as out:
                for block in iter(lambda:src.read(1024*1024),b''):out.write(block)
            temp.replace(target);written+=member.size
    while reader.read(1024*1024):pass
    assert reader.pos==item['size']
    assert reader.sha.hexdigest()==item['lfs']['oid'], 'Mirror SHA256 mismatch'
    assert reader.md5.hexdigest()==official_md5, 'Official MD5 mismatch'
    result={'file':item['path'],'revision':REV,'size':reader.pos,'sha256':reader.sha.hexdigest(),'md5':reader.md5.hexdigest(),'extracted_count':count,'new_bytes':written,'destination':str(dest)}
    ready.write_text(json.dumps(result,indent=2));print(json.dumps({'event':'verified',**result}),flush=True)

items={pathlib.Path(x['path']).name:x for x in json.loads((ROOT/'manifests/openlane_hf_raw.json').read_text())}
for name,kind,md5 in [
 ('OpenLane-V2_sample.tar','sample','21c607fa5a1930275b7f1409b25042a0'),
 ('OpenLane-V2_subset_A_info.tar','info','95bf28ccf22583d20434d75800be065d'),
 ('OpenLane-V2_subset_A_image_7.tar','images','c73af4a7aef2692b96e4e00795120504')]:
    extract(items[name],kind,md5)
