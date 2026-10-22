"""Inspect top-level pickle keys/globals without constructing graphs/arrays.

The symbolic stack supports standard protocol 0-5 opcodes and fails closed on
unknown operations. Compare the smaller HD file with real unpickling first.
"""
import json,pathlib,pickle,pickletools,time
R=pathlib.Path('/root/autodl-tmp/manner-repro')
OTHER=object();MARK=object();ROOT=object()
def inspect(path):
    stack=[];memo={};next_memo=0;keys=set();globals_=set();root_created=False
    def pop_mark():
        i=len(stack)-1
        while stack[i] is not MARK:i-=1
        values=stack[i+1:];del stack[i:];return values
    def store(index):
        nonlocal next_memo
        value=stack[-1]
        if value is OTHER:memo.pop(index,None)
        else:memo[index]=value
        next_memo=max(next_memo,index+1)
    with path.open('rb') as f:
        for op,arg,pos in pickletools.genops(f):
            n=op.name
            if n in ('PROTO','FRAME'):continue
            if n=='STOP':
                assert len(stack)==1 and stack[0] is ROOT,'Expected a top-level plain dictionary'
                break
            if n=='MARK':stack.append(MARK)
            elif n=='EMPTY_DICT':
                if not root_created and not stack:stack.append(ROOT);root_created=True
                else:stack.append(OTHER)
            elif n=='SETITEM':
                value=stack.pop();key=stack.pop()
                if stack[-1] is ROOT:
                    assert isinstance(key,str);keys.add(key)
            elif n=='SETITEMS':
                values=pop_mark()
                if stack[-1] is ROOT:
                    assert len(values)%2==0
                    for key in values[::2]:assert isinstance(key,str);keys.add(key)
            elif n in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','STRING','BINSTRING','SHORT_BINSTRING'):
                stack.append(arg)
            elif n in ('EMPTY_LIST','EMPTY_TUPLE','EMPTY_SET','NONE','NEWTRUE','NEWFALSE','INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4','FLOAT','BINFLOAT','BINBYTES','SHORT_BINBYTES','BINBYTES8','BYTEARRAY8','NEXT_BUFFER'):
                stack.append(OTHER)
            elif n in ('BINPUT','LONG_BINPUT','PUT'):store(int(arg))
            elif n=='MEMOIZE':store(next_memo)
            elif n in ('BINGET','LONG_BINGET','GET'):stack.append(memo.get(int(arg),OTHER))
            elif n=='GLOBAL':globals_.add(arg);stack.append(OTHER)
            elif n=='STACK_GLOBAL':
                name=stack.pop();module=stack.pop()
                assert isinstance(module,str) and isinstance(name,str)
                globals_.add(module+' '+name);stack.append(OTHER)
            elif n in ('TUPLE','LIST','DICT','FROZENSET','OBJ','INST'):
                pop_mark();stack.append(OTHER)
            elif n in ('TUPLE1','TUPLE2','TUPLE3'):
                count=int(n[-1]);del stack[-count:];stack.append(OTHER)
            elif n in ('REDUCE','NEWOBJ'):
                del stack[-2:];stack.append(OTHER)
            elif n=='NEWOBJ_EX':del stack[-3:];stack.append(OTHER)
            elif n in ('BUILD','APPEND','POP'):stack.pop()
            elif n in ('APPENDS','ADDITEMS','POP_MARK'):pop_mark()
            elif n=='DUP':stack.append(stack[-1])
            elif n in ('EXT1','EXT2','EXT4','PERSID'):stack.append(OTHER)
            elif n=='BINPERSID':stack[-1]=OTHER
            elif n=='READONLY_BUFFER':pass
            else:raise RuntimeError((n,pos))
    assert root_created and keys
    return keys,sorted(globals_)

required=json.loads((R/'manifests/nuscenes_required_files.json').read_text())
selected=set(required['files'].values());report={}
for name in ('nuscenes_trainval_hdmap.pkl','nuscenes_trainval_sdmap.pkl'):
    start=time.monotonic();path=R/'data/umpe_priors'/name
    keys,globals_=inspect(path)
    if 'hdmap' in name:
        with path.open('rb') as f:real=pickle.load(f)
        assert set(real)==keys,'Symbolic parser does not match actual HD pickle keys'
        del real
    missing=sorted(selected-keys)
    row=dict(keys=len(keys),selected_frames=len(selected),missing=missing,globals=globals_,seconds=time.monotonic()-start,inspection='symbolic pickle stack; HD keys additionally compared with real unpickling')
    report[name]=row
    print(json.dumps(row),flush=True)
    assert not missing,f'{name}: missing {len(missing)} vector priors'
(R/'manifests/umpe_prior_keys_verified.json').write_text(json.dumps(report,indent=2))
