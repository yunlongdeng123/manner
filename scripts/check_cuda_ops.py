import json,pathlib,torch
from projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function.geometric_kernel_attn_func import GeometricKernelAttentionFunc
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction,multi_scale_deformable_attn_pytorch
assert torch.cuda.is_available()
torch.manual_seed(0)
device='cuda';B,Q,H,C,L,P=2,3,2,4,2,5
shapes=torch.tensor([[3,4],[2,2]],device=device,dtype=torch.long)
starts=torch.tensor([0,12],device=device,dtype=torch.long)
def compare(name,custom,reference,value,weights):
    probe=torch.randn_like(custom)
    got=torch.autograd.grad(custom,(value,weights),probe,retain_graph=True)
    expected=torch.autograd.grad(reference,(value,weights),probe)
    assert torch.allclose(custom,reference,atol=2e-5,rtol=2e-5),name+' forward'
    errors=[]
    for a,b in zip(got,expected):
        assert torch.isfinite(a).all() and a.abs().sum()>0
        assert torch.allclose(a,b,atol=2e-5,rtol=2e-5),name+' backward'
        errors.append(float((a-b).abs().max()))
    return dict(forward_max_error=float((custom-reference).abs().max()),backward_max_errors=errors)
v=torch.randn(B,16,H,C,device=device,requires_grad=True)
w=torch.randn(B,Q,H,L,P,device=device,requires_grad=True)
loc=torch.randint(-2,6,(B,Q,H,L,P,2),device=device,dtype=torch.long)
out=GeometricKernelAttentionFunc.apply(v,shapes,starts,loc,w,2)
parts=[]
for b in range(B):
    queries=[]
    for q in range(Q):
        heads=[]
        for h in range(H):
            total=0
            for l,(height,width) in enumerate(shapes.tolist()):
                xy=loc[b,q,h,l];idx=starts[l]+xy[:,1].clamp(0,height-1)*width+xy[:,0].clamp(0,width-1)
                total=total+(v[b,idx,h,:]*w[b,q,h,l,:,None]).sum(0)
            heads.append(total)
        queries.append(torch.cat(heads))
    parts.append(torch.stack(queries))
record={'gkt':compare('GKT',out,torch.stack(parts),v,w)}
v=torch.randn(B,16,H,C,device=device,requires_grad=True)
w=torch.rand(B,Q,H,L,P,device=device,requires_grad=True)
loc=torch.rand(B,Q,H,L,P,2,device=device)
out=MultiScaleDeformableAttnFunction.apply(v,shapes,starts,loc,w,2)
ref=multi_scale_deformable_attn_pytorch(v,shapes,loc,w)
record['mmcv_deformable_attention']=compare('MMCV',out,ref,v,w)
record.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__)
pathlib.Path('/root/autodl-tmp/manner-repro/manifests/cuda_ops_verified.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
