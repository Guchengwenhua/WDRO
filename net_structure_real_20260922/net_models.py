"""Permutation-equivariant set / geometry-graph denoisers."""
import torch
from torch import nn

def mlp(a,h,b):return nn.Sequential(nn.Linear(a,h),nn.SiLU(),nn.Linear(h,b))

class Denoiser(nn.Module):
    def __init__(self,kind='set',width=96):
        super().__init__();self.kind=kind;self.width=width
        self.input=mlp(3,width,width)
        self.update=mlp(3*width,width,width)
        if kind=='graph':
            self.edge=mlp(2*width+4,width,width)
            self.graph_update=mlp(2*width,width,width)
        self.out=mlp(width,width,2)
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,y,sigma):
        b,n,_=y.shape;s=sigma.reshape(b,1,1).expand(-1,n,1)
        h=self.input(torch.cat([y-.5,s.log()/5],-1))
        h=h+self.update(torch.cat([h,h.mean(1,keepdim=True).expand(-1,n,-1),h.amax(1,keepdim=True).expand(-1,n,-1)],-1))
        if self.kind=='graph':
            dist=torch.cdist(y,y,p=1)+torch.eye(n,device=y.device)[None]*1e6
            order=torch.argsort(y[:,:,1],dim=1,stable=True)
            order=order.gather(1,torch.argsort(y[:,:,0].gather(1,order),dim=1,stable=True))
            ordered=dist.gather(2,order[:,None,:].expand(-1,n,-1))
            rank=torch.argsort(ordered,dim=2,stable=True)[:,:,:min(4,n-1)]
            idx=order[:,None,:].expand(-1,n,-1).gather(2,rank)
            batch=torch.arange(b,device=y.device)[:,None,None]
            hj=h[batch,idx];yj=y[batch,idx];hi=h[:,:,None].expand_as(hj)
            delta=yj-y[:,:,None];msg=self.edge(torch.cat([hi,hj,delta,delta.abs()],-1)).mean(2)
            h=h+self.graph_update(torch.cat([h,msg],-1))
        return y+s*self.out(h)

def losses(out,x,sigma,eta):
    s2=sigma.square().clamp_min(1e-6)
    rec=((out-x).square().mean((1,2))/s2).mean()
    if not eta:return rec,rec,rec.detach()*0
    n=x.shape[1];dx=torch.cdist(x,x,p=1);dy=torch.cdist(out,out,p=1)
    idx=(dx+torch.eye(n,device=x.device)[None]*1e6).topk(min(4,n-1),largest=False).indices
    edge=(dy.gather(2,idx)-dx.gather(2,idx)).square().mean((1,2))/s2
    bbox=((out.amax(1)-out.amin(1))-(x.amax(1)-x.amin(1))).square().mean(1)/s2
    ax=torch.exp(-((x[:,:,None]-x[:,None]).abs()/.01)).mean((1,2))
    ay=torch.exp(-((out[:,:,None]-out[:,None]).abs()/.01)).mean((1,2))
    axis=(ax-ay).square().mean(1)
    structure=(edge+bbox+.1*axis).mean()
    return rec+eta*structure,rec,structure
