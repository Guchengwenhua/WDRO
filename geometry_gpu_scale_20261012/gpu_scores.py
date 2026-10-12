"""Conditional H/G scores on the actor device; no host copies inside decoding.

G = -sum_{a<b} p_a p_b D_ab (log p_a + log p_b).
The diagonal vanishes and symmetry removes the factor 1/2. The route
descriptor and normalization are those of the historical NumPy implementation.
Floating-point execution order changes; no bitwise-equivalence claim is made.
"""
import torch

class DecoderScore:
    def __init__(self, actor, arm):
        self.arm=arm;self.actor=actor;self.n=actor.degree;self.device=actor.device
        self.j=torch.arange(self.n,device=self.device).repeat(2)
        self.direction=torch.arange(2*self.n,device=self.device)>=self.n
        self.pa,self.pb=torch.triu_indices(2*self.n,2*self.n,offset=1,device=self.device)
        self.hooks=[actor.ptr2.register_forward_hook(self.second)]
        if arm=='geometry':self.hooks.append(actor.ptr1.register_forward_hook(self.first))

    def start(self, pins):
        self.points=torch.as_tensor(pins,device=self.device,dtype=torch.float32)
        self.row=torch.arange(len(pins),device=self.device)[:,None]
        self.scale=(self.points.amax(1)-self.points.amin(1)).sum(-1).to(torch.float64)
        self.total=torch.zeros(len(pins),device=self.device,dtype=torch.float64)
        self.steps=0

    def first(self,module,args,logits):
        self.first_index=logits.detach().argmax(-1)[:,None]

    def second(self,module,args,logits):
        logits=logits.detach().to(torch.float64)
        p=torch.softmax(logits,dim=-1);logp=p.clamp_min(1e-300).log()
        if self.arm=='entropy':
            k=torch.isfinite(logits).sum(-1)
            h=-(p*logp).sum(-1)/k.clamp_min(2).to(torch.float64).log()
            value=torch.where(k>1,h,torch.zeros_like(h))
        else:
            u=torch.where(self.direction[None,:],self.first_index,self.j[None,:])
            v=torch.where(self.direction[None,:],self.j[None,:],self.first_index)
            start=self.points[self.row,u];end=self.points[self.row,v]
            bend=torch.stack((start[:,:,0],end[:,:,1]),dim=-1)
            a=(bend-start).abs().sum(-1);c=(end-bend).abs().sum(-1);half=(a+c)/2
            f1=torch.where(a>0,half/torch.where(a>0,a,torch.ones_like(a)),torch.zeros_like(a))
            f2=torch.where(c>0,(half-a)/torch.where(c>0,c,torch.ones_like(c)),torch.zeros_like(c))
            middle=torch.where((half<=a)[...,None],start+(bend-start)*f1[...,None],bend+(end-bend)*f2[...,None])
            route=torch.stack((start,middle,end),dim=2).to(torch.float64)
            left=route[:,self.pa];right=route[:,self.pb]
            direct=(left-right).abs().sum(-1).mean(-1)
            reverse=(left-right.flip(2)).abs().sum(-1).mean(-1)
            distance=torch.minimum(direct,reverse)/self.scale[:,None]
            value=-(p[:,self.pa]*p[:,self.pb]*distance*(logp[:,self.pa]+logp[:,self.pb])).sum(-1)
        self.total.add_(value);self.steps+=1

    def result(self):
        if self.steps!=self.n-1:raise RuntimeError('Incomplete conditional decoding trace')
        return self.total/self.steps

    def close(self):
        for hook in self.hooks:hook.remove()
