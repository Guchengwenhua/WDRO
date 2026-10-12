"""Unchanged legacy REST action space and exact precision_v2 regret."""
import numpy as np
import torch
from core import ROOT, MAIN
from src.rest import build_actor,load_trusted_legacy_actor,RestTreeEvaluator,LEGACY_ROOT
from src.geosteiner import GeoSteinerEvaluator
from src.candidates import ReferenceCache

class Adapter(torch.nn.Module):
    def __init__(self,degree,device,params):
        super().__init__();self.device=device
        self.actor=build_actor(degree,device)
        self.initialization=load_trusted_legacy_actor(self.actor,LEGACY_ROOT/f'save/DAC21/rsmt{degree}b.pt',map_location=device)
        self.rest=RestTreeEvaluator();self.geo=GeoSteinerEvaluator()
        self.cache=ReferenceCache(params.get('cache_path',ROOT/'artifacts/tmp/rest_smoke.sqlite'))
        self.reference_misses=0
    def evaluate(self,pins,greedy):
        output,log_prob=self.actor(pins,greedy)
        length=self.rest.lengths(pins,output.detach().cpu().numpy())
        reference,miss=self.cache.lengths(pins,self.geo);self.reference_misses+=miss
        regret=length/reference-1
        if np.min(regret)<-1e-5: raise ValueError('REST length is shorter than exact reference')
        return torch.as_tensor(regret,dtype=torch.float32,device=self.device),log_prob,{
            'length':length,'lstar':reference,'relative_regret':regret}
    def close(self): self.cache.close()
