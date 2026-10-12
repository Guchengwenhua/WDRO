"""Reproduce the failed test batch and independently check its lengths."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--preflight',type=Path,required=True);a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.rest import build_actor,RestTreeEvaluator,file_sha256
from src.geosteiner import GeoSteinerEvaluator
from src.trainer import configure_determinism,load_trainer_config
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage6.evaluation import load_group,atomic_npz,artifact

def main():
    host=json.loads(a.preflight.read_text());assert host['preflight_passed'] and a.gpu in host['eligible_gpu_indices']
    directory=EXP/'logs/stage6_test/erm_d40_s101/edge_bias';cursor=json.loads((directory/'cursor.json').read_text());m=cursor['binding']['model'];start=cursor['complete_nets']
    data,ref,sources=load_group(40,'edge_bias','test');configure_determinism(load_trainer_config());device=torch.device('cuda:0')
    ck=torch.load(m['best']['logical_path'],map_location='cpu',weights_only=False);assert file_sha256(m['best']['logical_path'])==m['best']['sha256']
    actor=build_actor(40,device);actor.load_state_dict(ck['actor_state_dict']);actor.eval()
    pins=data['pins'][start:start+256]
    with torch.no_grad():out,_=actor(pins,True)
    out=out.cpu().numpy();rest=RestTreeEvaluator();lengths=rest.lengths(pins,out);cached=ref['lstar'][start:start+256];regrets=(lengths-cached)/cached
    bad=np.flatnonzero(~np.isfinite(regrets)|(regrets<=-1e-5));records=[];gst=GeoSteinerEvaluator()
    for i in bad:
        xy=pins[i].astype(np.float64);top=out[i].reshape(-1,2)
        xlo=xy[:,0].copy();xhi=xlo.copy();ylo=xy[:,1].copy();yhi=ylo.copy()
        parent=list(range(40))
        def root(k):
            while parent[k]!=k:k=parent[k]
            return k
        for x,y in top:
            assert 0<=x<40 and 0<=y<40
            ylo[x]=min(ylo[x],xy[y,1]);yhi[x]=max(yhi[x],xy[y,1]);xlo[y]=min(xlo[y],xy[x,0]);xhi[y]=max(xhi[y],xy[x,0])
            parent[root(int(x))]=root(int(y))
        independent=float(np.sum(xhi-xlo)+np.sum(yhi-ylo));fresh=gst.length(xy)
        records.append(dict(net_index=int(start+i),net_id=data['net_ids'][start+i].decode(),rest=float(lengths[i]),rest_python=independent,
            cached_lstar=float(cached[i]),fresh_lstar=fresh,relative_regret=float(regrets[i]),connected_components=len({root(k) for k in range(40)})))
    path=EXP/'logs/stage6_negative_regret/failed_batch.npz';atomic_npz(path,dict(pins=pins,topology=out,rest_lengths=lengths,cached_lstar=cached,regret=regrets,net_ids=data['net_ids'][start:start+256]))
    report=dict(status='diagnosed',at_utc=utc_now(),failed_batch_start=start,bad_count=len(records),bad=records,preflight=str(a.preflight),gpu=a.gpu,
        input_model=m,sources=sources,batch=artifact(path),actor_rest_calls=256,additional_geosteiner_calls=len(records),all_finite=bool(np.isfinite(regrets).all()))
    atomic_write_json(EXP/'logs/stage6_negative_regret/diagnosis.json',report);print(json.dumps(report))
if __name__=='__main__':main()
