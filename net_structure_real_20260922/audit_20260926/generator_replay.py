from common import *
from net_models import Denoiser
import torch,time

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.05);start=time.monotonic();data=npz(ART/'data.npz');v=npz(ART/'multidegree/d30_A_val.npz');original=list(csv.DictReader((ROOT/'tables/generator_test.csv').open()));lookup={(r['kind'],int(r['seed']),r['split'],float(r['sigma']),r['design']):r for r in original};rows=[];maxerr=0.;predictions=0;prepost=[]
 for folder in sorted((ART/'generators').glob('gen_formal_*')):
  cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=Denoiser(cp['spec']['kind'],cp['spec']['width']).cuda();m.load_state_dict(cp['model']);m.eval();kind=cp['spec']['kind'];seed=cp['seed']
  for split in ['A_val','B_test','C_external']:
   if split=='A_val':x=v['pins'];ds=v['design'];rng=np.random.default_rng(99881)
   else:take=data['split']==split;x=data['pins'][take];ds=data['design_ids'][take];rng=np.random.default_rng(66492)
   scores=[]
   for sigma in CFG['generator']['sigmas']:
    noise=rng.normal(size=x.shape).astype(np.float32)*sigma;y=x+noise;out=[]
    with torch.no_grad():
     for b in range(0,len(x),128):out.append(m(torch.tensor(y[b:b+128],device='cuda'),torch.full((len(y[b:b+128]),),sigma,device='cuda')).cpu().numpy())
    out=np.concatenate(out);err=((out-x)**2).mean((1,2))/sigma**2;identity=(noise**2).mean((1,2))/sigma**2;predictions+=len(x)
    for design in sorted(set(ds)):
     score=float(err[ds==design].mean());scores.append(score)
     if split!='A_val':
      saved=float(lookup[kind,seed,split,sigma,design]['normalized_mse']);maxerr=max(maxerr,abs(score-saved))
     rows.append(dict(kind=kind,seed=seed,split=split,sigma=sigma,design=design,normalized_mse=score,identity_mse=float(identity[ds==design].mean())))
   if split=='A_val':maxerr=max(maxerr,abs(np.mean(scores)-cp['best']['score']))
  pre=torch.load(folder/'pre_last.pt',map_location='cpu',weights_only=False);prepost.append(dict(kind=kind,seed=seed,pre_last_score=pre['history'][-1]['validation']['score'],post_best_score=cp['best']['score'],post_improvement=pre['history'][-1]['validation']['score']-cp['best']['score']))
  print('GENERATOR_REPLAY',kind,seed,'max_error',maxerr,flush=True)
 csvout('generator_replayed_rows.csv',rows);csvout('generator_pre_post.csv',prepost);write(OUT/'generator_replay.json',dict(status='PASS' if maxerr<1e-7 else 'FAIL',max_mse_error=maxerr,predictions=predictions,rows=len(rows),seconds=time.monotonic()-start,time=now()))
if __name__=='__main__':main()
