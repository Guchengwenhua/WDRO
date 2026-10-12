"""Choose one common REST recipe using validation only, including all attempts."""
from exp_common import *

def main():
    if (ART/'done'/'recipe_selection.json').exists():return
    rows=[];parents=[]
    for recipe in CFG['recipe_selection']['tie_order']:
        scores=[]
        for seed in CFG['recipe_selection']['seeds']:
            for method in CFG['recipe_selection']['methods']:
                s=read_json(ART/'diagnostics'/f'{recipe}_{method}_s{seed}'/'summary.json')
                vals=s['validations'];last=next(v for v in vals if v['update']==256);zero=next(v for v in vals if v['update']==0)
                score=last.get('macro',last.get('design_macro_mean_regret'));baseline=zero.get('macro',zero.get('design_macro_mean_regret'))
                scores.append(score)
                if recipe==CFG['recipe_selection']['tie_order'][0]:parents.append(baseline)
                rows.append(dict(recipe=recipe,seed=seed,method=method,val_macro_256=score,val_macro_0=baseline,
                    delta_vs_parent=score-baseline,new_training_seconds=s.get('new_training_seconds',0.),reused=bool(s.get('reused_from'))))
    ranking=[dict(recipe=r,mean_val_macro=float(np.mean([x['val_macro_256'] for x in rows if x['recipe']==r]))) for r in CFG['recipe_selection']['tie_order']]
    chosen=min(ranking,key=lambda x:x['mean_val_macro']);baseline=float(np.mean(parents))
    result=dict(**chosen,parent_mean_val_macro=baseline,all_recipes_degrade=all(x['mean_val_macro']>=baseline for x in ranking),
        ranking=ranking,selection='A_val only; last@256 mean over ERM/RAW x seeds101/102',test_access=False,time=utc())
    write_csv(ROOT/'tables'/'recipe_selection.csv',rows);atomic_json(ART/'recipe_selected.json',result);mark_done('recipe_selection',**result)

if __name__=='__main__':main()
