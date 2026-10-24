"""Run 7-day past-diagnosis lookback sensitivity models."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age,get_cond_df
from src.streak_attrition_analysis import load_primary_config,prepare_daily_fitbit
from src.streak_timing_sensitivity_models import (add_static,complete_frame,eligible_controls,features_and_labels,latest_streaks,performance_rows,markdown_table)
from src.reviewer_sensitivity_performance_tests import fit_predict
OUT=ROOT/'results'/'past_lookback_sensitivity_7day'

def select(streaks,diag,cond,years):
 d=diag[diag.subcohort.eq(cond)].groupby('person_id',as_index=False).condition_start_date.agg(earliest='min',latest='max')
 x=streaks.merge(d); x=x[x.streak_start.ge(x.earliest)]
 if years is not None: x=x[(x.streak_start-x.latest).dt.days.le(years*365)]
 return x.sort_values('streak_start').groupby('person_id',as_index=False).tail(1)[['person_id','streak_start','streak_end']]

def main():
 OUT.mkdir(parents=True,exist_ok=True); primary=load_primary_config(); cfg=yaml.safe_load((ROOT/'conf'/'base_mdl_fitbit.yaml').read_text()); _,labels=features_and_labels(cfg)
 person=pd.read_csv(primary['person_df']); cdf=get_cond_df(primary,person); cdf['subcohort']=cdf.description.map(primary['condition_map']); diag=cdf[cdf.subcohort.notna()][['person_id','subcohort','condition_start_date']]; controls=eligible_controls(cdf,diag,primary)
 daily=prepare_daily_fitbit(primary['fitbit_summary_data'],person.person_id); streaks=filter_streak_by_age(find_strict_streaks(daily[['person_id','date']].copy(),'date',7),person.copy())
 policies={'no_lookback':None,'current_rules':primary['recency_filter'],'halved_rules':{k:v/2 for k,v in primary['recency_filter'].items()}}; rows=[]; samples=[]
 for policy,thresholds in policies.items():
  ctrl=latest_streaks(streaks,controls.person_id)
  for label in labels:
   cond=label.removeprefix('subcohort_'); cases=select(streaks,diag,cond,None if thresholds is None else thresholds[cond]); chosen=pd.concat([cases,ctrl],ignore_index=True).drop_duplicates('person_id'); cohort=chosen[['person_id','streak_start']].rename(columns={'streak_start':'index_date'}); cohort[label]=np.where(cohort.person_id.isin(cases.person_id),'case',''); cohort=add_static(cohort,person,primary); frame,features=complete_frame(cohort,chosen,daily,cfg,[label]); data=frame; nc=int(data[label].sum()); nn=len(data)-nc; samples.append({'policy':policy,'condition':cond,'n_cases':nc,'n_controls':nn})
   for model in ('logistic','xgboost'):
    _,y,prob,est=fit_predict(data,features,label,model); rows+=performance_rows(policy,'7_consecutive_days',label,model,y,prob,nc,nn,est)
 perf=pd.DataFrame(rows); sam=pd.DataFrame(samples); perf.to_csv(OUT/'performance.csv',index=False); sam.to_csv(OUT/'sample_sizes.csv',index=False)
 lines=['# Seven-day past-diagnosis lookback sensitivity','', 'Cases use the latest eligible 7-day streak after first mapped diagnosis. `no_lookback` has no upper recency limit; `current_rules` uses existing disease-specific limits; `halved_rules` uses half those limits. Controls use latest 7-day streak.','']
 for pol in policies:
  lines += ['## '+pol,'',markdown_table(sam[sam.policy.eq(pol)]),'']
  tab=[]
  for cond in sam.condition:
   r={'condition':cond}
   for m,n in [('logistic','LogReg'),('xgboost','XGBoost')]:
    z=perf[(perf.policy.eq(pol)) & (perf.condition.eq(cond)) & (perf.model.eq(m)) & (perf.metric.eq('auroc'))].iloc[0]; r[n+' AUROC (95% CI)']=f"{z.estimate:.3f} ({z.ci_lower:.3f}–{z.ci_upper:.3f})"
   tab.append(r)
  lines += [markdown_table(pd.DataFrame(tab)),'']
 (OUT/'past_lookback_sensitivity_results.md').write_text('\n'.join(lines))
if __name__=='__main__': main()
