"""Fit and compare 7-day past-diagnosis logistic coefficients."""
import sys
import argparse
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age,get_cond_df
from src.streak_attrition_analysis import load_primary_config,prepare_daily_fitbit
from src.streak_timing_sensitivity_models import condition_timing_cohort,complete_frame,eligible_controls,features_and_labels

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--days',type=int,default=7); args=parser.parse_args(); days=args.days
 out=ROOT/'results'/f'past_{days}day_logreg_coefficients';out.mkdir(parents=True,exist_ok=True)
 import yaml
 primary=load_primary_config();cfg=yaml.safe_load((ROOT/'conf'/'base_mdl_fitbit.yaml').read_text());_,labels=features_and_labels(cfg)
 person=pd.read_csv(primary['person_df']);cdf=get_cond_df(primary,person);cdf['subcohort']=cdf.description.map(primary['condition_map']);diag=cdf[cdf.subcohort.notna()][['person_id','subcohort','condition_start_date']];controls=eligible_controls(cdf,diag,primary)
 daily=prepare_daily_fitbit(primary['fitbit_summary_data'],person.person_id);streaks=filter_streak_by_age(find_strict_streaks(daily[['person_id','date']].copy(),'date',days),person.copy());rows=[]
 for label in labels:
  cohort,sel=condition_timing_cohort('past',label,streaks,diag,controls,person,primary);frame,features=complete_frame(cohort,sel,daily,cfg,[label]);train=frame[frame.split.eq('train')]; y=train[label]; model=sm.Logit(y,sm.add_constant(train[features])).fit(disp=False,maxiter=350);ci=model.conf_int();tab=pd.DataFrame({'condition':label.removeprefix('subcohort_'),'feature':model.params.index,'coef':model.params.values,'ci_lower':ci[0].values,'ci_upper':ci[1].values,'p_value':model.pvalues.values});tab['p_fdr_bh']=multipletests(tab.p_value,method='fdr_bh')[1];rows.append(tab)
 res=pd.concat(rows,ignore_index=True);res.to_csv(out/'logreg_coefficients.csv',index=False)
 sig=res[(res.feature!='const')&(res.p_fdr_bh<.05)].copy();sig['direction']=sig.coef.map(lambda x:'higher odds' if x>0 else 'lower odds');sig.to_csv(out/'fdr_significant_coefficients.csv',index=False)
 (out/'README.md').write_text(f'Coefficients use unregularized statsmodels Logit on the deterministic seed-42 70% training partition of the {days}-day, past-diagnosis, current-lookback cohort. FDR correction is within disease area across model coefficients.\n')
if __name__=='__main__':main()
