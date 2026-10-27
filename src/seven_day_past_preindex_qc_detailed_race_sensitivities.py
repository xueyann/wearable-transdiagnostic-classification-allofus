"""Run required seven-day pre-index past-diagnosis sensitivity analyses."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.model import delong_test
from src.seven_day_past_preindex import load_inputs,frame_for
from src.streak_attrition_analysis import prepare_daily_fitbit
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age
from src.detailed_race_features import add_detailed_race_features
from src.streak_timing_sensitivity_models import performance_rows
from src.reviewer_sensitivity_performance_tests import fit_predict
OUT=ROOT/'results'/'seven_day_past_preindex_qc_detailed_race_sensitivities'

MEDICATION_WINDOW_DAYS = 30
MEDICATION_FILE = ROOT / 'data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv'


def active_medication_ids(selected):
 """Return participants exposed during the Fitbit window expanded by 30 days.

 A recorded end date takes precedence. When it is absent, the exposure is
 treated as a one-day record ending on its start date.
 """
 drug = pd.read_csv(
  MEDICATION_FILE,
  usecols=['person_id', 'drug_exposure_start_datetime', 'drug_exposure_end_datetime'],
 )
 for column in ['drug_exposure_start_datetime', 'drug_exposure_end_datetime']:
  drug[column] = pd.to_datetime(drug[column], utc=True, errors='coerce').dt.tz_localize(None)
 drug['exposure_end'] = drug['drug_exposure_end_datetime'].fillna(
  drug['drug_exposure_start_datetime']
 )
 exposure = selected.merge(
  drug[['person_id', 'drug_exposure_start_datetime', 'exposure_end']], how='left'
 )
 window_start = exposure['streak_start'] - pd.Timedelta(days=MEDICATION_WINDOW_DAYS)
 window_end = exposure['streak_end'] + pd.Timedelta(days=MEDICATION_WINDOW_DAYS)
 active = (
  exposure['drug_exposure_start_datetime'].le(window_end)
  & exposure['exposure_end'].ge(window_start)
 )
 return set(exposure.loc[active, 'person_id'])

def qualifying_count(frame,diag,primary,labels):
 out=pd.Series(0,index=frame.index)
 for lab in labels:
  c=lab.removeprefix('subcohort_'); x=frame[['person_id','index_date']].merge(diag[diag.subcohort.eq(c)],how='left')
  x=x[x.condition_start_date.le(x.index_date)]
  latest=x.groupby('person_id').condition_start_date.max(); valid=(frame.index_date-pd.to_datetime(frame.person_id.map(latest))).dt.days.le(primary['recency_filter'][c]*365)
  out+=valid.fillna(False).astype(int)
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 primary,cfg,labels,person,diag,ctrl,_,_=load_inputs(); perf=[];samples=[];comp=[];within=[];case_sets={};primary_predictions=[]
 daily=prepare_daily_fitbit(primary['fitbit_summary_data'],person.person_id,min_hr_zone_minutes=600,min_steps=100)
 streaks=filter_streak_by_age(find_strict_streaks(daily[['person_id','date']].copy(),'date',7),person.copy())
 policies=['primary','single_condition','stricter_comparator','medication_unexposed','two_date_confirmation']
 for lab in labels:
  cond=lab.removeprefix('subcohort_'); base,features,cases,selected=frame_for(cond,lab,primary,cfg,person,diag,ctrl,daily,streaks,primary['recency_filter'][cond]); base,features=add_detailed_race_features(base,features,person)
  # participant-level descriptive overlap for primary-selected cases
  cnt=qualifying_count(base,diag,primary,labels); case_sets[cond]=set(base.loc[base[lab].eq(1),'person_id']); within.append(pd.DataFrame({'condition':cond,'person_id':base.loc[base[lab].eq(1),'person_id'],'additional_conditions':cnt[base[lab].eq(1)].sub(1).values}))
  frames={'primary':base}
  frames['single_condition']=base.loc[(base[lab].eq(0))|(cnt.eq(1))].copy()
  drug=pd.read_csv(ROOT/'data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv',usecols=['person_id']);clean=set(ctrl.person_id)-set(drug.person_id);frames['stricter_comparator']=base.loc[(base[lab].eq(1))|base.person_id.isin(clean)].copy()
  blocked=active_medication_ids(selected);frames['medication_unexposed']=base.loc[~base.person_id.isin(blocked)].copy()
  two,two_features,_,_=frame_for(cond,lab,primary,cfg,person,diag,ctrl,daily,streaks,primary['recency_filter'][cond],min_dates=2);two,two_features=add_detailed_race_features(two,two_features,person);frames['two_date_confirmation']=two
  primary_pred={}
  for a,frame in frames.items():
   nc=int(frame[lab].sum());nn=len(frame)-nc;samples.append({'analysis':a,'condition':cond,'n_cases':nc,'n_controls':nn})
   for model in ('logistic','xgboost'):
    ids,y,p,est=fit_predict(frame,features,lab,model);perf+=performance_rows(a,'7_consecutive_preindex',lab,model,y,p,nc,nn,est)
    pred=pd.DataFrame({'person_id':ids,'y':y,'p':p}).set_index('person_id')
    if a=='primary':
     primary_pred[model]=pred
     primary_predictions.append(pd.DataFrame({'person_id':ids,'condition':cond,'model':model,'y_true':y,'probability':p,'estimator':est}))
     continue
    joined=pred.join(primary_pred[model][['p']],how='inner',rsuffix='_primary')
    if len(joined) and joined.y.nunique()==2:
     test=delong_test(joined.y.values,joined.p_primary.values,joined.p.values,alternative='two-sided')
     comp.append({'analysis':a,'condition':cond,'model':model,'n_test':len(joined),'auc_primary':roc_auc_score(joined.y,joined.p_primary),'auc_sensitivity':roc_auc_score(joined.y,joined.p),'difference':test['mean'],'ci_lower':test['ci_lower'],'ci_upper':test['ci_upper'],'p_value':test['p_value']})
 matrix=[{'row':a,'column':b,'n':len(case_sets[a]&case_sets[b])} for a in case_sets for b in case_sets]
 pd.DataFrame(perf).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(samples).to_csv(OUT/'sample_sizes.csv',index=False);pd.concat(primary_predictions,ignore_index=True).to_csv(OUT/'primary_test_predictions.csv',index=False);c=pd.DataFrame(comp);c['p_fdr_bh']=multipletests(c.p_value,method='fdr_bh')[1];c.to_csv(OUT/'matched_test_delong.csv',index=False);overlap = pd.DataFrame(matrix); overlap.to_csv(OUT/'descriptive_overlap_long.csv',index=False); overlap.pivot(index='row', columns='column', values='n').to_csv(OUT/'descriptive_overlap_matrix.csv'); pd.concat(within).to_csv(OUT/'within_cohort_additional_conditions.csv',index=False)
if __name__=='__main__':main()
