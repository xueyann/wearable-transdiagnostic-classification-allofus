"""Shared builders for the corrected seven-day pre-index past-diagnosis cohort."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age,get_cond_df
from src.streak_attrition_analysis import load_primary_config,prepare_daily_fitbit
from src.streak_timing_sensitivity_models import add_static,complete_frame,eligible_controls,features_and_labels,latest_streaks

def select_cases(streaks, diagnoses, condition, lookback, min_dates=1):
 d=diagnoses[diagnoses.subcohort.eq(condition)][['person_id','condition_start_date']].drop_duplicates()
 rows=[]
 for pid,g in d.groupby('person_id'):
  for s in streaks[streaks.person_id.eq(pid)].itertuples(index=False):
   pre=g[g.condition_start_date.le(s.streak_start)]
   if pre.condition_start_date.nunique()<min_dates: continue
   if lookback is not None and (s.streak_start-pre.condition_start_date.max()).days>lookback*365: continue
   rows.append((pid,s.streak_start,s.streak_end))
 return pd.DataFrame(rows,columns=['person_id','streak_start','streak_end']).sort_values('streak_start').groupby('person_id',as_index=False).tail(1) if rows else pd.DataFrame(columns=['person_id','streak_start','streak_end'])

def load_inputs():
 primary=load_primary_config(); cfg=yaml.safe_load((ROOT/'conf'/'base_mdl_fitbit.yaml').read_text());_,labels=features_and_labels(cfg);person=pd.read_csv(primary['person_df']);cdf=get_cond_df(primary,person);cdf['subcohort']=cdf.description.map(primary['condition_map']);diag=cdf[cdf.subcohort.notna()][['person_id','subcohort','condition_start_date']];ctrl=eligible_controls(cdf,diag,primary);daily=prepare_daily_fitbit(primary['fitbit_summary_data'],person.person_id);streaks=filter_streak_by_age(find_strict_streaks(daily[['person_id','date']].copy(),'date',7),person.copy());return primary,cfg,labels,person,diag,ctrl,daily,streaks

def frame_for(condition,label,primary,cfg,person,diag,ctrl,daily,streaks,lookback,min_dates=1,control_filter=None):
 cases=select_cases(streaks,diag,condition,lookback,min_dates); controls=ctrl if control_filter is None else control_filter(ctrl.copy()); control_streaks=latest_streaks(streaks,controls.person_id);selected=pd.concat([cases,control_streaks],ignore_index=True).drop_duplicates('person_id');cohort=selected[['person_id','streak_start']].rename(columns={'streak_start':'index_date'});cohort[label]=np.where(cohort.person_id.isin(cases.person_id),'case','');cohort=add_static(cohort,person,primary);frame,features=complete_frame(cohort,selected,daily,cfg,[label]);return frame,features,cases,selected
