import pandas as pd
import yaml
import os
import numpy as np



def simplify_cat_ans(qn, ans):
    qn = qn.split(': ')[1]
    return ans.replace(f'{qn}: ', '')

def process_survey_df(df, config):

    # filter df to only include relevant questions
    qn_list = list(config['ordinal_encoding'].keys()) + config['continuous_var'] + config['categorical_vars']
    df = df.loc[df['question'].isin(qn_list)]

    unknown_ans = df.loc[df['answer'].str.startswith('PMI:'), 'answer'].unique()

    # simplify the answers
    df['answer'] = df.apply(lambda x: simplify_cat_ans(x['question'], x['answer']), axis=1)
    # standardise unknown answers to unknown
    df.loc[df['answer'].isin(unknown_ans), 'answer'] = 'Unknown'

    # check no duplicate questions
    assert df[['person_id', 'question']].drop_duplicates().shape[0] == df.shape[0]

    # ordinal encoding for answers
    for qn, ans_map in config['ordinal_encoding'].items():
        df.loc[df['question'] == qn, 'answer'] = df.loc[df['question'] == qn, 'answer'].apply(lambda x: ans_map.get(x, x))

    # pivot questions out
    df = df.pivot(columns = 'question', values = 'answer', index = 'person_id')

    # binning for categorical vars
    for var, var_map in config['categorical_bins'].items():
        df[var] = df[var].map(lambda x: var_map.get(x) if x in var_map.keys() else x)
        print(df[var].unique())

    # smoking other than cigarettes
    df['other_smoking'] = df[config['other_smoking']].all(axis=1)
    df = df.drop(columns = config['drop_col_other_smoking'])
    
    # rename columns for cleaner names
    df = df.rename(columns = config['rename_map'])
    print(df.columns)

    ######## clean up alcohol #######
    # people who answered yes to alcohol drinking but zero past year drink frequency - likely just social drinkers
    # df.loc[(df['alcohol_drinking']==True) & (df['past_year_drink_freq']==0)]
    # assign 0 past year drink frequency for those answered no to alcohol drinking
    df.loc[df['alcohol_drinking']=='No', 'past_year_drink_freq'] = 'Never'
    df.loc[df['alcohol_drinking']=='No', 'daily_drink_count'] = 0
    df.loc[df['alcohol_drinking']=='No', 'binge_drinking_freq'] = 'Never In Last Year'

    df.loc[df['alcohol_drinking']=='Unknown', 'daily_drink_count'] = 'Unknown'
    df.loc[df['alcohol_drinking']=='Unknown', 'binge_drinking_freq'] = 'Unknown'


    df.loc[df['ever_smoker']=='No', 'daily_cig_count'] = 0
    df.loc[df['ever_smoker']=='No', 'smoke_freq'] = 'Never'
    df.loc[df['ever_smoker']=='Unknown', 'smoke_freq'] = 'Unknown'
    
    # create a flag for former smoker
    df.loc[((df['ever_smoker']=='Yes') & (df['smoke_freq']=='Not At All')), 'smoke_freq'] = 'Former smoker'
    # create a category of never smoker
    df.loc[df['ever_smoker']=='No', 'smoke_freq'] = 'Never smoker'
    # # assign binge drinking freq to 0 if past year drinking frequency is 0
    # df.loc[df['past_year_drink_freq']==0, 'binge_drinking_freq'] = 0

    ######## clean up smoking ########
    df.loc[df['smoke_quit_age']=='Unknown', 'smoke_quit_age'] = np.nan
    df.loc[df['smoke_start_age']=='Unknown', 'smoke_start_age'] = np.nan
    df.loc[df['smoke_years']=='Unknown', 'smoke_years'] = np.nan
    # if current daily cig count is not 0, assume person didn't quit
    df.loc[~df['current_daily_cig_count'].isin(['0', 0]), 'smoke_quit_age'] = np.nan

    # check whether reported smoking years match smoke start and quit age
    df_match_smoke_years = df.loc[(df['smoke_quit_attempt']==True), ['smoke_start_age', 'smoke_quit_age']].dropna().astype(int)
    df_match_smoke_years['smoke_years_cal'] = df_match_smoke_years['smoke_quit_age'] - df_match_smoke_years['smoke_start_age'] 
    # if smoke quit age is lower than smoke start age, assume person did not quit
    person_no_quit = list(df_match_smoke_years.loc[df_match_smoke_years['smoke_years_cal']<0].index)
    df.loc[person_no_quit, 'smoke_quit_age'] = np.nan
    # smoke years
    df = df.join(df_match_smoke_years[['smoke_years_cal']], how = 'left')
    df.loc[df['smoke_years'].isna(), 'smoke_years'] = 0
    df.loc[df['smoke_years_cal'].isna(), 'smoke_years_cal'] = 0
    # choose the longer one for smoke years
    df['smoke_years'] = df[['smoke_years', 'smoke_years_cal']].astype(int).max(axis=1)



    # save a copy for no one hot encode --> descriptives
    df_cat = df.copy()
    df_cat = df_cat[config['keep_cols_cat']]
    df_cat = df_cat.reset_index()

    # # one hot encoding for some variables
    # print(df[config['one_hot_encode']].isna().any(axis=1))
    # print(df.loc[df[config['one_hot_encode']].isna().any(axis=1)])
    for col in config['one_hot_encode']:
        df[col] = df[col].fillna('Unknown')
    df = pd.get_dummies(df, columns = config['one_hot_encode'])

    df = df[config['keep_cols']]
    # print(df.loc[(df=='Unknown').any(axis=1)])
    df = df.reset_index()

    return df, df_cat

# def handle_missing(df):
#     # create a missing flag and convert unknowns to 0 
#     df['missing_flag'] = False
#     for col in ['alcohol_drinking', 'ever_smoker', 'binge_drinking_freq', 'daily_drink_count',\
#                 'binge_drinking_freq', 'daily_cig_count', 'past_year_drink_freq']:
#         df.loc[df[col]=='Unknown', 'missing_flag'] = True

#     # map everything to 0 and 1
#     df[['alcohol_drinking', 'ever_smoker', 'other_smoking', 'missing_flag']] = \
#         df[['alcohol_drinking', 'ever_smoker', 'other_smoking', 'missing_flag']]\
#         .map(lambda x: {'True': 1, 'False': 0, True: 1, False: 0}.get(x, x))

#     # # fillna and convert everything to integer
#     # df = df.fillna(0)
#     # df = df.astype(int)
#     df.loc[df.isna().any(axis=1)]

#     return df 

def main():
    config = yaml.safe_load(open('conf/dp_survey.yaml'))
    processed_df = []
    processed_df_cat = []
    i = 0
    while os.path.exists(config['survey_df_path_pattern'].format(i=i)):
        df = pd.read_csv(config['survey_df_path_pattern'].format(i=i))
        df, df_cat = process_survey_df(df, config)
        processed_df.append(df)
        processed_df_cat.append(df_cat)
        i += 1
    processed_df = pd.concat(processed_df)
    processed_df.to_csv(config['savepath'], index=False)
    processed_df_cat = pd.concat(processed_df_cat)
    processed_df_cat.to_csv(config['savepath_cat'], index=False)


if __name__ == '__main__':
    main()
