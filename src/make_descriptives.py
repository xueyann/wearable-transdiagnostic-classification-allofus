import sys
sys.path.append('./')
import os
import re
from pathlib import Path
from lib.descriptives import generate_descriptives_split
import pandas as pd
import numpy as np
import yaml
from functools import reduce


def main():

    config = yaml.safe_load(open('conf/make_descriptives.yaml'))

    ### Get feature names ###
    original_demo_features = config['original_demo_features']
    static_features = config['static_features']
    survey_features = config['survey_features']
    time_series_features = config['time_series_features']
    sumstats_features = [f"{x}_{suffix}" for x in time_series_features for suffix in ['mean', 'std']]
    feature_rename_map = config['feature_rename_map']
    drug_class_rename_map = config['drug_class_rename_map']
    subcohort_cols = config['subcohort_cols']
    # categorical features
    categorical_vars = config['categorical_vars']
    continuous_vars = [x for x in original_demo_features + static_features + sumstats_features if x not in categorical_vars]

    ### Get all the dataframes ###
    person_df = pd.read_csv(config['person_df'])
    cohort = pd.read_csv(config['cohort'])
    survey_df = pd.read_csv(config['survey_df']).astype(int)
    sumstats_df = pd.read_csv(config['time_series_sumstats'])
    drug_df = pd.read_csv(config['drug_df'])
    
    ### Get drug mapping and map drug names to classes ###
    drug_mapping = yaml.safe_load(open('conf/drug_list.yaml'))
    drug_mapping = pd.DataFrame(pd.Series(drug_mapping)).reset_index()
    drug_mapping = drug_mapping.rename(columns = {0: 'drug_name', 'index': 'drug_class'})
    drug_mapping = drug_mapping.explode('drug_name')
    # map drug name to classes
    drug_df = drug_df.merge(drug_mapping).drop_duplicates()
    # merge drug df with cohort
    drug_df = drug_df.merge(cohort[['person_id'] + subcohort_cols])

    ### Merge all variables into a long df ###
    df = cohort[['person_id'] + subcohort_cols + static_features]\
        .merge(person_df[['person_id'] + original_demo_features])\
        .merge(survey_df[['person_id'] + survey_features])\
        .merge(sumstats_df[['person_id'] + sumstats_features])

    ### map some features back to the categories ###
    for feat, value_map in config['feature_value_map'].items():
        df[feat] = df[feat].map(lambda x: value_map.get(int(x)) if not np.isnan(x) else np.nan)
    
    desc = []
    desc.append(generate_descriptives_split(df.loc[df[subcohort_cols].isna().all(axis=1)],
                                            drug_df.loc[drug_df[subcohort_cols].isna().all(axis=1)],
                                            'Control', 
                                            continuous_vars = continuous_vars, 
                                            categorical_vars = categorical_vars, 
                                            var_map = feature_rename_map,
                                            drug_rename_map = drug_class_rename_map))
    for subcohort in subcohort_cols:
        desc.append(generate_descriptives_split(df.loc[~df[subcohort].isna()],
                                                drug_df.loc[~drug_df[subcohort].isna()],
                                                subcohort.replace('subcohort_', ''), 
                                                continuous_vars = continuous_vars, 
                                                categorical_vars = categorical_vars, 
                                                var_map = feature_rename_map,
                                                drug_rename_map = drug_class_rename_map))
        desc_df = reduce(lambda x, y: x.join(y), desc)

    # Make sure directory exists
    os.makedirs(os.path.dirname(Path(config['output'])), exist_ok = True)

    latex = desc_df.to_latex(None,
                        index=True,
                        escape = True, 
                        float_format="%.2f",
                        bold_rows = True,
                        multicolumn_format = 'c',
                        longtable = True)
    # some manual modification of the latex file
    # specify the width of the first column so latex can wrap the text in cell
    latex = re.sub(r'\\multirow\[t\]\{(\d+)\}\{\*\}',
                    r'\\multirow[t]{\1}{2.1cm}',
                    latex)
    latex = latex.replace(r"\begin{longtable}{lllllllllll}", 
                          r"""\begin{landscape}
                                \footnotesize
                                \begin{longtable}{>{\raggedright\arraybackslash}p{2.1cm}l>{\raggedright\arraybackslash}m{2.6cm}p{1.95cm}p{1.95cm}p{1.95cm}p{1.95cm}p{1.95cm}p{1.95cm}p{1.95cm}p{1.95cm}}""")
    latex = latex.replace(r""" &  &  & Control & CVD & Diabetes & OSA & MDD & Anxiety & Bipolar & ADHD \\""", 
                          r"""  &  &  & Control & CVD & Diabetes & OSA & MDD & Anxiety & Bipolar & ADHD \\
                                \cmidrule(rl){4-4}
                                \cmidrule(rl){5-5}
                                \cmidrule(rl){6-6}
                                \cmidrule(rl){7-7}
                                \cmidrule(rl){8-8}
                                \cmidrule(rl){9-9}
                                \cmidrule(rl){10-10}
                                \cmidrule(rl){11-11}""")
    latex = latex.replace(r"""\end{longtable}""",
                        r"""\end{longtable}
                            \end{landscape}""")
    with open(config['output'], 'w') as f:
        f.write(latex)
        f.write


    #### descriptive of medications ####
    return


if __name__ == '__main__':
    main()