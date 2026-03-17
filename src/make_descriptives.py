import sys
sys.path.append('./')
import os
import re
from pathlib import Path
from lib.descriptives import generate_descriptives_split, generate_descriptive_and_comparison
import pandas as pd
import numpy as np
import yaml
from functools import reduce
import warnings
warnings.filterwarnings("ignore")


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
    person_df['race'] = person_df['race'].fillna('Unknown')
    person_df['ethnicity'] = person_df['ethnicity'].fillna('Unknown')
    cohort = pd.read_csv(config['cohort'])
    survey_df = pd.read_csv(config['survey_df'])
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
        .merge(person_df[['person_id'] + original_demo_features], on = 'person_id')\
        .merge(survey_df[['person_id'] + survey_features], on = 'person_id')\
        .merge(sumstats_df[['person_id'] + sumstats_features], on = 'person_id')
        # .dropna(subset = original_demo_features)
    df = df.dropna(subset = 'sex_at_birth')
    
    # print(df.loc[df[static_features+original_demo_features].isna().any(axis=1), ['person_id'] + static_features+original_demo_features])
    # print(df.loc[df['sex_at_birth'].isna()].shape[0])
    # print(person_df.loc[person_df['person_id'].isin(df.loc[df['sex_at_birth'].isna(), 'person_id'])])
    # print(df.shape[0])
    # sys.exit()

    
    # ### map some features back to the categories ###
    # for feat, value_map in config['feature_value_map'].items():
    #     df[feat] = df[feat].map(lambda x: value_map.get(int(x)) if not np.isnan(x) else np.nan)
    
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
                        caption = 'Descriptives of demographic, lifestyle and wearable-derived features and medication across subcohorts',
                        label = 'tab:descriptives',
                        longtable = True)
    # some manual modification of the latex file
    # specify the width of the first column so latex can wrap the text in cell
    latex = re.sub(r'\\multirow\[t\]\{(\d+)\}\{\*\}',
                    r'\\multirow[t]{\1}{2.1cm}',
                    latex)
    latex = latex.replace(r"\begin{longtable}{lllllllllll}", 
                          r"""\begin{landscape}
                                \footnotesize
                                \begin{longtable}{{>{\setstretch{0.9}\raggedright\arraybackslash}p{2.1cm}l>{\setstretch{0.9}\raggedright\arraybackslash}m{2.6cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}>{\setstretch{0.9}}p{1.95cm}}}""")
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

    ###################################################################################
    #### Concise set of descriptives ####   
    ###################################################################################
    concise_features = [tuple(x) for x in config['concise_features']]
    # remove the IQR for medians
    median_features = [tuple(x) for x in config['concise_features'] if 'Median (IQR)' in x]
    # desc_df.to_csv('results/temp/desc.csv', index=True)
    concise_df = desc_df.loc[concise_features].copy()
    concise_df.loc[median_features] = concise_df.loc[median_features]\
                                    .map(lambda x: float(re.sub(r"\([^)]*\)", "", x)))
    # remove IQR from index
    concise_df.index = pd.MultiIndex.from_tuples([tuple({'Median (IQR)': 'Median'}.get(x, x) for x in tup) \
                                                for tup in config['concise_features']])
    print(concise_df)
    
    concise_latex = concise_df.to_latex(None,
            index=True,
            escape = True, 
            float_format="%.2f",
            bold_rows = True,
            multicolumn_format = 'c',
            caption = 'Descriptives of key demographic and lifestyle features across subcohorts',
            label = 'tab:concise_descriptives',
            column_format = r""">{\setstretch{0.9}\raggedright\arraybackslash}p{1.5cm}
            >{\setstretch{0.9}\raggedright\arraybackslash}m{1.5cm}
            >{\setstretch{0.9}\raggedright\arraybackslash}p{1.5cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}
            >{\setstretch{0.9}}p{1.3cm}""",
            longtable = True)
    with open(config['output_concise'], 'w') as f:
        f.write(r'{\footnotesize' + concise_latex + '}')

    ###################################################################################
    #### Compare control cohort with a clean version that has no medication at all ####   
    ###################################################################################
    # create a clean control cohort with no 
    df_control = df.loc[(df[subcohort_cols].isna().all(axis=1))]
    df_control_clean = df.loc[(df[subcohort_cols].isna().all(axis=1)) & 
                            (~df['person_id'].isin(drug_df['person_id']))]

    # TODO compare between the two cohorts
    comp_with_clean = generate_descriptive_and_comparison(df_control, df_control_clean,
                                        cohort_names = ("Control", "Control (Clean)"),
                                        continuous_vars = continuous_vars,
                                        categorical_vars = categorical_vars,
                                        var_map = feature_rename_map)

    # comp_with_clean.to_csv('results/temp/comp_with_clean.csv', index=True)
    
    comp_latex = comp_with_clean.to_latex(None,
                                        index=True,
                                        escape = True, 
                                        float_format="%.2f",
                                        bold_rows = True,
                                        multicolumn_format = 'c',
                                        caption = 'Comparison of demographic, lifestyle and wearable-derived features between Control and Clean Control cohort',
                                        label = 'tab:cleancontrolcomp',
                                        column_format = r""">{\setstretch{0.9}\raggedright\arraybackslash}p{2.1cm}
                                                                >{\setstretch{0.9}\raggedright\arraybackslash}m{2.6cm}
                                                                >{\setstretch{0.9}\raggedright\arraybackslash}p{1.95cm}
                                                                >{\setstretch{0.9}}p{1.95cm}
                                                                >{\setstretch{0.9}}p{1.95cm}
                                                                >{\setstretch{0.9}}p{1.95cm}
                                                                >{\setstretch{0.9}}p{1.95cm}
                                                                >{\setstretch{0.9}}p{1.95cm}""",
                                        longtable = True)
    # formatting: multirow wrap text
    comp_latex = re.sub(r'\\multirow\[t\]\{(\d+)\}\{\*\}',
                    r'\\multirow[t]{\1}{2.1cm}',
                    comp_latex)
    comp_latex = comp_latex.replace(r'cline{2-7}', r'cline{2-5}')
    comp_latex = comp_latex.replace(r'\$<0.001*\$', r'$<0.001*$')
    with open(config['output_clean_control'], 'w') as f:
        f.write(r'{\footnotesize' + comp_latex + '}')
    return



if __name__ == '__main__':
    main()