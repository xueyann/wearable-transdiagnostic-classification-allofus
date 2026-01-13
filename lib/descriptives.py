import numpy as np
import pandas as pd
# Statistical tests
from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    ks_2samp,
    iqr
)

def compute_cohens_d(x1, x2):
    """
    Function to compute cohen's d"""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(x1) - np.mean(x2)) / pooled_std

def compare_two_cohorts_side_by_side(df1, df2, features, 
                                     label1='group1', 
                                     label2='group2',
                                     save_folder = None):
    """
    Generate descriptive comparison tables for the two cohorts"""
    results = []

    for feat in features:
        data1 = df1[feat].dropna()
        data2 = df2[feat].dropna()

        # Basic descriptive stats
        stat_dict = {
            'feature': feat,
            f'{label1}_mean': data1.mean(),
            f'{label2}_mean': data2.mean(),
            f'{label1}_std': data1.std(),
            f'{label2}_std': data2.std(),
            f'{label1}_median': data1.median(),
            f'{label2}_median': data2.median(),
            f'{label1}_iqr': iqr(data1),
            f'{label2}_iqr': iqr(data2),
            f'{label1}_n': len(data1),
            f'{label2}_n': len(data2),
            f'{label1}_missing': df1[feat].isna().sum(),
            f'{label2}_missing': df2[feat].isna().sum(),
        }

        # Normality checks
        try:
            normal1 = shapiro(data1)[1] > 0.05
            normal2 = shapiro(data2)[1] > 0.05
        except:
            normal1 = normal2 = False

        # Variance equality check
        try:
            var_equal = levene(data1, data2)[1] > 0.05
        except:
            var_equal = False

        # Choose appropriate test
        if normal1 and normal2:
            test_used = 't-test'
            stat, p_value = ttest_ind(data1, data2, equal_var=var_equal)
        else:
            test_used = 'Mann–Whitney U'
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

        # KS test
        ks_stat, ks_p = ks_2samp(data1, data2)

        # Effect size
        cohen_d = compute_cohens_d(data1, data2)

        # Add test results
        stat_dict.update({
            'test_used': test_used,
            'p_value': p_value,
            'ks_p_value': ks_p,
            'cohens_d': cohen_d,
            'normal_group1': normal1,
            'normal_group2': normal2,
            'var_equal': var_equal
        })

        results.append(stat_dict)
    
    results = pd.DataFrame(results)
    results = results.sort_values('cohens_d', ascending=False)
    
    if save_folder:
        results.to_csv(f"{save_folder}/comp.csv")

    return results

import pandas as pd
import numpy as np

def get_drug_descriptives(df, 
                          subcohort_col,
                          drug_rename_map = None):
    n = df['person_id'].nunique()
    df = df[['person_id', 'drug_class']].groupby('drug_class').nunique()
    subcohort_col = subcohort_col.replace('subcohort_', '')
    df[subcohort_col] = (df['person_id']/n).apply(lambda x: f"{'{:.1%}'.format(x)}")
    df[subcohort_col] = df.apply(lambda x: f'{x['person_id']} ({x[subcohort_col]})', axis=1)

    if drug_rename_map is not None:
        df.index = df.index.map(drug_rename_map)
    return df[[subcohort_col]]

def generate_descriptives_split(df, 
                                drug_df,
                                subcohort_name, 
                                continuous_vars = ['age', 'waist_circum', 'hip_circum', 'BMI'], 
                                categorical_vars = ['sex_at_birth', 'race', 'ethnicity'], 
                                var_map = None, 
                                drug_rename_map = None):
    """
    Generate R-style descriptive statistics for continuous and categorical variables.

    Input dataframes:
    df: contains all the variables desired
    drug_df: contains drug df, it is unique because prescriptions are not mutually exclusive

    Returns two LaTeX-ready DataFrames:
    1. continuous_df: One row per variable with mean (SD), median (IQR)
    2. categorical_df: Multirow-compatible table of counts and percentages
    """
    ### CONTINUOUS ###
    cont_rows = []
    for var in continuous_vars:
        series = df[var]
        n = series.count()
        missing = series.isna().sum() # keep na in series so it can count missing
        mean = series.mean()
        std = series.std()
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        cont_rows.append({
            'Variable': var,
            'N (Missing %)': f"{n} ({missing:.2f})",
            'Mean (SD)': f"{mean:.2f} ({std:.2f})",
            'Median (IQR)': f"{median:.2f} ({q1:.2f} - {q3:.2f})"
        })
    continuous_df = pd.DataFrame(cont_rows)
    # formatting
    continuous_df = pd.DataFrame(continuous_df.set_index('Variable').stack(), 
                                 columns = [subcohort_name])    
    continuous_df = continuous_df.reset_index().rename(columns={'level_1': 'Measure'})
    continuous_df.insert(2, 'Value', '')

    ### CATEGORICAL ###
    cat_rows = []
    for var, values in categorical_vars.items():
        counts = df[var].value_counts(dropna=False)
        percents = df[var].value_counts(normalize=True, dropna=False) * 100
        first = True
        for level in values: # Use the order of values specified
            if level in list(counts.index):
                label = str(level) if pd.notna(level) else 'Unknown'
                row = {
                    'Variable': var,
                    'Value': label,
                    subcohort_name: f"{counts[level]} ({percents[level]:.2f})"
                }
                cat_rows.append(row)
                first = False
    categorical_df = pd.DataFrame(cat_rows)
    categorical_df.insert(1, 'Measure', 'N (%)')

    ### Drug descriptive ###
    drug_desc = get_drug_descriptives(drug_df, subcohort_name, drug_rename_map).reset_index(names = 'Value')
    drug_desc.insert(1, 'Measure', 'N (%)')
    drug_desc.insert(1, 'Variable', 'Medication')

    # concatenate continuous and categorical
    final_df = pd.concat([continuous_df, categorical_df, drug_desc])
    # Map variable name if any
    # filter variable map
    var_map = {k: v for k, v in var_map.items() if k in final_df['Variable'].unique()}
    final_df['Variable'] = final_df['Variable'].apply(lambda x: var_map[x] if x in var_map.keys() else x)
    final_df = final_df.set_index(['Variable', 'Measure', 'Value'])
    # order the variables
    final_df = final_df.loc[var_map.values()]

    return final_df

