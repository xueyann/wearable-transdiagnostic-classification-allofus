import numpy as np
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
