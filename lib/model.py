import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from lib.helpers import format_p


def compute_midrank(x):
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n)
    
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    
    result = np.empty(n)
    result[sorted_idx] = midranks
    return result


def fast_delong(predictions, labels):
    order = np.argsort(-labels)
    labels = labels[order]
    predictions = predictions[:, order]
    
    m = int(np.sum(labels))
    n = len(labels) - m
    
    pos_preds = predictions[:, :m]
    neg_preds = predictions[:, m:]
    
    k = predictions.shape[0]
    
    tx = np.array([compute_midrank(x) for x in pos_preds])
    ty = np.array([compute_midrank(x) for x in neg_preds])
    tz = np.array([compute_midrank(x) for x in predictions])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    
    return aucs, s


def delong_test(y_true, pred1, pred2, alternative="two-sided", alpha=0.05):
    y_true, pred1, pred2 = np.array(y_true), np.array(pred1), np.array(pred2)
    preds = np.vstack([pred1, pred2])
    aucs, cov = fast_delong(preds, y_true)
    
    # Difference and variance
    diff = aucs[1] - aucs[0]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(var)
    
    # Z-statistic
    z = diff / se
    
    # p-value
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("Invalid alternative")
    
    # Two-sided confidence interval
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = diff - z_crit * se
    ci_upper = diff + z_crit * se
    
    return {
        "mean": np.mean(diff),
        "z_stat": z,
        "p_value": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def bootstrap(y_true, y_proba, func, alpha = 0.05, n_bootstrap=1000, seed=42):

    # change to np array to avoid indexing problem with pandas (index not starting from 1)
    y_true, y_proba = np.array(y_true), np.array(y_proba)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = []

    for _ in range(n_bootstrap):
        # sample indices with replacement
        idx = rng.choice(n, size=n, replace=True)

        # guard against degenerate samples (all one class)
        if len(np.unique(y_true[idx])) < 2:
            continue

        score = func(y_true[idx], y_proba[idx])
        scores.append(score)
    
    scores = np.array(scores) 

    ci_lower, ci_upper = np.percentile(scores, [(alpha*100)/2, 100-(alpha*100)/2])

    return {
        "mean": scores.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper)
    }


def bootstrap_fmt(y_true, y_proba, func, n_bootstrap=1000, seed=42):
    res = bootstrap(y_true, y_proba, func, n_bootstrap=n_bootstrap, seed=seed)
    return f"{res['mean']:.3f} ({res['ci_lower']:.3f} - {res['ci_upper']:.3f})"

def paired_bootstrap_fmt(y_true, y_proba1, y_proba2, func, n_bootstrap=1000, seed=42):
    res = paired_bootstrap(y_true, y_proba1, y_proba2, func, n_bootstrap=n_bootstrap, seed=seed)
    return f"{res['mean']:.3f} ({res['ci_lower']:.3f} - {res['ci_upper']:.3f})"


def paired_bootstrap(y_true, y_proba1, y_proba2, func, 
                     alpha = 0.05, n_bootstrap=1000, seed=42, 
                     alternative = 'two-sided'):
    """
    y_true: array of true label (binary)
    y_proba1: predicted probabilities with the first model (null model)
    y_proba2: predicted probabilities with the second model 
    func: Callable, takes two arrays y_true and y_proba
    """
    # change to np array to avoid indexing problem with pandas (index not starting from 1)
    y_true, y_proba1, y_proba2 = np.array(y_true), np.array(y_proba1), np.array(y_proba2)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    deltas = []

    for _ in range(n_bootstrap):
        # sample indices with replacement
        idx = rng.choice(n, size=n, replace=True)

        # guard against degenerate samples (all one class)
        if len(np.unique(y_true[idx])) < 2:
            continue

        score1 = func(y_true[idx], y_proba1[idx])
        score2 = func(y_true[idx], y_proba2[idx])

        deltas.append(score2 - score1)

    deltas = np.array(deltas) 

    # two sided CI still
    ci_lower, ci_upper = np.percentile(deltas, [(alpha*100)/2, 100-(alpha*100)/2])

    # Two-sided p-value
    if alternative == 'two-sided':
        p = 2 * min(np.mean(deltas >= 0),
                    np.mean(deltas <= 0))
    elif alternative == 'greater':
        p = np.mean(deltas >= 0)
    elif alternative == 'less':
        p = np.mean(deltas <= 0)

    # # one sided significance test: whether delta > 0, i.e mdl2 is better than mdl1
    # critical_val = np.percentile(deltas, alpha*100)

    return {
        "mean": deltas.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p
    }

def paired_bootstrap_fmt(y_true, y_proba1, y_proba2, func, n_bootstrap=1000, seed=42, alpha = 0.05):
    res = paired_bootstrap(y_true, y_proba1, y_proba2, func, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)

    fmt_mean = f"{res['mean']:.3f} ({res['ci_lower']:.3f} - {res['ci_upper']:.3f})"
    fmt_p = format_p(res['p_value'], alpha = alpha)
    return fmt_mean, fmt_p

def delong_fmt(y_true, y_proba1, y_proba2, alpha = 0.05):
    """
    No function needed because it is specific to AUC ROC"""
    res = delong_test(y_true, y_proba1, y_proba2, alternative="greater", alpha=alpha)
    fmt_mean = f"{res['mean']:.3f} ({res['ci_lower']:.3f} - {res['ci_upper']:.3f})"
    fmt_p = format_p(res['p_value'], alpha = alpha)
    return fmt_mean, fmt_p