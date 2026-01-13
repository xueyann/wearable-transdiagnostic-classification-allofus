import numpy as np
import pandas as pd


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

    ci_lower, ci_upper = np.percentile(scores, [(alpha*100)/2, 1-(alpha*100)/2])

    return {
        "mean": scores.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper)
    }

def paired_bootstrap_fmt(y_true, y_proba1, y_proba2, func, n_bootstrap=1000, seed=42):
    res = paired_bootstrap(y_true, y_proba1, y_proba2, func, n_bootstrap=n_bootstrap, seed=seed)
    return f"{res['mean']:.3f} ({res['ci_lower']:.3f} - {res['ci_upper']:.3f})"


def paired_bootstrap(y_true, y_proba1, y_proba2, func, n_bootstrap=1000, seed=42):
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

    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])

    return {
        "mean": deltas.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper)
    }
