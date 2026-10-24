"""Run reviewer-requested multimorbidity and medication sensitivity analyses.

The script uses the reconstructed primary cohort and the same complete-case
modelling variables as the primary static-model pipeline.  It writes all
derived tables and model summaries under ``results/reviewer_sensitivity``.
"""

from __future__ import annotations

from pathlib import Path

import json
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import xgboost as xgb


SEED = 42
N_BOOTSTRAP = 1_000


def bootstrap_interval(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str
) -> tuple[float, float, float]:
    """Return a metric estimate and percentile bootstrap confidence interval."""
    rng = np.random.default_rng(SEED)
    values = []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, n)
        sample_y = y_true[indices]
        if sample_y.min() == sample_y.max():
            continue
        sample_prob = y_prob[indices]
        if metric == "auroc":
            values.append(roc_auc_score(sample_y, sample_prob))
        else:
            prevalence = sample_y.mean()
            values.append(1 - brier_score_loss(sample_y, sample_prob) / prevalence)
    estimate = (
        roc_auc_score(y_true, y_prob)
        if metric == "auroc"
        else 1 - brier_score_loss(y_true, y_prob) / y_true.mean()
    )
    return estimate, *np.quantile(values, [0.025, 0.975])


def get_model_frame(config: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the primary complete-case model frame using configured variables."""
    cohort = pd.read_csv(config["cohort"])
    sumstats = pd.read_csv(config["time_series_sumstats"])
    survey = pd.read_csv(config["survey_df"])
    labels = config["subcohort_cols"]
    sumstat_features = [
        f"{feature}_{stat}"
        for feature in config["time_series_features"]
        for stat in ["mean", "std"]
    ] + [f"{feature}_rmssd" for feature in config["rmssd_features"]]
    features = config["static_features"] + config["survey_features"] + sumstat_features
    excluded = [
        f"{feature}_{stat}"
        for feature in config["excl_vars_ts"]
        for stat in ["mean", "std", "rmssd"]
    ] + config["excl_vars_static"]
    features = [feature for feature in features if feature not in excluded]
    survey = survey[["person_id", *config["survey_features"]]].astype(int)
    frame = cohort.merge(sumstats, on="person_id").merge(survey, on="person_id", how="left")
    frame = frame[["person_id", "index_date", *labels, *features]].dropna(subset=features)
    frame[labels] = frame[labels].notna().astype(int)
    return frame, features, labels


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the primary pipeline's fixed 70/30 participant split."""
    rng = np.random.default_rng(SEED)
    indices = frame.index.to_numpy(copy=True)
    rng.shuffle(indices)
    n_test = int(len(indices) * 0.3)
    frame = frame.copy()
    frame["split"] = "train"
    frame.loc[indices[:n_test], "split"] = "test"
    return frame


def fit_models(
    frame: pd.DataFrame,
    features: list[str],
    labels: list[str],
    name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Fit seeded logistic and XGBoost disease-versus-control models."""
    rows = []
    controls = frame.loc[frame[labels].sum(axis=1).eq(0)]
    for label in labels:
        cases = frame.loc[frame[label].eq(1)]
        data = pd.concat([cases, controls], ignore_index=True)
        train = data.loc[data["split"].eq("train")]
        test = data.loc[data["split"].eq("test")]
        x_train, x_test = train[features], test[features]
        y_train, y_test = train[label].to_numpy(), test[label].to_numpy()

        models = {
            "xgboost": GridSearchCV(
                xgb.XGBClassifier(
                    random_state=SEED,
                    n_jobs=4,
                    tree_method="hist",
                    eval_metric="logloss",
                ),
                {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200]},
                cv=5,
                n_jobs=2,
            ),
        }
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                logistic_model = sm.Logit(y_train, sm.add_constant(x_train)).fit(
                    disp=False, maxiter=350
                )
            if not logistic_model.mle_retvals.get("converged", False):
                raise RuntimeError("Unregularized logistic regression did not converge.")
            probability = logistic_model.predict(sm.add_constant(x_test)).to_numpy()
            logistic_estimator = "unregularized_statsmodels"
        except (np.linalg.LinAlgError, RuntimeError, ConvergenceWarning):
            logistic_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1_000_000, max_iter=5_000, random_state=SEED),
            ).fit(x_train, y_train)
            probability = logistic_model.predict_proba(x_test)[:, 1]
            logistic_estimator = "weakly_regularized_fallback"
        for metric in ["auroc", "bss"]:
            estimate, lower, upper = bootstrap_interval(y_test, probability, metric)
            rows.append(
                {
                    "analysis": name,
                    "condition": label.removeprefix("subcohort_"),
                    "model": "logistic",
                    "estimator": logistic_estimator,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_cases": int(len(cases)),
                    "n_controls": int(len(controls)),
                    "n_test": int(len(test)),
                }
            )
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            for metric in ["auroc", "bss"]:
                estimate, lower, upper = bootstrap_interval(y_test, probability, metric)
                rows.append(
                    {
                        "analysis": name,
                        "condition": label.removeprefix("subcohort_"),
                        "model": model_name,
                        "estimator": "seeded_xgboost",
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_cases": int(len(cases)),
                        "n_controls": int(len(controls)),
                        "n_test": int(len(test)),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"{name}_performance.csv", index=False)
    return result


def main() -> None:
    """Run multimorbidity and medication reviewer sensitivity analyses."""
    config = yaml.safe_load(open("conf/base_mdl_fitbit.yaml"))
    output_dir = Path("results/reviewer_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, features, labels = get_model_frame(config)
    frame = assign_split(frame)
    frame.to_csv(output_dir / "primary_complete_case_frame.csv", index=False)

    condition_count = frame[labels].sum(axis=1)
    overlap = frame[labels].T.dot(frame[labels])
    overlap.index = overlap.index.str.removeprefix("subcohort_")
    overlap.columns = overlap.columns.str.removeprefix("subcohort_")
    overlap.to_csv(output_dir / "condition_overlap_matrix.csv")
    condition_count.value_counts().sort_index().rename("participants").to_csv(
        output_dir / "target_condition_count.csv"
    )

    drug = pd.read_csv(
        "data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv",
        usecols=["person_id", "drug_exposure_start_datetime"],
    )
    drug["drug_exposure_start_datetime"] = pd.to_datetime(
        drug["drug_exposure_start_datetime"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    first_drug = drug.groupby("person_id", as_index=False)["drug_exposure_start_datetime"].min()
    frame = frame.merge(first_drug, on="person_id", how="left")
    index_date = pd.to_datetime(frame["index_date"])
    premed = frame["drug_exposure_start_datetime"].isna() | (
        index_date + pd.Timedelta(days=20) < frame["drug_exposure_start_datetime"]
    )
    control = frame[labels].sum(axis=1).eq(0)
    single_case = frame[labels].sum(axis=1).eq(1)
    clean_control = control & frame["drug_exposure_start_datetime"].isna()

    definitions = {
        "single_condition": control | single_case,
        "clean_control": (~control) | clean_control,
        "premedication_cases_clean_control": (control & clean_control) | (~control & premed),
    }
    summary = []
    results = []
    for name, mask in definitions.items():
        subset = frame.loc[mask].drop(columns="drug_exposure_start_datetime")
        summary.append(
            {
                "analysis": name,
                "participants": len(subset),
                "controls": int(subset[labels].sum(axis=1).eq(0).sum()),
                "multi_condition_cases": int((subset[labels].sum(axis=1) > 1).sum()),
            }
        )
        results.append(fit_models(subset, features, labels, name, output_dir))
    pd.DataFrame(summary).to_csv(output_dir / "sensitivity_cohort_sizes.csv", index=False)
    pd.concat(results, ignore_index=True).to_csv(output_dir / "all_sensitivity_performance.csv", index=False)
    with open(output_dir / "run_metadata.json", "w") as file:
        json.dump({"seed": SEED, "bootstrap_replicates": N_BOOTSTRAP}, file, indent=2)


if __name__ == "__main__":
    main()
