"""Formally compare primary and sensitivity model AUROCs on matched test sets.

For every outcome and sensitivity definition, the primary-trained model and the
sensitivity-trained model are evaluated on the same held-out sensitivity cohort.
Their correlated AUROCs are compared using DeLong's two-sided test.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.model import delong_test
from reviewer_sensitivity_analyses import SEED, assign_split, get_model_frame


ANALYSES = (
    "single_condition",
    "clean_control",
    "premedication_cases_clean_control",
)


def make_sensitivity_masks(
    frame: pd.DataFrame, labels: list[str]
) -> dict[str, pd.Series]:
    """Return the three pre-specified sensitivity-analysis inclusion masks."""
    drug = pd.read_csv(
        "data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv",
        usecols=["person_id", "drug_exposure_start_datetime"],
    )
    drug["drug_exposure_start_datetime"] = pd.to_datetime(
        drug["drug_exposure_start_datetime"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    first_drug = drug.groupby("person_id", as_index=False)[
        "drug_exposure_start_datetime"
    ].min()
    frame_with_drug = frame.merge(first_drug, on="person_id", how="left")
    index_date = pd.to_datetime(frame_with_drug["index_date"])
    premedication = frame_with_drug["drug_exposure_start_datetime"].isna() | (
        index_date + pd.Timedelta(days=20)
        < frame_with_drug["drug_exposure_start_datetime"]
    )
    control = frame_with_drug[labels].sum(axis=1).eq(0)
    single_case = frame_with_drug[labels].sum(axis=1).eq(1)
    clean_control = control & frame_with_drug["drug_exposure_start_datetime"].isna()
    return {
        "single_condition": control | single_case,
        "clean_control": (~control) | clean_control,
        "premedication_cases_clean_control": clean_control | (~control & premedication),
    }


def condition_data(
    frame: pd.DataFrame, labels: list[str], label: str
) -> pd.DataFrame:
    """Construct a disease-versus-control data set for one target condition."""
    controls = frame.loc[frame[labels].sum(axis=1).eq(0)]
    cases = frame.loc[frame[label].eq(1)]
    return pd.concat([cases, controls], ignore_index=True)


def fit_predict(
    data: pd.DataFrame,
    features: list[str],
    label: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Fit one seeded model and return held-out IDs, labels, and probabilities."""
    train = data.loc[data["split"].eq("train")]
    test = data.loc[data["split"].eq("test")]
    x_train, x_test = train[features], test[features]
    y_train, y_test = train[label].to_numpy(), test[label].to_numpy()

    if model_name == "logistic":
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model = sm.Logit(y_train, sm.add_constant(x_train)).fit(
                    disp=False, maxiter=350
                )
            if not model.mle_retvals.get("converged", False):
                raise RuntimeError("Unregularized logistic regression did not converge.")
            probability = model.predict(sm.add_constant(x_test)).to_numpy()
            estimator = "unregularized_statsmodels"
        except (np.linalg.LinAlgError, RuntimeError, ConvergenceWarning):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1_000_000, max_iter=5_000, random_state=SEED),
            ).fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            estimator = "weakly_regularized_fallback"
    elif model_name == "xgboost":
        model = GridSearchCV(
            xgb.XGBClassifier(
                random_state=SEED,
                n_jobs=4,
                tree_method="hist",
                eval_metric="logloss",
            ),
            {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200]},
            cv=5,
            n_jobs=2,
        ).fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        estimator = "seeded_xgboost"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return test["person_id"].to_numpy(), y_test, probability, estimator


def main() -> None:
    """Run matched-test-set DeLong comparisons for the reviewer sensitivities."""
    config = yaml.safe_load(Path("conf/base_mdl_fitbit.yaml").read_text())
    output_dir = Path("results/reviewer_sensitivity")
    frame, features, labels = get_model_frame(config)
    frame = assign_split(frame).reset_index(drop=True)
    masks = make_sensitivity_masks(frame, labels)
    model_names = ("logistic", "xgboost")
    rows: list[dict[str, object]] = []

    for label in labels:
        primary_data = condition_data(frame, labels, label)
        primary_predictions = {}
        for model_name in model_names:
            ids, y_true, probability, estimator = fit_predict(
                primary_data, features, label, model_name
            )
            primary_predictions[model_name] = pd.DataFrame(
                {"person_id": ids, "y_true": y_true, "primary_probability": probability}
            ).set_index("person_id")
            primary_predictions[model_name]["primary_estimator"] = estimator

        for analysis in ANALYSES:
            sensitivity_frame = frame.loc[masks[analysis]]
            sensitivity_data = condition_data(sensitivity_frame, labels, label)
            for model_name in model_names:
                ids, y_true, probability, estimator = fit_predict(
                    sensitivity_data, features, label, model_name
                )
                sensitivity_predictions = pd.DataFrame(
                    {
                        "person_id": ids,
                        "y_true": y_true,
                        "sensitivity_probability": probability,
                    }
                ).set_index("person_id")
                joined = sensitivity_predictions.join(
                    primary_predictions[model_name].drop(columns="y_true"), how="inner"
                )
                if len(joined) != len(sensitivity_predictions):
                    raise RuntimeError(
                        f"Primary predictions missing for {analysis}, {label}, {model_name}."
                    )
                # Verify that the two fitted models are evaluated against identical labels.
                primary_y = primary_predictions[model_name].loc[joined.index, "y_true"]
                if not np.array_equal(joined["y_true"].to_numpy(), primary_y.to_numpy()):
                    raise RuntimeError("Label mismatch between matched test predictions.")
                result = delong_test(
                    joined["y_true"].to_numpy(),
                    joined["primary_probability"].to_numpy(),
                    joined["sensitivity_probability"].to_numpy(),
                    alternative="two-sided",
                )
                rows.append(
                    {
                        "analysis": analysis,
                        "condition": label.removeprefix("subcohort_"),
                        "model": model_name,
                        "n_test": len(joined),
                        "n_test_cases": int(joined["y_true"].sum()),
                        "n_test_controls": int((1 - joined["y_true"]).sum()),
                        "primary_auc_on_matched_test": roc_auc_score(
                            joined["y_true"], joined["primary_probability"]
                        ),
                        "sensitivity_auc_on_matched_test": roc_auc_score(
                            joined["y_true"], joined["sensitivity_probability"]
                        ),
                        "auroc_difference_sensitivity_minus_primary": result["mean"],
                        "ci_lower": result["ci_lower"],
                        "ci_upper": result["ci_upper"],
                        "z_statistic": result["z_stat"],
                        "p_value_delong_two_sided": result["p_value"],
                        "primary_estimator": primary_predictions[model_name][
                            "primary_estimator"
                        ].iloc[0],
                        "sensitivity_estimator": estimator,
                    }
                )

    results = pd.DataFrame(rows)
    results["p_value_fdr_bh"] = multipletests(
        results["p_value_delong_two_sided"], method="fdr_bh"
    )[1]
    results.to_csv(output_dir / "matched_test_delong_comparisons.csv", index=False)
    (output_dir / "matched_test_delong_metadata.json").write_text(
        json.dumps(
            {
                "test": "Two-sided paired DeLong test for correlated AUROCs",
                "comparison": (
                    "Primary-trained and sensitivity-trained models evaluated on the "
                    "same held-out sensitivity-cohort participants"
                ),
                "seed": SEED,
                "multiple_testing_adjustment": "Benjamini-Hochberg across 42 tests",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
