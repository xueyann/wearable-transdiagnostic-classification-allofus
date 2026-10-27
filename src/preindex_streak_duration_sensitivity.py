"""Compare corrected pre-index past-diagnosis models across streak durations.

The seven-day definition is primary. This script rebuilds the past-diagnosis
cohort for 7, 14, and 21 consecutive eligible Fitbit days, always requiring a
qualifying diagnosis on or before the selected (latest eligible) streak.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.model import delong_test
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.seven_day_past_preindex import frame_for, load_inputs
from src.streak_timing_sensitivity_models import performance_rows


OUTPUT = ROOT / "results" / "preindex_streak_duration_sensitivity"
STREAK_DAYS = (7, 14, 21)
MODEL_NAMES = ("logistic", "xgboost")


def coefficient_rows(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    days: int,
) -> pd.DataFrame:
    """Fit deterministic training-partition logistic coefficients for one cohort."""
    train = frame.loc[frame["split"].eq("train")]
    model = sm.Logit(train[label], sm.add_constant(train[features])).fit(
        disp=False, maxiter=350
    )
    confidence_interval = model.conf_int()
    result = pd.DataFrame(
        {
            "condition": label.removeprefix("subcohort_"),
            "streak_days": days,
            "feature": model.params.index,
            "coefficient": model.params.to_numpy(),
            "ci_lower": confidence_interval[0].to_numpy(),
            "ci_upper": confidence_interval[1].to_numpy(),
            "p_value": model.pvalues.to_numpy(),
        }
    )
    result["p_fdr_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def main() -> None:
    """Run duration sensitivity, matched DeLong tests, and coefficient audits."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary, config, labels, person, diagnoses, controls, daily, _ = load_inputs()
    performances: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    coefficients: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for days in STREAK_DAYS:
        streaks = find_strict_streaks(daily[["person_id", "date"]].copy(), "date", days)
        streaks = filter_streak_by_age(streaks, person.copy())
        for label in labels:
            condition = label.removeprefix("subcohort_")
            frame, features, _, _ = frame_for(
                condition,
                label,
                primary,
                config,
                person,
                diagnoses,
                controls,
                daily,
                streaks,
                primary["recency_filter"][condition],
            )
            n_cases = int(frame[label].sum())
            n_controls = len(frame) - n_cases
            samples.append(
                {
                    "streak_days": days,
                    "condition": condition,
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "n_total": len(frame),
                }
            )
            coefficients.append(coefficient_rows(frame, features, label, days))
            for model_name in MODEL_NAMES:
                ids, y_true, probability, estimator = fit_predict(
                    frame, features, label, model_name
                )
                performances.extend(
                    performance_rows(
                        "past_preindex",
                        f"{days}_consecutive_days",
                        label,
                        model_name,
                        y_true,
                        probability,
                        n_cases,
                        n_controls,
                        estimator,
                    )
                )
                predictions[(str(days), f"{condition}:{model_name}")] = pd.DataFrame(
                    {"person_id": ids, "y": y_true, "probability": probability}
                ).set_index("person_id")

    comparisons: list[dict[str, object]] = []
    for days in (14, 21):
        for label in labels:
            condition = label.removeprefix("subcohort_")
            for model_name in MODEL_NAMES:
                seven = predictions[("7", f"{condition}:{model_name}")]
                sensitivity = predictions[(str(days), f"{condition}:{model_name}")]
                joined = seven.join(
                    sensitivity[["y", "probability"]], how="inner", rsuffix="_sensitivity"
                )
                if not np.array_equal(joined["y"].to_numpy(), joined["y_sensitivity"].to_numpy()):
                    raise RuntimeError("Labels differed in a matched duration comparison.")
                test = delong_test(
                    joined["y"].to_numpy(),
                    joined["probability"].to_numpy(),
                    joined["probability_sensitivity"].to_numpy(),
                    alternative="two-sided",
                )
                comparisons.append(
                    {
                        "comparison": f"{days}_vs_7_days",
                        "condition": condition,
                        "model": model_name,
                        "n_matched_test": len(joined),
                        "auroc_7_days": roc_auc_score(joined["y"], joined["probability"]),
                        "auroc_sensitivity": roc_auc_score(
                            joined["y"], joined["probability_sensitivity"]
                        ),
                        "difference_sensitivity_minus_7_days": test["mean"],
                        "ci_lower": test["ci_lower"],
                        "ci_upper": test["ci_upper"],
                        "p_value_delong_two_sided": test["p_value"],
                    }
                )

    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["p_fdr_bh"] = multipletests(
        comparison_frame["p_value_delong_two_sided"], method="fdr_bh"
    )[1]
    pd.DataFrame(performances).to_csv(OUTPUT / "performance.csv", index=False)
    pd.DataFrame(samples).to_csv(OUTPUT / "sample_sizes.csv", index=False)
    pd.concat(coefficients, ignore_index=True).to_csv(
        OUTPUT / "logreg_coefficients.csv", index=False
    )
    comparison_frame.to_csv(OUTPUT / "matched_test_delong.csv", index=False)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "primary_definition": "7 consecutive Fitbit days, past diagnosis",
                "past_case_rule": "latest eligible streak after qualifying diagnosis",
                "seed": 42,
                "delong_test": "two-sided paired test for correlated AUROCs",
                "multiple_testing": "Benjamini-Hochberg across 28 comparisons",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
