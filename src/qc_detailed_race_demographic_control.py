"""Compare full detailed-race models with a detailed-race demographic control."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.model import delong_test
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age
from src.detailed_race_features import RACE_FEATURES, add_detailed_race_features
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.seven_day_past_preindex import frame_for, load_inputs
from src.streak_attrition_analysis import prepare_daily_fitbit
from src.streak_timing_sensitivity_models import performance_rows


OUTPUT = ROOT / "results" / "qc_detailed_race_demographic_control"
MIN_HR_ZONE_MINUTES = 600
MIN_STEPS = 100
DEMOGRAPHIC_FEATURES = ["age", "sex_male", "BMI", *RACE_FEATURES]


def main() -> None:
    """Fit full and demographic-only detailed-race models on the QC cohort."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary, config, labels, person, diagnoses, controls, _, _ = load_inputs()
    daily = prepare_daily_fitbit(
        primary["fitbit_summary_data"],
        person.person_id,
        min_hr_zone_minutes=MIN_HR_ZONE_MINUTES,
        min_steps=MIN_STEPS,
    )
    streaks = filter_streak_by_age(
        find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 7),
        person.copy(),
    )
    performance: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        frame, base_features, _, _ = frame_for(
            condition, label, primary, config, person, diagnoses, controls, daily,
            streaks, primary["recency_filter"][condition],
        )
        frame, detailed_features = add_detailed_race_features(frame, base_features, person)
        models = {
            "primary_detailed_race": detailed_features,
            "demographic_only_detailed_race": DEMOGRAPHIC_FEATURES,
        }
        predictions: dict[tuple[str, str], pd.DataFrame] = {}
        n_cases = int(frame[label].sum())
        n_controls = len(frame) - n_cases
        for analysis, features in models.items():
            samples.append({
                "analysis": analysis,
                "condition": condition,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "n_total": len(frame),
                "features": "|".join(features),
            })
            for model_name in ("logistic", "xgboost"):
                ids, y_true, probability, estimator = fit_predict(
                    frame, features, label, model_name
                )
                performance.extend(performance_rows(
                    analysis, "7_consecutive_days_qc", label, model_name, y_true,
                    probability, n_cases, n_controls, estimator,
                ))
                predictions[(analysis, model_name)] = pd.DataFrame({
                    "person_id": ids, "y": y_true, "probability": probability,
                }).set_index("person_id")
        for model_name in ("logistic", "xgboost"):
            reference = predictions[("primary_detailed_race", model_name)]
            comparator = predictions[("demographic_only_detailed_race", model_name)]
            if not np.array_equal(reference.y.to_numpy(), comparator.y.to_numpy()):
                raise RuntimeError("Primary and comparison labels differ.")
            test = delong_test(
                reference.y.to_numpy(), reference.probability.to_numpy(),
                comparator.probability.to_numpy(), alternative="two-sided",
            )
            comparisons.append({
                "comparison": "demographic_only_detailed_race",
                "condition": condition,
                "model": model_name,
                "n_test": len(reference),
                "auc_primary": roc_auc_score(reference.y, reference.probability),
                "auc_comparison": roc_auc_score(reference.y, comparator.probability),
                "difference_comparison_minus_primary": test["mean"],
                "ci_lower": test["ci_lower"],
                "ci_upper": test["ci_upper"],
                "p_value_delong_two_sided": test["p_value"],
            })

    pd.DataFrame(performance).to_csv(OUTPUT / "performance.csv", index=False)
    pd.DataFrame(samples).to_csv(OUTPUT / "sample_sizes.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "matched_test_delong.csv", index=False)
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "qc_definition": "At least 600 HR-zone minutes and 100 steps daily before 7-day streak selection; HR-zone minutes are a provisional summary-level proxy.",
        "primary_features": "existing primary feature set with detailed race indicators",
        "demographic_only_features": DEMOGRAPHIC_FEATURES,
        "detailed_race_reference": "White",
        "detailed_race_indicators": RACE_FEATURES,
        "seed": 42,
        "test": "two-sided paired DeLong test for correlated AUROCs",
    }, indent=2))


if __name__ == "__main__":
    main()
