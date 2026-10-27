"""Run QC-matched demographic-only and detailed-race primary-model analyses."""
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
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.seven_day_past_preindex import frame_for, load_inputs
from src.streak_attrition_analysis import prepare_daily_fitbit
from src.streak_timing_sensitivity_models import performance_rows

OUTPUT = ROOT / "results" / "qc_primary_control_race_sensitivity"
MIN_HR_ZONE_MINUTES = 600
MIN_STEPS = 100
DEMOGRAPHIC_FEATURES = ["age", "sex_male", "race_nonwhite", "BMI"]
RACE_FEATURES = [
    "race_black",
    "race_asian",
    "race_multiple",
    "race_other_small_unknown",
]


def detailed_race(person: pd.DataFrame) -> pd.DataFrame:
    """Return requested race indicators with White as the omitted reference."""
    race = person[["person_id", "race"]].copy()
    race["race_group"] = race["race"].replace(
        {
            "Black or African American": "black",
            "Asian": "asian",
            "More than one population": "multiple",
            "White": "white",
        }
    )
    race.loc[~race["race_group"].isin(["white", "black", "asian", "multiple"]), "race_group"] = "other_small_unknown"
    for group in ("black", "asian", "multiple", "other_small_unknown"):
        race[f"race_{group}"] = race["race_group"].eq(group).astype(int)
    return race[["person_id", *RACE_FEATURES]]


def main() -> None:
    """Fit QC primary, demographic-only, and detailed-race models."""
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
    race = detailed_race(person)
    performance: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        frame, primary_features, _, _ = frame_for(
            condition, label, primary, config, person, diagnoses, controls, daily,
            streaks, primary["recency_filter"][condition],
        )
        frame = frame.merge(race, on="person_id", how="left", validate="one_to_one")
        detailed_features = [
            feature for feature in primary_features if feature != "race_nonwhite"
        ] + RACE_FEATURES
        models = {
            "primary": primary_features,
            "demographic_only": DEMOGRAPHIC_FEATURES,
            "detailed_race": detailed_features,
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
        for analysis in ("demographic_only", "detailed_race"):
            for model_name in ("logistic", "xgboost"):
                reference = predictions[("primary", model_name)]
                sensitivity = predictions[(analysis, model_name)]
                if not np.array_equal(reference.y.to_numpy(), sensitivity.y.to_numpy()):
                    raise RuntimeError("Primary and comparison labels differ.")
                test = delong_test(
                    reference.y.to_numpy(), reference.probability.to_numpy(),
                    sensitivity.probability.to_numpy(), alternative="two-sided",
                )
                comparisons.append({
                    "comparison": analysis,
                    "condition": condition,
                    "model": model_name,
                    "n_test": len(reference),
                    "auc_primary": roc_auc_score(reference.y, reference.probability),
                    "auc_comparison": roc_auc_score(reference.y, sensitivity.probability),
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
        "primary_features": "existing primary feature set",
        "demographic_only_features": DEMOGRAPHIC_FEATURES,
        "detailed_race_reference": "White",
        "detailed_race_indicators": RACE_FEATURES,
        "seed": 42,
        "test": "two-sided paired DeLong test for correlated AUROCs",
    }, indent=2))


if __name__ == "__main__":
    main()
