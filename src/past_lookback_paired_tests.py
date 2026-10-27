"""Run raw paired DeLong AUROC tests for seven-day lookback sensitivities.

The current disease-specific lookback rule is the primary definition. Each
alternative policy is refit, then its model and the primary model are evaluated
on participants assigned to the held-out test partition in both cohorts.
"""

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
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.seven_day_past_preindex import frame_for, load_inputs


OUTPUT = ROOT / "results" / "past_lookback_sensitivity_7day_preindex"
MODEL_NAMES = ("logistic", "xgboost")


def prediction_frame(
    frame: pd.DataFrame, features: list[str], label: str, model_name: str
) -> tuple[pd.DataFrame, str]:
    """Fit one seeded model and return its held-out predictions by participant."""
    person_ids, y_true, probability, estimator = fit_predict(
        frame, features, label, model_name
    )
    return (
        pd.DataFrame(
            {"person_id": person_ids, "y_true": y_true, "probability": probability}
        ).set_index("person_id"),
        estimator,
    )


def main() -> None:
    """Compare no/halved-lookback models with the current-lookback primary model."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary, config, labels, person, diagnoses, controls, daily, streaks = load_inputs()
    policies: dict[str, int | None | dict[str, float]] = {
        "current_rules": primary["recency_filter"],
        "no_lookback": None,
        "halved_rules": {key: value / 2 for key, value in primary["recency_filter"].items()},
    }
    rows: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        frames: dict[str, tuple[pd.DataFrame, list[str]]] = {}
        for policy, thresholds in policies.items():
            lookback = None if thresholds is None else thresholds[condition]
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
                lookback,
            )
            frames[policy] = (frame, features)

        for model_name in MODEL_NAMES:
            primary_predictions, primary_estimator = prediction_frame(
                *frames["current_rules"], label, model_name
            )
            for policy in ("no_lookback", "halved_rules"):
                sensitivity_predictions, sensitivity_estimator = prediction_frame(
                    *frames[policy], label, model_name
                )
                joined = primary_predictions.join(
                    sensitivity_predictions, how="inner", rsuffix="_sensitivity"
                )
                if not np.array_equal(
                    joined["y_true"].to_numpy(),
                    joined["y_true_sensitivity"].to_numpy(),
                ):
                    raise RuntimeError("Labels differed in a matched lookback comparison.")
                test = delong_test(
                    joined["y_true"].to_numpy(),
                    joined["probability"].to_numpy(),
                    joined["probability_sensitivity"].to_numpy(),
                    alternative="two-sided",
                )
                rows.append(
                    {
                        "policy": policy,
                        "condition": condition,
                        "model": model_name,
                        "n_matched_test": len(joined),
                        "auc_primary_current_rules": roc_auc_score(
                            joined["y_true"], joined["probability"]
                        ),
                        "auc_sensitivity": roc_auc_score(
                            joined["y_true"], joined["probability_sensitivity"]
                        ),
                        "difference_sensitivity_minus_primary": test["mean"],
                        "ci_lower": test["ci_lower"],
                        "ci_upper": test["ci_upper"],
                        "p_value_delong_two_sided": test["p_value"],
                        "primary_estimator": primary_estimator,
                        "sensitivity_estimator": sensitivity_estimator,
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT / "matched_test_delong.csv", index=False)
    (OUTPUT / "matched_test_delong_metadata.json").write_text(
        json.dumps(
            {
                "test": "two-sided paired DeLong test for correlated AUROCs",
                "primary_policy": "current_rules",
                "seed": 42,
                "multiple_testing_adjustment": "none; raw P-values reported",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
