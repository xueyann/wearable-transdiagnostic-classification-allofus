"""Rerun the medication-unexposed sensitivity with a 30-day Fitbit buffer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.model import delong_test
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.seven_day_past_preindex import frame_for, load_inputs
from src.seven_day_past_preindex_sensitivities import active_medication_ids
from src.streak_timing_sensitivity_models import markdown_table, performance_rows


OUTPUT = ROOT / "results" / "medication_unexposed_30day_window"
ANALYSIS = "medication_unexposed_30day_window"


def main() -> None:
    """Fit the primary and buffered medication-unexposed models."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary, cfg, labels, person, diag, ctrl, daily, streaks = load_inputs()
    performance: list[dict[str, object]] = []
    sample_sizes: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        base, features, _, selected = frame_for(
            condition,
            label,
            primary,
            cfg,
            person,
            diag,
            ctrl,
            daily,
            streaks,
            primary["recency_filter"][condition],
        )
        blocked = active_medication_ids(selected)
        sensitivity = base.loc[~base.person_id.isin(blocked)].copy()
        frames = {"primary": base, ANALYSIS: sensitivity}
        primary_predictions: dict[str, pd.DataFrame] = {}

        for analysis, frame in frames.items():
            n_cases = int(frame[label].sum())
            n_controls = len(frame) - n_cases
            sample_sizes.append(
                {
                    "analysis": analysis,
                    "condition": condition,
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "n_total": len(frame),
                    "n_excluded_medication": len(base) - len(sensitivity),
                }
            )
            for model in ("logistic", "xgboost"):
                ids, y_true, probability, estimator = fit_predict(
                    frame, features, label, model
                )
                performance.extend(
                    performance_rows(
                        analysis,
                        "7_consecutive_preindex",
                        label,
                        model,
                        y_true,
                        probability,
                        n_cases,
                        n_controls,
                        estimator,
                    )
                )
                predictions = pd.DataFrame(
                    {"person_id": ids, "y_true": y_true, "probability": probability}
                ).set_index("person_id")
                if analysis == "primary":
                    primary_predictions[model] = predictions
                    continue

                joined = predictions.join(
                    primary_predictions[model][["probability"]],
                    how="inner",
                    rsuffix="_primary",
                )
                primary_y = primary_predictions[model].loc[joined.index, "y_true"]
                if not joined.empty and not (joined["y_true"].to_numpy() == primary_y.to_numpy()).all():
                    raise RuntimeError("Label mismatch between matched test predictions.")
                if len(joined) and joined["y_true"].nunique() == 2:
                    test = delong_test(
                        joined["y_true"].to_numpy(),
                        joined["probability_primary"].to_numpy(),
                        joined["probability"].to_numpy(),
                        alternative="two-sided",
                    )
                    comparisons.append(
                        {
                            "analysis": ANALYSIS,
                            "condition": condition,
                            "model": model,
                            "n_matched_test": len(joined),
                            "n_matched_test_cases": int(joined["y_true"].sum()),
                            "n_matched_test_controls": int(
                                (1 - joined["y_true"]).sum()
                            ),
                            "auc_primary": roc_auc_score(
                                joined["y_true"], joined["probability_primary"]
                            ),
                            "auc_sensitivity": roc_auc_score(
                                joined["y_true"], joined["probability"]
                            ),
                            "difference_sensitivity_minus_primary": test["mean"],
                            "ci_lower": test["ci_lower"],
                            "ci_upper": test["ci_upper"],
                            "p_value_delong_two_sided": test["p_value"],
                        }
                    )

    performance_df = pd.DataFrame(performance)
    samples_df = pd.DataFrame(sample_sizes)
    comparisons_df = pd.DataFrame(comparisons)
    performance_df.to_csv(OUTPUT / "performance.csv", index=False)
    samples_df.to_csv(OUTPUT / "sample_sizes.csv", index=False)
    comparisons_df.to_csv(OUTPUT / "matched_test_delong.csv", index=False)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "analysis": ANALYSIS,
                "medication_definition": (
                    "Exclude a participant if a relevant drug record overlaps the "
                    "Fitbit streak expanded by 30 calendar days on either side. "
                    "Use the recorded exposure end date when available; otherwise "
                    "treat the record as a one-day exposure ending on its start date."
                ),
                "paired_test": "Two-sided paired DeLong test for correlated AUROCs",
                "multiple_testing_correction": "None",
            },
            indent=2,
        )
        + "\n"
    )

    auroc = performance_df.loc[performance_df.metric.eq("auroc")].copy()
    display = auroc.pivot(
        index=["condition", "model"], columns="analysis", values="estimate"
    ).reset_index()
    report = [
        "# Medication-unexposed sensitivity: ±30-day Fitbit window",
        "",
        "A participant is excluded from the medication-unexposed cohort when a "
        "relevant drug record overlaps the Fitbit streak expanded by 30 calendar "
        "days on each side. Records with no end date are treated as one-day "
        "exposures (end date = start date).",
        "",
        "## Sample sizes",
        "",
        markdown_table(samples_df),
        "",
        "## AUROC",
        "",
        markdown_table(display),
        "",
        "## Paired comparisons",
        "",
        "The primary and sensitivity-trained models are compared on their shared "
        "held-out participants with a two-sided paired DeLong test; p-values are raw "
        "and not adjusted for multiple testing.",
        "",
        markdown_table(comparisons_df),
    ]
    (OUTPUT / "medication_unexposed_30day_window_results.md").write_text(
        "\n".join(report) + "\n"
    )


if __name__ == "__main__":
    main()
