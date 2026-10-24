"""Run past- versus future-diagnosis timing sensitivity analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.model import delong_test
from reviewer_sensitivity_analyses import (
    N_BOOTSTRAP,
    SEED,
    assign_split,
    bootstrap_interval,
    get_model_frame,
)
from reviewer_sensitivity_performance_tests import condition_data, fit_predict


COHORT_PATH = Path("data/02_intermediate/study_cohorts_20260225/study_cohort_20260225.csv")
STREAK_PATH = Path("data/02_intermediate/study_cohorts_20260225/streaks_20260225.csv")
OUTPUT_DIR = Path("results/diagnosis_timing_sensitivity")
TIMING_ANALYSES = ("past_diagnosis", "future_diagnosis")


def first_streak_end() -> pd.DataFrame:
    """Return the end date of each participant's first analysed Fitbit streak."""
    streaks = pd.read_csv(STREAK_PATH, usecols=["person_id", "streak_start", "streak_end"])
    streaks[["streak_start", "streak_end"]] = streaks[["streak_start", "streak_end"]].apply(pd.to_datetime)
    return (
        streaks.sort_values(["person_id", "streak_start", "streak_end"])
        .groupby("person_id", as_index=False)
        .first()[["person_id", "streak_start", "streak_end"]]
    )


def add_diagnosis_dates(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Attach each stored earliest diagnosis date and first-streak end date."""
    cohort = pd.read_csv(COHORT_PATH, usecols=["person_id", *labels])
    dates = cohort.rename(columns={label: f"{label}_diagnosis_date" for label in labels})
    for label in labels:
        dates[f"{label}_diagnosis_date"] = pd.to_datetime(
            dates[f"{label}_diagnosis_date"], errors="coerce"
        )
    result = frame.merge(dates, on="person_id", how="left", validate="one_to_one")
    result = result.merge(first_streak_end(), on="person_id", how="left", validate="one_to_one")
    result["index_date"] = pd.to_datetime(result["index_date"])
    if not result["streak_start"].eq(result["index_date"]).all():
        raise RuntimeError("First streak start does not match the primary model index date.")
    return result


def timing_case_mask(frame: pd.DataFrame, label: str, timing: str) -> pd.Series:
    """Identify cases with an earliest diagnosis before or after the analysed streak."""
    diagnosis_date = frame[f"{label}_diagnosis_date"]
    if timing == "past_diagnosis":
        return frame[label].eq(1) & diagnosis_date.lt(frame["index_date"])
    if timing == "future_diagnosis":
        return frame[label].eq(1) & diagnosis_date.gt(frame["streak_end"])
    raise ValueError(f"Unknown timing analysis: {timing}")


def timing_condition_data(
    frame: pd.DataFrame, labels: list[str], label: str, timing: str
) -> pd.DataFrame:
    """Construct timing-restricted cases versus the original shared controls."""
    controls = frame.loc[frame[labels].sum(axis=1).eq(0)]
    cases = frame.loc[timing_case_mask(frame, label, timing)]
    return pd.concat([cases, controls], ignore_index=True)


def scaled_brier_skill_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute Brier skill score using the outcome prevalence as reference risk."""
    prevalence = y_true.mean()
    return 1 - brier_score_loss(y_true, probabilities) / prevalence


def performance_rows(
    analysis: str,
    condition: str,
    model: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_cases: int,
    n_controls: int,
    estimator: str,
) -> list[dict[str, object]]:
    """Summarize AUROC and Brier skill score with bootstrap confidence intervals."""
    rows = []
    for metric in ("auroc", "bss"):
        estimate, lower, upper = bootstrap_interval(y_true, probabilities, metric)
        rows.append(
            {
                "analysis": analysis,
                "condition": condition.removeprefix("subcohort_"),
                "model": model,
                "estimator": estimator,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "n_test": len(y_true),
            }
        )
    return rows


def describe_cases(
    frame: pd.DataFrame, labels: list[str], label: str, timing: str
) -> dict[str, object]:
    """Describe timing-restricted cases in the complete-case model frame."""
    cases = frame.loc[timing_case_mask(frame, label, timing)].copy()
    timing_days = (cases[f"{label}_diagnosis_date"] - cases["index_date"]).dt.days
    distance_days = -timing_days if timing == "past_diagnosis" else timing_days
    return {
        "analysis": timing,
        "condition": label.removeprefix("subcohort_"),
        "n_cases": len(cases),
        "age_mean": cases["age"].mean(),
        "age_sd": cases["age"].std(),
        "male_pct": 100 * cases["sex_male"].mean(),
        "nonwhite_pct": 100 * cases["race_nonwhite"].mean(),
        "bmi_mean": cases["BMI"].mean(),
        "bmi_sd": cases["BMI"].std(),
        "target_conditions_median": cases[labels].sum(axis=1).median(),
        "time_from_streak_days_median": distance_days.median(),
        "time_from_streak_days_q25": distance_days.quantile(0.25),
        "time_from_streak_days_q75": distance_days.quantile(0.75),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a dependency-free Markdown table."""
    rows = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def format_p(value: float) -> str:
    """Format p-values for the Markdown report."""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    """Run timing-restricted models, descriptions, and matched DeLong comparisons."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import yaml

    config = yaml.safe_load(Path("conf/base_mdl_fitbit.yaml").read_text())
    frame, features, labels = get_model_frame(config)
    frame = add_diagnosis_dates(assign_split(frame).reset_index(drop=True), labels)
    performance = []
    comparisons = []
    descriptions = []

    for label in labels:
        primary_data = condition_data(frame, labels, label)
        primary_predictions: dict[str, pd.DataFrame] = {}
        for model in ("logistic", "xgboost"):
            ids, y_true, probability, estimator = fit_predict(primary_data, features, label, model)
            primary_predictions[model] = pd.DataFrame(
                {
                    "person_id": ids,
                    "y_true": y_true,
                    "primary_probability": probability,
                    "primary_estimator": estimator,
                }
            ).set_index("person_id")

        for timing in TIMING_ANALYSES:
            descriptions.append(describe_cases(frame, labels, label, timing))
            data = timing_condition_data(frame, labels, label, timing)
            n_cases = int(data[label].sum())
            n_controls = int(len(data) - n_cases)
            for model in ("logistic", "xgboost"):
                ids, y_true, probability, estimator = fit_predict(data, features, label, model)
                performance.extend(
                    performance_rows(
                        timing, label, model, y_true, probability,
                        n_cases, n_controls, estimator,
                    )
                )
                timing_predictions = pd.DataFrame(
                    {
                        "person_id": ids,
                        "y_true": y_true,
                        "timing_probability": probability,
                    }
                ).set_index("person_id")
                joined = timing_predictions.join(
                    primary_predictions[model].drop(columns="y_true"), how="inner"
                )
                primary_y = primary_predictions[model].loc[joined.index, "y_true"]
                if len(joined) != len(timing_predictions) or not np.array_equal(
                    joined["y_true"].to_numpy(), primary_y.to_numpy()
                ):
                    raise RuntimeError("Could not align primary and timing test predictions.")
                test = delong_test(
                    joined["y_true"].to_numpy(),
                    joined["primary_probability"].to_numpy(),
                    joined["timing_probability"].to_numpy(),
                    alternative="two-sided",
                )
                comparisons.append(
                    {
                        "analysis": timing,
                        "condition": label.removeprefix("subcohort_"),
                        "model": model,
                        "n_test_cases": int(joined["y_true"].sum()),
                        "n_test_controls": int((1 - joined["y_true"]).sum()),
                        "primary_auc_on_matched_test": roc_auc_score(
                            joined["y_true"], joined["primary_probability"]
                        ),
                        "timing_auc_on_matched_test": roc_auc_score(
                            joined["y_true"], joined["timing_probability"]
                        ),
                        "auroc_difference_timing_minus_primary": test["mean"],
                        "ci_lower": test["ci_lower"],
                        "ci_upper": test["ci_upper"],
                        "z_statistic": test["z_stat"],
                        "p_value_delong_two_sided": test["p_value"],
                        "primary_estimator": primary_predictions[model]["primary_estimator"].iloc[0],
                        "timing_estimator": estimator,
                    }
                )

    performance_df = pd.DataFrame(performance)
    comparison_df = pd.DataFrame(comparisons)
    comparison_df["p_value_fdr_bh"] = multipletests(
        comparison_df["p_value_delong_two_sided"], method="fdr_bh"
    )[1]
    description_df = pd.DataFrame(descriptions)
    performance_df.to_csv(OUTPUT_DIR / "diagnosis_timing_performance.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "diagnosis_timing_delong_comparisons.csv", index=False)
    description_df.to_csv(OUTPUT_DIR / "diagnosis_timing_case_descriptives.csv", index=False)
    (OUTPUT_DIR / "diagnosis_timing_metadata.json").write_text(
        json.dumps(
            {
                "timing_definitions": {
                    "past_diagnosis": "Earliest diagnosis before the first day of the selected Fitbit streak.",
                    "future_diagnosis": "Earliest diagnosis after the last day of the selected Fitbit streak.",
                },
                "controls": "Original primary controls in both timing analyses.",
                "comparison_test": "Two-sided paired DeLong test for correlated AUROCs on each matched held-out timing cohort.",
                "seed": SEED,
                "bootstrap_replicates": N_BOOTSTRAP,
                "multiple_testing_adjustment": "Benjamini-Hochberg across 28 timing comparisons.",
            },
            indent=2,
        )
    )

    summary_rows = []
    for timing in TIMING_ANALYSES:
        for condition in [label.removeprefix("subcohort_") for label in labels]:
            description = description_df.loc[
                (description_df["analysis"] == timing)
                & (description_df["condition"] == condition)
            ].iloc[0]
            row = {
                "condition": condition,
                "cases/controls": f"{int(description.n_cases):,}/2,817",
                "timing_from_fitbit_days_median_iqr": (
                    f"{description.time_from_streak_days_median:.0f} "
                    f"[{description.time_from_streak_days_q25:.0f}, {description.time_from_streak_days_q75:.0f}]"
                ),
            }
            for model, prefix in (("logistic", "LogReg"), ("xgboost", "XGBoost")):
                perf = performance_df.loc[
                    (performance_df["analysis"] == timing)
                    & (performance_df["condition"] == condition)
                    & (performance_df["model"] == model)
                    & (performance_df["metric"] == "auroc")
                ].iloc[0]
                comparison = comparison_df.loc[
                    (comparison_df["analysis"] == timing)
                    & (comparison_df["condition"] == condition)
                    & (comparison_df["model"] == model)
                ].iloc[0]
                row[f"{prefix} AUROC (95% CI)"] = (
                    f"{perf.estimate:.3f} ({perf.ci_lower:.3f}–{perf.ci_upper:.3f})"
                )
                row[f"{prefix} Δ vs primary (95% CI), raw/FDR p"] = (
                    f"{comparison.auroc_difference_timing_minus_primary:+.3f} "
                    f"({comparison.ci_lower:+.3f} to {comparison.ci_upper:+.3f}); "
                    f"{format_p(comparison.p_value_delong_two_sided)}/"
                    f"{format_p(comparison.p_value_fdr_bh)}"
                )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    past_table = summary.iloc[:len(labels)]
    future_table = summary.iloc[len(labels):]
    report = f"""# Past versus future diagnosis timing sensitivity analysis

## Definitions

- **Past diagnosis:** the earliest mapped diagnosis is before the first day of the 21-day Fitbit streak used for modelling.
- **Future diagnosis:** the earliest mapped diagnosis is after the final day of that 21-day Fitbit streak.
- Diagnoses during the streak are excluded from both analyses.
- Each timing-restricted case group is compared with the original primary control group (2,817 complete-case controls), so the case timing is the only intended cohort restriction.

## Descriptive case characteristics

The accompanying CSV contains age, sex, race, BMI, multimorbidity count, and timing summaries for each timing cohort. Timing below is days from Fitbit collection: for past cases it is days since diagnosis at Fitbit start; for future cases it is days to diagnosis after Fitbit start.

{markdown_table(description_df.round(1))}

## Model performance and comparison with the primary model

AUROC intervals use 1,000 percentile-bootstrap resamples. For every outcome/model/timing comparison, the primary-trained and timing-trained model are evaluated on the same held-out timing-specific participants, then compared by a two-sided paired DeLong test for correlated AUROCs. The reported difference is timing-trained minus primary-trained AUROC. Raw and Benjamini–Hochberg FDR-adjusted p-values are shown (28 comparisons total).

### Past diagnosis only

{markdown_table(past_table)}

### Future diagnosis only

{markdown_table(future_table)}

## Interpretation

These are timing-restricted, disease-versus-original-control analyses. They do not establish incident-disease prediction: the diagnosis records can reflect later coding or clinical documentation rather than true onset. The DeLong comparison answers whether the timing-trained and primary-trained models differ on the same timing-specific held-out cohort; it does not compare unpaired AUROCs from different test populations.

## Files

- `diagnosis_timing_case_descriptives.csv`
- `diagnosis_timing_performance.csv`
- `diagnosis_timing_delong_comparisons.csv`
- `diagnosis_timing_metadata.json`
- Analysis code: `src/diagnosis_timing_sensitivity.py`

No manuscript files were changed.
"""
    (OUTPUT_DIR / "diagnosis_timing_sensitivity_results.md").write_text(report)


if __name__ == "__main__":
    main()
