"""Fit timing sensitivities across 21-, 14-, and 7-day continuous Fitbit streaks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.time_series import find_strict_streaks
from src.create_study_cohort import (
    exclude_by_icd, filter_by_recency, filter_streak_by_age, get_cond_df,
    get_fitbit_sumstats, get_static_features,
)
from src.past_diagnosis_streak_selection import select_optimal_past_streaks
from src.reviewer_sensitivity_analyses import SEED, assign_split, bootstrap_interval
from src.reviewer_sensitivity_performance_tests import fit_predict
from src.streak_attrition_analysis import load_primary_config, prepare_daily_fitbit

OUTPUT = ROOT / "results" / "streak_timing_sensitivity"
DEFINITIONS = {"21_consecutive_days": 21, "14_consecutive_days": 14, "7_consecutive_days": 7}


def features_and_labels(config: dict) -> tuple[list[str], list[str]]:
    """Return primary static-model feature and target-column specifications."""
    labels = config["subcohort_cols"]
    features = config["static_features"] + config["survey_features"]
    features += [f"{x}_{s}" for x in config["time_series_features"] for s in ("mean", "std")]
    features += [f"{x}_rmssd" for x in config["rmssd_features"]]
    excluded = [f"{x}_{s}" for x in config["excl_vars_ts"] for s in ("mean", "std", "rmssd")]
    excluded += config["excl_vars_static"]
    return [x for x in features if x not in excluded], labels


def earliest_streaks(streaks: pd.DataFrame) -> pd.DataFrame:
    """Select first qualifying continuous streak for each participant."""
    return streaks.sort_values(["person_id", "streak_start"]).groupby("person_id", as_index=False).first()


def latest_streaks(streaks: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Select latest qualifying continuous streak for controls in past analyses."""
    return (streaks.loc[streaks.person_id.isin(ids)].sort_values(["person_id", "streak_start"])
            .groupby("person_id", as_index=False).tail(1))


def eligible_controls(cdf: pd.DataFrame, diagnoses: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return primary controls: lifetime-free of every target condition and ICD exclusions."""
    controls = cdf.loc[~cdf.person_id.isin(diagnoses.person_id)].copy()
    return exclude_by_icd(controls, config["icd_excl"])[["person_id"]].drop_duplicates()


def add_static(cohort: pd.DataFrame, person_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply original age, sex, race, and BMI eligibility."""
    return get_static_features(
        cohort, person_df.copy(), config["person_features"],
        pd.read_csv(config["physical_measurement_df"]), config["physical_measurement_features"],
    )


def complete_frame(
    cohort: pd.DataFrame, selected: pd.DataFrame, daily: pd.DataFrame, config: dict, labels: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Build a primary-feature complete-case frame from selected Fitbit streaks."""
    features, _ = features_and_labels(config)
    selected = selected[["person_id", "streak_start", "streak_end"]].drop_duplicates("person_id")
    data = daily.merge(selected)
    data = data.loc[(data.date >= data.streak_start) & (data.date <= data.streak_end)]
    sumstats = get_fitbit_sumstats(data, config["time_series_features"], config["rmssd_features"])
    survey = pd.read_csv(config["survey_df"])[["person_id", *config["survey_features"]]].astype(int)
    frame = cohort.merge(sumstats).merge(survey, how="left")
    frame = frame[["person_id", "index_date", *labels, *features]].dropna(subset=features)
    frame[labels] = frame[labels].replace("", np.nan).notna().astype(int)
    return assign_split(frame).reset_index(drop=True), features


def combined_cohort(
    streaks: pd.DataFrame, cdf: pd.DataFrame, person_df: pd.DataFrame, config: dict, labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create timing-unrestricted primary-style cohort using first selected streak."""
    selected = earliest_streaks(streaks)
    cdf = cdf.loc[cdf.person_id.isin(selected.person_id)].copy()
    cases = cdf.loc[cdf.subcohort.notna(), ["person_id", "subcohort", "condition_start_date"]].drop_duplicates()
    controls = eligible_controls(cdf, cases, config)
    recent = filter_by_recency(cases, selected, cdf, config["recency_filter"])
    recent = recent.groupby(["person_id", "subcohort"], as_index=False).min()
    wide = pd.get_dummies(recent, columns=["subcohort"])
    for label in labels:
        condition = label.removeprefix("subcohort_")
        if label not in wide:
            wide[label] = False
        wide[label] = np.where(wide[label], wide["condition_start_date"].astype(str), "")
    wide = wide.drop(columns="condition_start_date").groupby("person_id", as_index=False)[labels].sum()
    cohort = pd.concat([wide, controls], ignore_index=True, sort=False)
    cohort = cohort.merge(selected[["person_id", "streak_start"]].rename(columns={"streak_start": "index_date"}))
    cohort = add_static(cohort, person_df, config)
    return cohort, selected


def condition_timing_cohort(
    timing: str, label: str, streaks: pd.DataFrame, diagnoses: pd.DataFrame,
    controls: pd.DataFrame, person_df: pd.DataFrame, config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one disease-versus-control timing-restricted cohort and its selected streaks."""
    condition = label.removeprefix("subcohort_")
    if timing == "past":
        case_streaks = select_optimal_past_streaks(
            streaks, diagnoses, condition, config["recency_filter"][condition]
        )
        control_streaks = latest_streaks(streaks, controls.person_id)
    elif timing == "future":
        first = earliest_streaks(streaks)
        diag = diagnoses.loc[diagnoses.subcohort.eq(condition)].groupby("person_id", as_index=False).agg(
            first_diagnosis=("condition_start_date", "min"),
            latest_diagnosis=("condition_start_date", "max"),
        )
        case_streaks = first.merge(diag)
        recency = (case_streaks.streak_start - case_streaks.latest_diagnosis).dt.days
        case_streaks = case_streaks.loc[
            case_streaks.first_diagnosis.gt(case_streaks.streak_end)
            & recency.le(config["recency_filter"][condition] * 365)
        ]
        control_streaks = first.loc[first.person_id.isin(controls.person_id)]
    else:
        raise ValueError(timing)
    case_streaks = case_streaks[["person_id", "streak_start", "streak_end"]].drop_duplicates("person_id")
    control_streaks = control_streaks[["person_id", "streak_start", "streak_end"]].drop_duplicates("person_id")
    selected = pd.concat([case_streaks, control_streaks], ignore_index=True)
    cohort = selected[["person_id", "streak_start"]].rename(columns={"streak_start": "index_date"})
    cohort[label] = np.where(cohort.person_id.isin(case_streaks.person_id), "case", "")
    cohort = add_static(cohort, person_df, config)
    return cohort, selected


def performance_rows(analysis: str, definition: str, label: str, model: str, y: np.ndarray, p: np.ndarray,
                     n_cases: int, n_controls: int, estimator: str) -> list[dict[str, object]]:
    """Produce AUROC and Brier-skill performance rows with bootstrap intervals."""
    rows = []
    for metric in ("auroc", "bss"):
        estimate, low, high = bootstrap_interval(y, p, metric)
        rows.append({"wear_time_definition": definition, "analysis": analysis,
                     "condition": label.removeprefix("subcohort_"), "model": model, "estimator": estimator,
                     "metric": metric, "estimate": estimate, "ci_lower": low, "ci_upper": high,
                     "n_cases": n_cases, "n_controls": n_controls, "n_test": len(y)})
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional dependencies."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main() -> None:
    """Fit all requested timing-by-wear-time models and write a report."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = load_primary_config()
    model_config = yaml.safe_load((ROOT / "conf" / "base_mdl_fitbit.yaml").read_text())
    _, labels = features_and_labels(model_config)
    person_df = pd.read_csv(primary["person_df"])
    cdf = get_cond_df(primary, person_df)
    cdf["subcohort"] = cdf.description.map(primary["condition_map"])
    diagnoses = cdf.loc[cdf.subcohort.notna(), ["person_id", "subcohort", "condition_start_date"]]
    controls = eligible_controls(cdf, diagnoses, primary)
    daily = prepare_daily_fitbit(primary["fitbit_summary_data"], person_df.person_id)
    performances: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []

    for definition, days in DEFINITIONS.items():
        streaks = find_strict_streaks(daily[["person_id", "date"]].copy(), "date", days)
        streaks = filter_streak_by_age(streaks, person_df.copy())
        combined, combined_selected = combined_cohort(streaks, cdf, person_df, primary, labels)
        combined_frame, features = complete_frame(combined, combined_selected, daily, model_config, labels)
        for analysis in ("combined", "past", "future"):
            for label in labels:
                if analysis == "combined":
                    frame = combined_frame
                    selected = combined_selected
                else:
                    cohort, selected = condition_timing_cohort(
                        analysis, label, streaks, diagnoses, controls, person_df, primary
                    )
                    frame, features = complete_frame(cohort, selected, daily, model_config, [label])
                controls_frame = frame.loc[frame[label].eq(0)]
                cases_frame = frame.loc[frame[label].eq(1)]
                data = pd.concat([cases_frame, controls_frame], ignore_index=True)
                samples.append({"wear_time_definition": definition, "analysis": analysis,
                                "condition": label.removeprefix("subcohort_"), "n_cases": len(cases_frame),
                                "n_controls": len(controls_frame), "n_total": len(data),
                                "n_test_cases": int((data.split.eq("test") & data[label].eq(1)).sum()),
                                "n_test_controls": int((data.split.eq("test") & data[label].eq(0)).sum())})
                selected_out = selected.copy()
                selected_out["wear_time_definition"] = definition
                selected_out["analysis"] = analysis
                selected_out["condition"] = label.removeprefix("subcohort_")
                selected_rows.append(selected_out)
                for model in ("logistic", "xgboost"):
                    ids, y, probability, estimator = fit_predict(data, features, label, model)
                    performances.extend(performance_rows(
                        analysis, definition, label, model, y, probability,
                        len(cases_frame), len(controls_frame), estimator
                    ))
    performance = pd.DataFrame(performances)
    sample_df = pd.DataFrame(samples)
    performance.to_csv(OUTPUT / "performance.csv", index=False)
    sample_df.to_csv(OUTPUT / "sample_sizes.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(OUTPUT / "selected_streaks.csv", index=False)
    report = ["# Fitbit streak-duration and diagnosis-timing sensitivity analysis", "",
              "## Definitions", "",
              "- **Combined:** all primary-eligible cases regardless of diagnosis timing, indexed at the first qualifying streak.",
              "- **Past:** selected streak begins after the first mapped diagnosis; its start also satisfies the original latest-diagnosis recency criterion. The latest such streak is selected condition-specifically.",
              "- **Future:** first mapped diagnosis occurs after the end of the first qualifying streak; original latest-diagnosis recency criterion is retained.",
              "- Controls are lifetime-free of all target conditions and ICD exclusions. Combined/future controls use the first qualifying streak; past controls use the latest qualifying streak.",
              "- Past and future can include the same participant across disease areas, but cases for a given disease cannot use the same streak as both past and future because the first diagnosis is on opposite sides of the future streak.",
              "", "## Sample sizes", ""]
    order = ["CVD", "Diabetes", "OSA", "MDD", "Anxiety", "Bipolar", "ADHD"]
    for definition in DEFINITIONS:
        report += [f"### {definition.replace('_', ' ')}", ""]
        display = sample_df.loc[sample_df.wear_time_definition.eq(definition)].copy()
        display["cases/controls"] = display.apply(lambda x: f"{x.n_cases:,}/{x.n_controls:,}", axis=1)
        report.append(markdown_table(display[["analysis", "condition", "cases/controls", "n_test_cases", "n_test_controls"]]))
        report.append("")
        rows = []
        for analysis in ("combined", "past", "future"):
            for condition in order:
                row = {"analysis": analysis, "condition": condition}
                for model, title in (("logistic", "LogReg"), ("xgboost", "XGBoost")):
                    r = performance.loc[(performance.wear_time_definition.eq(definition)) & (performance.analysis.eq(analysis)) & (performance.condition.eq(condition)) & (performance.model.eq(model)) & (performance.metric.eq("auroc"))].iloc[0]
                    row[title + " AUROC (95% CI)"] = f"{r.estimate:.3f} ({r.ci_lower:.3f}–{r.ci_upper:.3f})"
                rows.append(row)
        report += ["#### AUROC", "", markdown_table(pd.DataFrame(rows)), ""]
    report += ["## Methods", "", "Both model types use the existing fixed seed (42), fixed 70/30 participant split within each analysis frame, primary feature set, 1,000 percentile-bootstrap AUROC/Brier-skill intervals, unregularized logistic regression with the existing weakly-regularized fallback, and 5-fold grid-search-tuned XGBoost.", "", "## Files", "", "- `sample_sizes.csv`", "- `performance.csv`", "- `selected_streaks.csv`", "- Analysis code: `src/streak_timing_sensitivity_models.py`"]
    (OUTPUT / "streak_timing_sensitivity_results.md").write_text("\n".join(report) + "\n")
    (OUTPUT / "metadata.json").write_text(json.dumps({"seed": SEED, "bootstrap_replicates": 1000}, indent=2))


if __name__ == "__main__":
    main()
