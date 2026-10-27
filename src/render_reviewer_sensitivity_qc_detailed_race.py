"""Render the completed QC detailed-race sensitivity analysis report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reviewer_sensitivity_analysis_7day_past_qc_detailed_race.md"
DURATION = ROOT / "results" / "preindex_streak_duration_qc_detailed_race"
CORE = ROOT / "results" / "seven_day_past_preindex_qc_detailed_race_sensitivities"
LOOKBACK = ROOT / "results" / "past_lookback_sensitivity_qc_detailed_race_7day_preindex"
FUTURE = ROOT / "results" / "qc_future_diagnosis_detailed_race_exploratory"
MISSINGNESS = ROOT / "results" / "seven_day_past_preindex_qc_detailed_race_missingness"
CONTROL = ROOT / "results" / "qc_detailed_race_demographic_control"
CONDITIONS = ["CVD", "Diabetes", "OSA", "MDD", "Anxiety", "Bipolar", "ADHD"]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a Markdown table without optional dependencies."""
    if frame.empty:
        return "_No rows available._"
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "|" + "|".join("---" for _ in values.columns) + "|"
    body = ["| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def auc_cell(row: pd.Series) -> str:
    """Format AUROC with its bootstrap interval and analytic sample size."""
    return (
        f"{row.estimate:.3f} ({row.ci_lower:.3f}–{row.ci_upper:.3f}); "
        f"{int(row.n_cases):,}/{int(row.n_controls):,}"
    )


def p_value(value: float) -> str:
    """Format raw paired P-values and emphasize nominal significance."""
    text = f"{value:.3f}"
    return f"**{text}**" if value < 0.05 else text


def performance_table(
    performance: pd.DataFrame,
    analyses: list[tuple[str, str]],
    model: str,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Create a disease-by-analysis performance table."""
    rows = []
    for condition in CONDITIONS:
        row: dict[str, object] = {"Disease area": condition}
        for analysis, heading in analyses:
            match = performance.loc[
                performance.analysis.eq(analysis)
                & performance.condition.eq(condition)
                & performance.model.eq(model)
                & performance.metric.eq(metric)
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one {analysis}/{condition}/{model}/{metric} row.")
            row[heading] = auc_cell(match.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def duration_table(performance: pd.DataFrame, model: str) -> pd.DataFrame:
    """Create the seven-, 14-, and 21-day AUROC table."""
    rows = []
    for condition in CONDITIONS:
        row: dict[str, object] = {"Disease area": condition}
        for days in (7, 14, 21):
            match = performance.loc[
                performance.wear_time_definition.eq(f"{days}_consecutive_days")
                & performance.analysis.eq("past_preindex")
                & performance.condition.eq(condition)
                & performance.model.eq(model)
                & performance.metric.eq("auroc")
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one {days}-day duration row.")
            heading = f"{days} days" + (" (primary)" if days == 7 else "")
            row[heading] = auc_cell(match.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(frame: pd.DataFrame, model: str, policy: bool = False) -> pd.DataFrame:
    """Format paired DeLong outputs from either core or lookback sensitivity."""
    frame = frame.loc[frame.model.eq(model)].copy()
    if policy:
        frame["Policy"] = frame.policy.replace({
            "no_lookback": "No lookback", "halved_rules": "Halved lookback"
        })
        frame["Matched test N"] = frame.n_matched_test.map(lambda value: f"{value:,}")
        frame["Difference (95% CI)"] = frame.apply(
            lambda row: f"{row.difference_sensitivity_minus_primary:+.3f} "
            f"({row.ci_lower:+.3f} to {row.ci_upper:+.3f})", axis=1
        )
        frame["Raw paired P"] = frame.p_value_delong_two_sided.map(p_value)
        return frame[["Policy", "condition", "Matched test N", "Difference (95% CI)", "Raw paired P"]].rename(columns={"condition": "Disease area"})
    frame["Matched test N"] = frame.n_test.map(lambda value: f"{value:,}")
    frame["Primary AUROC"] = frame.auc_primary.map(lambda value: f"{value:.3f}")
    frame["Sensitivity AUROC"] = frame.auc_sensitivity.map(lambda value: f"{value:.3f}")
    frame["Difference (95% CI)"] = frame.apply(
        lambda row: f"{row.difference:+.3f} ({row.ci_lower:+.3f} to {row.ci_upper:+.3f})",
        axis=1,
    )
    frame["Raw paired P"] = frame.p_value.map(p_value)
    return frame[["condition", "Matched test N", "Primary AUROC", "Sensitivity AUROC", "Difference (95% CI)", "Raw paired P"]].rename(columns={"condition": "Disease area"})


def main() -> None:
    """Write the self-contained detailed-race reviewer response document."""
    duration_performance = pd.read_csv(DURATION / "performance.csv")
    duration_samples = pd.read_csv(DURATION / "sample_sizes.csv")
    duration_tests = pd.read_csv(DURATION / "matched_test_delong.csv")
    concordance = pd.read_csv(DURATION / "coefficient_direction_concordance.csv")
    core_performance = pd.read_csv(CORE / "performance.csv")
    core_tests = pd.read_csv(CORE / "matched_test_delong.csv")
    overlap = pd.read_csv(CORE / "descriptive_overlap_matrix.csv", index_col=0)
    lookback_performance = pd.read_csv(LOOKBACK / "performance.csv")
    lookback_tests = pd.read_csv(LOOKBACK / "matched_test_delong.csv")
    future_performance = pd.read_csv(FUTURE / "performance.csv")
    missing_flow = pd.read_csv(MISSINGNESS / "cohort_flow_complete_case.csv")
    retained = pd.read_csv(MISSINGNESS / "retention_characteristics.csv")
    control_performance = pd.read_csv(CONTROL / "performance.csv")
    control_tests = pd.read_csv(CONTROL / "matched_test_delong.csv")

    lines = [
        "# QC-filtered seven-day past-diagnosis sensitivity analysis: detailed-race rerun",
        "",
        "## Scope and primary definition",
        "",
        "This complete rerun uses the QC-filtered Fitbit summary cohort: each eligible day has at least 600 HR-zone minutes and 100 steps before continuous-streak selection. HR-zone minutes are a provisional summary-level proxy for observed heart-rate coverage, not validated minute-level coverage.",
        "",
        "The primary model uses seven consecutive QC-eligible days and the latest eligible streak. Cases have a qualifying diagnosis on or before streak start and meet the disease-specific current lookback rule. Controls are lifetime-free of all target diagnoses and relevant ICD exclusions. The deterministic seed-42 participant split is used throughout.",
        "",
        "Race is coded with White as the reference and four indicators: Black or African American, Asian, more than one population, and other/small/unknown. The final group combines all remaining recorded values, including sparse categories and no-response values. This is the primary covariate coding in every analysis below; no binary-race versus detailed-race comparison is reported.",
        "",
        "**Results/code:** duration: `src/preindex_streak_duration_qc_detailed_race.py`, `results/preindex_streak_duration_qc_detailed_race/`; core: `src/seven_day_past_preindex_qc_detailed_race_sensitivities.py`, `results/seven_day_past_preindex_qc_detailed_race_sensitivities/`; lookback: `src/past_lookback_qc_detailed_race_sensitivity.py`, `src/past_lookback_qc_detailed_race_paired_tests.py`; future timing: `src/qc_future_diagnosis_detailed_race_exploratory.py`; missingness: `src/seven_day_preindex_qc_detailed_race_missingness.py`; demographic control: `src/qc_detailed_race_demographic_control.py`.",
        "",
        "Paired comparisons use two-sided DeLong tests for correlated AUROCs on shared held-out participants. Bold P-values are raw P < 0.05; no multiplicity correction was applied.",
        "",
        "## 1. Primary QC model and duration sensitivity",
        "",
    ]
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        lines.extend([f"### {title}", "", markdown_table(duration_table(duration_performance, model)), ""])
    duration_tests["Comparison"] = duration_tests["comparison"].str.replace("_", " ")
    duration_tests["Difference (95% CI)"] = duration_tests.apply(lambda row: f"{row.difference_sensitivity_minus_7_days:+.3f} ({row.ci_lower:+.3f} to {row.ci_upper:+.3f})", axis=1)
    duration_tests["Raw paired P"] = duration_tests.p_value_delong_two_sided.map(p_value)
    lines.extend(["### Paired duration comparisons", "", markdown_table(duration_tests[["Comparison", "condition", "model", "n_matched_test", "Difference (95% CI)", "Raw paired P"]].rename(columns={"condition": "Disease area", "model": "Model", "n_matched_test": "Matched test N"})), ""])

    lines.extend(["## 2. Multimorbidity", "", "The non-exclusive overlap matrix is descriptive; condition-specific case cohorts are not a single common cohort.", "", markdown_table(overlap.reset_index().rename(columns={overlap.index.name or "index": "Primary-case cohort"})), ""])

    lines.extend(["## 3. Core robustness analyses", ""])
    core_analyses = [("primary", "Primary"), ("single_condition", "Single condition"), ("stricter_comparator", "Stricter comparator"), ("medication_unexposed", "Medication-unexposed"), ("two_date_confirmation", "Two diagnosis dates")]
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        lines.extend([f"### {title}", "", markdown_table(performance_table(core_performance, core_analyses, model)), ""])
    labels = {"single_condition": "Single-condition", "stricter_comparator": "Stricter-comparator", "medication_unexposed": "Medication-unexposed", "two_date_confirmation": "Two-date-confirmation"}
    for analysis, title in labels.items():
        lines.extend([f"### {title} paired comparisons", ""])
        for model, model_title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
            lines.extend([f"#### {model_title}", "", markdown_table(paired_table(core_tests.loc[core_tests.analysis.eq(analysis)], model)), ""])
    lines.extend(["**Definitions.** The stricter comparator excludes controls with any listed relevant medication record. The medication-unexposed analysis excludes cases and controls with a listed medication overlapping `[streak start − 30 days, streak end + 30 days]`; missing end dates are treated as one-day records. Two-date confirmation requires diagnoses on two distinct dates.", ""])

    lines.extend(["## 4. Lookback sensitivity", "", "Cases use the latest eligible seven-day streak under each policy. `no_lookback` accepts any diagnosis before index; `current_rules` is primary; `halved_rules` halves each disease-specific upper limit.", ""])
    lookback_analyses = [("current_rules", "Current (primary)"), ("no_lookback", "No lookback"), ("halved_rules", "Halved lookback")]
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        lines.extend([f"### {title}", "", markdown_table(performance_table(lookback_performance, lookback_analyses, model)), ""])
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        lines.extend([f"### {title} paired comparisons", "", markdown_table(paired_table(lookback_tests, model, policy=True)), ""])

    lines.extend(["## 5. Future-diagnosis exploratory analysis", "", "This appendix-only analysis uses the first QC-eligible seven-day streak. Future cases have their first mapped diagnosis after its end and no earlier mapped diagnosis for that condition; controls use their first QC-eligible streak.", ""])
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        lines.extend([f"### {title}", "", markdown_table(performance_table(future_performance, [("future", "Future diagnosis")], model)), ""])

    lines.extend(["## 6. Missingness and complete-case assessment", "", "Wearable QC eligibility is applied before streak selection. Missing Fitbit-derived data are not imputed; absence of a QC-eligible streak is structural non-observation. Source demographic and survey variables have zero recorded missingness, so the small incremental loss occurs during analytic-frame construction.", "", "### Cohort flow", "", markdown_table(missing_flow.rename(columns={"condition": "Disease area", "selected_cases_before_static": "Cases selected", "selected_controls_before_static": "Controls selected", "cases_after_static": "Cases after static", "controls_after_static": "Controls after static", "complete_case_cases": "Complete-case cases", "complete_case_controls": "Complete-case controls", "complete_case_total": "Complete-case total"})), "", "### Retained-versus-excluded characteristics", "", "Excluded groups are small and should not be overinterpreted.", "", markdown_table(retained), ""])

    bss_rows = []
    for condition in CONDITIONS:
        row = {"Disease area": condition}
        for model, title in (("logistic", "Logistic BSS (95% CI)"), ("xgboost", "XGBoost BSS (95% CI)")):
            metric = duration_performance.loc[(duration_performance.wear_time_definition.eq("7_consecutive_days")) & (duration_performance.condition.eq(condition)) & (duration_performance.model.eq(model)) & (duration_performance.metric.eq("bss"))].iloc[0]
            row[title] = f"{metric.estimate:.3f} ({metric.ci_lower:.3f}–{metric.ci_upper:.3f})"
        bss_rows.append(row)
    lines.extend(["## 7. Primary-model calibration metric and coefficient stability", "", "Brier skill score (BSS) is reported only for the detailed-race QC seven-day primary model and uses the held-out prevalence as reference with 1,000 percentile-bootstrap intervals.", "", markdown_table(pd.DataFrame(bss_rows)), "", "The deterministic training-partition logistic coefficients are in `results/preindex_streak_duration_qc_detailed_race/logreg_coefficients.csv`.", "", markdown_table(concordance.rename(columns={"condition": "Disease area", "features_with_all_three_estimates": "Features compared", "same_coefficient_direction_all_three": "Same direction, all", "direction_concordance_percent": "Direction concordance (%)", "fdr_significant_all_three": "FDR-significant, all", "same_direction_among_fdr_significant_all_three": "Same direction among FDR-significant"})), "", "Calibration curves in the manuscript must be regenerated for this QC/detailed-race primary cohort before finalization.", ""])

    lines.extend(["## 8. QC-matched demographic-only control", "", "The full detailed-race primary model and demographic-only control use the exact same QC-filtered seven-day past-diagnosis cohort and seed-42 split. The control includes age, sex at birth, BMI, and the same four detailed-race indicators; it omits Fitbit, survey, and other clinical covariates.", ""])
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        rows = []
        for condition in CONDITIONS:
            row: dict[str, object] = {"Disease area": condition}
            for analysis, label in (("primary_detailed_race", "Primary"), ("demographic_only_detailed_race", "Demographic-only")):
                auroc = control_performance.loc[
                    control_performance.analysis.eq(analysis)
                    & control_performance.condition.eq(condition)
                    & control_performance.model.eq(model)
                    & control_performance.metric.eq("auroc")
                ].iloc[0]
                bss = control_performance.loc[
                    control_performance.analysis.eq(analysis)
                    & control_performance.condition.eq(condition)
                    & control_performance.model.eq(model)
                    & control_performance.metric.eq("bss")
                ].iloc[0]
                row[f"{label} AUROC"] = f"{auroc.estimate:.3f} ({auroc.ci_lower:.3f}–{auroc.ci_upper:.3f})"
                row[f"{label} BSS"] = f"{bss.estimate:.3f} ({bss.ci_lower:.3f}–{bss.ci_upper:.3f})"
            rows.append(row)
        lines.extend([f"### {title}", "", markdown_table(pd.DataFrame(rows)), ""])
    control_tests["Matched test N"] = control_tests.n_test.map(lambda value: f"{value:,}")
    control_tests["Primary AUROC"] = control_tests.auc_primary.map(lambda value: f"{value:.3f}")
    control_tests["Comparison AUROC"] = control_tests.auc_comparison.map(lambda value: f"{value:.3f}")
    control_tests["Difference (95% CI)"] = control_tests.apply(lambda row: f"{row.difference_comparison_minus_primary:+.3f} ({row.ci_lower:+.3f} to {row.ci_upper:+.3f})", axis=1)
    control_tests["Raw paired P"] = control_tests.p_value_delong_two_sided.map(p_value)
    for model, title in (("logistic", "Logistic regression"), ("xgboost", "XGBoost")):
        table = control_tests.loc[control_tests.model.eq(model), ["condition", "Matched test N", "Primary AUROC", "Comparison AUROC", "Difference (95% CI)", "Raw paired P"]].rename(columns={"condition": "Disease area"})
        lines.extend([f"### Demographic-only versus primary: {title}", "", markdown_table(table), ""])
    lines.extend(["## Interpretation", "", "The QC-filtered detailed-race rerun retains the same primary scientific framing: the models describe association with recorded, potentially treated prevalent disease among participants meeting a summary-level QC criterion. They are neither incident-disease prediction models nor causal medication analyses. Future-diagnosis results are exploratory and belong in an appendix.", ""])
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
