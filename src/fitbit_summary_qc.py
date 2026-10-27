"""Audit summary-level Fitbit QC before primary preprocessing and streak selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age, get_cond_df
from src.seven_day_past_preindex import frame_for
from src.streak_attrition_analysis import (
    count_raw_fitbit_people,
    load_primary_config,
    prepare_daily_fitbit,
)
from src.streak_timing_sensitivity_models import (
    eligible_controls,
    features_and_labels,
    markdown_table,
)


OUTPUT = ROOT / "results" / "fitbit_summary_qc"
MIN_HR_ZONE_MINUTES = 600
MIN_STEPS = 100
STREAK_DAYS = 7


def source_row_count(path: str) -> int:
    """Return Parquet row count from file metadata without materializing records."""
    source = Path(path)
    files = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
    return sum(pq.ParquetFile(file).metadata.num_rows for file in files)


def summary_row(
    values: pd.Series, threshold: float | None = None
) -> dict[str, float | int]:
    """Return requested distributional statistics for one daily QC measure."""
    valid = values.dropna()
    row: dict[str, float | int] = {"n_days": len(valid)}
    for quantile, name in ((0.05, "p5"), (0.25, "q1"), (0.5, "median"),
                           (0.75, "q3"), (0.95, "p95"), (0.99, "p99")):
        row[name] = float(valid.quantile(quantile)) if len(valid) else np.nan
    row["min"] = float(valid.min()) if len(valid) else np.nan
    row["max"] = float(valid.max()) if len(valid) else np.nan
    if threshold is not None:
        row["proportion_meeting_threshold"] = float((valid >= threshold).mean())
    return row


def selected_daily(
    daily: pd.DataFrame, selected: pd.DataFrame, final_ids: pd.Series
) -> pd.DataFrame:
    """Return daily summaries from final analytic participants' selected windows."""
    windows = selected.loc[selected.person_id.isin(final_ids), [
        "person_id", "streak_start", "streak_end"
    ]].drop_duplicates("person_id")
    result = daily.merge(windows, on="person_id", how="inner")
    return result.loc[
        result.date.between(result.streak_start, result.streak_end)
    ].copy()


def final_window_descriptives(
    condition: str, definition: str, daily: pd.DataFrame
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize adherence and plausibility for one final selected-window set."""
    adherence: list[dict[str, object]] = []
    for metric, threshold in (("hr_zone_coverage", MIN_HR_ZONE_MINUTES / 1_440),
                              ("hr_zone_minutes", MIN_HR_ZONE_MINUTES),
                              ("steps", MIN_STEPS)):
        row = {"condition": condition, "definition": definition, "metric": metric}
        row.update(summary_row(daily[metric], threshold))
        adherence.append(row)

    activity_total = daily[[
        "sedentary_minutes", "lightly_active_minutes", "fairly_active_minutes",
        "very_active_minutes",
    ]].sum(axis=1)
    stage_total = daily[["minute_deep", "minute_light", "minute_rem", "minute_wake"]].sum(axis=1)
    measures = {
        "steps": daily.steps,
        "activity_minutes_total": activity_total,
        "hr_zone_minutes": daily.hr_zone_minutes,
        "minute_in_bed": daily.minute_in_bed,
        "minute_asleep": daily.minute_asleep,
        "sleep_stage_minutes_total": stage_total,
        "minute_deep": daily.minute_deep,
        "minute_light": daily.minute_light,
        "minute_rem": daily.minute_rem,
        "minute_wake": daily.minute_wake,
    }
    plausibility: list[dict[str, object]] = []
    for metric, values in measures.items():
        row = {"condition": condition, "definition": definition, "metric": metric}
        row.update(summary_row(values))
        plausibility.append(row)

    checks = {
        "steps_gt_100000": daily.steps.gt(100_000),
        "activity_minutes_total_gt_1440": activity_total.gt(1_440),
        "hr_zone_minutes_gt_1440": daily.hr_zone_minutes.gt(1_440),
        "minute_in_bed_gt_1440": daily.minute_in_bed.gt(1_440),
        "minute_asleep_gt_in_bed": daily.minute_asleep.gt(daily.minute_in_bed),
        "sleep_stage_sum_gt_in_bed": stage_total.gt(daily.minute_in_bed),
    }
    for name, mask in checks.items():
        plausibility.append({
            "condition": condition,
            "definition": definition,
            "metric": name,
            "n_days": len(daily),
            "flagged_days": int(mask.sum()),
            "flagged_proportion": float(mask.mean()),
        })
    return adherence, plausibility


def main() -> None:
    """Run summary-level QC and write machine-readable attrition/descriptive outputs."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = load_primary_config()
    cfg = yaml.safe_load((ROOT / "conf" / "base_mdl_fitbit.yaml").read_text())
    _, labels = features_and_labels(cfg)
    person = pd.read_csv(primary["person_df"])
    data_path = primary["fitbit_summary_data"]

    daily_primary = prepare_daily_fitbit(data_path, person.person_id)
    daily_qc = prepare_daily_fitbit(
        data_path,
        person.person_id,
        min_hr_zone_minutes=MIN_HR_ZONE_MINUTES,
        min_steps=MIN_STEPS,
    )
    primary_streaks = find_strict_streaks(
        daily_primary[["person_id", "date"]].copy(), "date", STREAK_DAYS
    )
    qc_streaks = find_strict_streaks(
        daily_qc[["person_id", "date"]].copy(), "date", STREAK_DAYS
    )
    primary_adult_streaks = filter_streak_by_age(primary_streaks, person.copy())
    qc_adult_streaks = filter_streak_by_age(qc_streaks, person.copy())

    hr_valid = daily_primary.hr_zone_minutes.ge(MIN_HR_ZONE_MINUTES)
    step_valid = daily_primary.steps.ge(MIN_STEPS)
    attrition = pd.DataFrame([
        {
            "stage": "Raw exported daily-summary rows",
            "primary_n": source_row_count(data_path),
            "qc_n": source_row_count(data_path),
            "unit": "rows",
        },
        {
            "stage": "People with any raw exported summary row",
            "primary_n": count_raw_fitbit_people(data_path),
            "qc_n": count_raw_fitbit_people(data_path),
            "unit": "people",
        },
        {
            "stage": "Daily records after sleep normalization and existing preprocessing",
            "primary_n": len(daily_primary),
            "qc_n": len(daily_primary),
            "unit": "person-days",
        },
        {
            "stage": f"Daily records with HR-zone minutes ≥ {MIN_HR_ZONE_MINUTES}",
            "primary_n": len(daily_primary),
            "qc_n": int(hr_valid.sum()),
            "unit": "person-days",
        },
        {
            "stage": f"Daily records with steps ≥ {MIN_STEPS}",
            "primary_n": len(daily_primary),
            "qc_n": int(step_valid.sum()),
            "unit": "person-days",
        },
        {
            "stage": "Daily records meeting both QC thresholds",
            "primary_n": len(daily_primary),
            "qc_n": int((hr_valid & step_valid).sum()),
            "unit": "person-days",
        },
        {
            "stage": "People with at least one QC-valid day",
            "primary_n": daily_primary.person_id.nunique(),
            "qc_n": daily_qc.person_id.nunique(),
            "unit": "people",
        },
        {
            "stage": f"People with a {STREAK_DAYS}-consecutive-day streak",
            "primary_n": primary_streaks.person_id.nunique(),
            "qc_n": qc_streaks.person_id.nunique(),
            "unit": "people",
        },
        {
            "stage": "People with a qualifying streak at age 18 years or older",
            "primary_n": primary_adult_streaks.person_id.nunique(),
            "qc_n": qc_adult_streaks.person_id.nunique(),
            "unit": "people",
        },
    ])
    attrition["absolute_loss"] = attrition.primary_n - attrition.qc_n
    attrition["retention_vs_primary"] = attrition.qc_n / attrition.primary_n

    cdf = get_cond_df(primary, person)
    cdf["subcohort"] = cdf.description.map(primary["condition_map"])
    diagnoses = cdf.loc[cdf.subcohort.notna(), [
        "person_id", "subcohort", "condition_start_date"
    ]]
    controls = eligible_controls(cdf, diagnoses, primary)
    cohort_rows: list[dict[str, object]] = []
    adherence_rows: list[dict[str, object]] = []
    plausibility_rows: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        for definition, daily, streaks in (
            ("primary", daily_primary, primary_adult_streaks),
            ("qc_600_hr_zone_minutes_100_steps", daily_qc, qc_adult_streaks),
        ):
            frame, _, cases, selected = frame_for(
                condition,
                label,
                primary,
                cfg,
                person,
                diagnoses,
                controls,
                daily,
                streaks,
                primary["recency_filter"][condition],
            )
            n_cases = int(frame[label].sum())
            n_controls = len(frame) - n_cases
            cohort_rows.append({
                "condition": condition,
                "definition": definition,
                "selected_case_streaks": len(cases),
                "selected_control_streaks": len(selected) - len(cases),
                "final_complete_case_cases": n_cases,
                "final_complete_case_controls": n_controls,
                "final_complete_case_total": len(frame),
            })
            final_daily = selected_daily(daily, selected, frame.person_id)
            adherence, plausibility = final_window_descriptives(
                condition, definition, final_daily
            )
            adherence_rows.extend(adherence)
            plausibility_rows.extend(plausibility)

    cohort_attrition = pd.DataFrame(cohort_rows)
    adherence_df = pd.DataFrame(adherence_rows)
    plausibility_df = pd.DataFrame(plausibility_rows)
    attrition.to_csv(OUTPUT / "qc_attrition_overall.csv", index=False)
    cohort_attrition.to_csv(OUTPUT / "qc_attrition_by_condition.csv", index=False)
    adherence_df.to_csv(OUTPUT / "final_window_adherence_distribution.csv", index=False)
    plausibility_df.to_csv(OUTPUT / "final_window_plausibility.csv", index=False)
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "daily_hr_proxy": "fat_burn_min + cardio_min + peak_min + out_of_range_min",
        "hr_zone_minimum": MIN_HR_ZONE_MINUTES,
        "steps_minimum": MIN_STEPS,
        "streak_definition": (
    f"Continuous run of at least {STREAK_DAYS} QC-valid days; all selected run days are used."
),
        "caveat": "HR-zone minutes are not called observed HR minutes without CDR metadata confirmation.",
    }, indent=2) + "\n")

    report = [
        "# Fitbit summary-level QC results",
        "",
        "## Overall attrition",
        "",
        markdown_table(attrition.assign(
            retention_vs_primary=attrition.retention_vs_primary.map(lambda x: f"{x:.1%}")
        )),
        "",
        "## Final complete-case cohort retention",
        "",
        markdown_table(cohort_attrition),
        "",
        "## Files",
        "",
        "- `qc_attrition_overall.csv`",
        "- `qc_attrition_by_condition.csv`",
        "- `final_window_adherence_distribution.csv`",
        "- `final_window_plausibility.csv`",
    ]
    (OUTPUT / "fitbit_summary_qc_results.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
