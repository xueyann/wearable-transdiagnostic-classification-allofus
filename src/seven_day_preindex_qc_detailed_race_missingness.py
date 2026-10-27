"""Describe complete-case retention in corrected seven-day pre-index cohorts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seven_day_past_preindex import frame_for, load_inputs
from src.streak_attrition_analysis import prepare_daily_fitbit
from lib.time_series import find_strict_streaks
from src.create_study_cohort import filter_streak_by_age
from src.detailed_race_features import add_detailed_race_features
from src.streak_timing_sensitivity_models import add_static


OUTPUT = ROOT / "results" / "seven_day_past_preindex_qc_detailed_race_missingness"


def retention_characteristics(
    static_cohort: pd.DataFrame, frame: pd.DataFrame, label: str, condition: str
) -> list[dict[str, object]]:
    """Summarize demographic and BMI characteristics by complete-case retention."""
    static_cohort = static_cohort.copy()
    static_cohort["retention_group"] = static_cohort["person_id"].isin(
        frame["person_id"]
    ).map({True: "retained_complete_case", False: "excluded_after_static"})
    rows = []
    for group, data in static_cohort.groupby("retention_group"):
        rows.append(
            {
                "condition": condition,
                "retention_group": group,
                "n": len(data),
                "case_percent": 100 * data[label].eq("case").mean(),
                "age_mean": data["age"].mean(),
                "bmi_mean": data["BMI"].mean(),
                "male_percent": 100 * data["sex_male"].mean(),
                "nonwhite_percent": 100 * data["race_nonwhite"].mean(),
            }
        )
    return rows


def main() -> None:
    """Write cohort flow, variable missingness, and retention-characteristic tables."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary, config, labels, person, diagnoses, controls, _, _ = load_inputs()
    daily = prepare_daily_fitbit(
        primary["fitbit_summary_data"], person.person_id, min_hr_zone_minutes=600, min_steps=100
    )
    streaks = filter_streak_by_age(
        find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 7), person.copy()
    )
    flow_rows: list[dict[str, object]] = []
    characteristics: list[dict[str, object]] = []

    for label in labels:
        condition = label.removeprefix("subcohort_")
        frame, features, cases, selected = frame_for(
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
        frame, _ = add_detailed_race_features(frame, features, person)
        cohort = selected[["person_id", "streak_start"]].rename(
            columns={"streak_start": "index_date"}
        )
        cohort[label] = cohort["person_id"].isin(cases["person_id"]).map(
            {True: "case", False: ""}
        )
        static_cohort = add_static(cohort, person, primary)
        flow_rows.append(
            {
                "condition": condition,
                "selected_cases_before_static": len(cases),
                "selected_controls_before_static": selected["person_id"].nunique() - len(cases),
                "cases_after_static": int(static_cohort[label].eq("case").sum()),
                "controls_after_static": int(static_cohort[label].eq("").sum()),
                "complete_case_cases": int(frame[label].sum()),
                "complete_case_controls": int((1 - frame[label]).sum()),
                "complete_case_total": len(frame),
            }
        )
        characteristics.extend(
            retention_characteristics(static_cohort, frame, label, condition)
        )

    pd.DataFrame(flow_rows).to_csv(OUTPUT / "cohort_flow_complete_case.csv", index=False)
    pd.DataFrame(characteristics).to_csv(
        OUTPUT / "retention_characteristics.csv", index=False
    )
    survey = pd.read_csv(config["survey_df"])
    survey_features = config["survey_features"]
    pd.DataFrame(
        {
            "variable": survey_features,
            "missing_n": [int(survey[column].isna().sum()) for column in survey_features],
            "missing_pct": [
                100 * survey[column].isna().mean() for column in survey_features
            ],
        }
    ).to_csv(OUTPUT / "survey_variable_missingness.csv", index=False)
    person_missingness = (
        person[["date_of_birth", "race", "sex_at_birth"]]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    person_missingness.to_csv(OUTPUT / "person_variable_missingness.csv", index=False)


if __name__ == "__main__":
    main()
