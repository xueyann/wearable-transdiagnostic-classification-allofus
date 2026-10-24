"""Estimate past-diagnosis cohort size after optimally selecting Fitbit streaks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.time_series import find_strict_streaks
from src.create_study_cohort import (
    exclude_by_icd,
    filter_streak_by_age,
    get_static_features,
    get_fitbit_sumstats,
)
from src.streak_attrition_analysis import (
    load_primary_config,
    prepare_daily_fitbit,
)

OUTPUT = ROOT / "results" / "past_diagnosis_optimal_streak"


def find_all_21_of_30_windows(daily: pd.DataFrame) -> pd.DataFrame:
    """Return every 21-observation window that fits within 30 calendar days."""
    rows: list[dict[str, object]] = []
    dates_df = daily[["person_id", "date"]].sort_values(["person_id", "date"])
    max_span = pd.Timedelta(days=29)
    for person_id, group in dates_df.groupby("person_id", sort=False):
        dates = group["date"].to_numpy()
        for start in range(len(dates) - 20):
            end = start + 20
            if dates[end] - dates[start] <= max_span:
                rows.append(
                    {
                        "person_id": person_id,
                        "streak_start": dates[start],
                        "streak_end": dates[end],
                    }
                )
    return pd.DataFrame(rows)


def model_complete_ids(
    cohort: pd.DataFrame,
    selected_streaks: pd.DataFrame,
    daily: pd.DataFrame,
    base_config: dict,
) -> set[int]:
    """Apply the primary model's exact complete-case feature requirement."""
    selected = daily.merge(selected_streaks[["person_id", "streak_start", "streak_end"]])
    selected = selected.loc[
        (selected["date"] >= selected["streak_start"])
        & (selected["date"] <= selected["streak_end"])
    ]
    ts_features = base_config["time_series_features"]
    sumstats = get_fitbit_sumstats(selected, ts_features, base_config["rmssd_features"])
    survey_features = base_config["survey_features"]
    survey = pd.read_csv(base_config["survey_df"])[["person_id"] + survey_features]
    features = base_config["static_features"] + survey_features
    features += [f"{feature}_{stat}" for feature in ts_features for stat in ["mean", "std"]]
    features += [f"{feature}_rmssd" for feature in base_config["rmssd_features"]]
    excluded = [
        f"{feature}_{stat}"
        for feature in base_config["excl_vars_ts"]
        for stat in ["mean", "std", "rmssd"]
    ] + base_config["excl_vars_static"]
    features = [feature for feature in features if feature not in excluded]
    model_df = cohort.merge(sumstats).merge(survey, how="left").dropna(subset=features)
    return set(model_df["person_id"])


def select_optimal_past_streaks(
    candidate_streaks: pd.DataFrame,
    diagnoses: pd.DataFrame,
    condition: str,
    max_years: int,
) -> pd.DataFrame:
    """Choose the latest streak after first diagnosis within primary recency rules.

    "Past diagnosis" means the selected streak starts on/after the first mapped
    diagnosis for that disease. Recency retains the primary definition: streak
    start minus the latest mapped diagnosis must be no more than the
    disease-specific lookback (with no lower bound).
    """
    diag = diagnoses.loc[diagnoses["subcohort"] == condition]
    diag = diag.groupby("person_id", as_index=False)["condition_start_date"].agg(
        earliest_diagnosis="min", latest_diagnosis="max"
    )
    candidates = candidate_streaks.merge(diag)
    recency_days = (candidates["streak_start"] - candidates["latest_diagnosis"]).dt.days
    candidates = candidates.loc[
        (candidates["streak_start"] >= candidates["earliest_diagnosis"])
        & (recency_days <= max_years * 365)
    ]
    return candidates.sort_values("streak_start").groupby("person_id", as_index=False).tail(1)


def select_latest_streaks(candidate_streaks: pd.DataFrame, person_ids: pd.Series) -> pd.DataFrame:
    """Select the latest qualifying streak for controls, which have no recency rule."""
    return (
        candidate_streaks.loc[candidate_streaks["person_id"].isin(person_ids)]
        .sort_values("streak_start")
        .groupby("person_id", as_index=False)
        .tail(1)
    )


def main() -> None:
    """Run past-diagnosis sample-size estimates across wear-time definitions."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = load_primary_config()
    with open(ROOT / "conf" / "base_mdl_fitbit.yaml") as f:
        base_config = yaml.safe_load(f)
    person_df = pd.read_csv(primary["person_df"])
    from src.create_study_cohort import get_cond_df

    cdf = get_cond_df(primary, person_df)
    cdf["subcohort"] = cdf["description"].map(primary["condition_map"])
    diagnoses = cdf.loc[cdf["subcohort"].notna(), ["person_id", "subcohort", "condition_start_date"]]
    controls = cdf.loc[~cdf["person_id"].isin(diagnoses["person_id"])].copy()
    controls = exclude_by_icd(controls, primary["icd_excl"])[["person_id"]].drop_duplicates()
    daily = prepare_daily_fitbit(primary["fitbit_summary_data"], person_df["person_id"])
    definitions = {
        "21 consecutive days": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 21),
        "21 days within 30 days": find_all_21_of_30_windows(daily),
        "14 consecutive days": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 14),
        "7 consecutive days": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 7),
    }
    definitions = {
        definition: filter_streak_by_age(streaks, person_df.copy())
        for definition, streaks in definitions.items()
    }
    rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for definition, candidates in definitions.items():
        control_streaks = select_latest_streaks(candidates, controls["person_id"])
        for condition, max_years in primary["recency_filter"].items():
            case_streaks = select_optimal_past_streaks(candidates, diagnoses, condition, max_years)
            case_streaks = case_streaks[[
                "person_id", "streak_start", "streak_end", "earliest_diagnosis", "latest_diagnosis"
            ]]
            case_streaks["group"] = "case"
            ctrl = control_streaks[["person_id", "streak_start", "streak_end"]].copy()
            ctrl["earliest_diagnosis"] = pd.NaT
            ctrl["latest_diagnosis"] = pd.NaT
            ctrl["group"] = "control"
            selected = pd.concat([case_streaks, ctrl], ignore_index=True)
            cohort = selected[["person_id", "streak_start"]].rename(columns={"streak_start": "index_date"})
            cohort[f"subcohort_{condition}"] = np.where(cohort["person_id"].isin(case_streaks["person_id"]), "case", "")
            cohort = get_static_features(
                cohort,
                person_df.copy(),
                primary["person_features"],
                pd.read_csv(primary["physical_measurement_df"]),
                primary["physical_measurement_features"],
            )
            selected = selected.merge(cohort[["person_id"]], how="inner")
            model_ids = model_complete_ids(cohort, selected, daily, base_config)
            case_ids = set(case_streaks["person_id"])
            control_ids = set(ctrl["person_id"])
            rows.append(
                {
                    "definition": definition,
                    "disease_area": condition,
                    "candidate_past_cases_before_demographics": len(case_ids),
                    "past_cases_demographic_ready": len(case_ids & set(cohort["person_id"])),
                    "past_cases_model_complete": len(case_ids & model_ids),
                    "controls_demographic_ready": len(control_ids & set(cohort["person_id"])),
                    "controls_model_complete": len(control_ids & model_ids),
                    "latest_diagnosis_to_selected_streak_days_median_iqr": "{:.0f} ({:.0f}-{:.0f})".format(
                        (case_streaks["streak_start"] - case_streaks["latest_diagnosis"]).dt.days.median(),
                        (case_streaks["streak_start"] - case_streaks["latest_diagnosis"]).dt.days.quantile(.25),
                        (case_streaks["streak_start"] - case_streaks["latest_diagnosis"]).dt.days.quantile(.75),
                    ),
                }
            )
            case_streaks["definition"] = definition
            case_streaks["disease_area"] = condition
            details.append(case_streaks)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "past_diagnosis_optimal_streak_sample_sizes.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(OUTPUT / "selected_case_streaks.csv", index=False)
    table = result.pivot(index="disease_area", columns="definition", values="past_cases_model_complete")
    controls_table = result.pivot(index="disease_area", columns="definition", values="controls_model_complete")
    lines = [
        "# Past-diagnosis sample size with condition-specific optimal streak selection", "",
        "A case qualifies when the selected Fitbit streak begins after the first mapped diagnosis of that disease area. The original recency rule is retained: the difference between selected streak start and the latest mapped diagnosis must be no more than the disease-specific lookback. For each case, the latest qualifying streak is selected. Controls remain lifetime-free of all target conditions and use their latest qualifying streak.", "",
        "## Final complete-case sample sizes (cases / controls)", "",
        "| Disease area | 21 consecutive | 21 within 30 | 14 consecutive | 7 consecutive |",
        "|---|---:|---:|---:|---:|",
    ]
    order = ["CVD", "Diabetes", "OSA", "MDD", "Anxiety", "Bipolar", "ADHD"]
    labels = {"21 consecutive days": "21 consecutive", "21 days within 30 days": "21 within 30", "14 consecutive days": "14 consecutive", "7 consecutive days": "7 consecutive"}
    for condition in order:
        vals = []
        for definition in labels:
            case_n = int(table.loc[condition, definition])
            ctrl_n = int(controls_table.loc[condition, definition])
            vals.append(f"{case_n:,} / {ctrl_n:,}")
        lines.append(f"| {condition} | " + " | ".join(vals) + " |")
    lines += ["", "The CSV additionally reports counts before demographics, demographic-ready counts, and the diagnosis-to-selected-streak lag."]
    (OUTPUT / "past_diagnosis_optimal_streak_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
