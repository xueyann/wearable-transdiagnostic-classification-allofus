"""Audit cohort attrition under alternative Fitbit wear-time definitions.

This script reproduces the original primary-cohort rules (notably, no
medication or OSA-device exclusion) after changing only the Fitbit wear-time
qualification.  It also reports the number and timing of qualifying streaks.
"""

from __future__ import annotations

import json
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
    filter_by_recency,
    filter_streak_by_age,
    get_cond_df,
    get_static_features,
    get_fitbit_sumstats,
)


OUTPUT = ROOT / "results" / "streak_definition_attrition"
BASELINE_CONFIG_COMMIT = "34f1659^"


def load_primary_config() -> dict:
    """Load the historic configuration used for the February 2025 primary cohort."""
    import subprocess

    config_text = subprocess.check_output(
        ["git", "show", f"{BASELINE_CONFIG_COMMIT}:conf/create_study_cohort.yaml"],
        cwd=ROOT,
        text=True,
    )
    return yaml.safe_load(config_text)["total_cohort"]


def prepare_daily_fitbit(
    path: str,
    person_ids: pd.Series,
    min_hr_zone_minutes: int | None = None,
    min_steps: int | None = None,
) -> pd.DataFrame:
    """Normalize daily summaries and apply optional valid-day QC before features.

    Thresholds are applied after main-sleep and nap records are reconciled, but
    before feature engineering, complete-case filtering, and streak finding.
    ``hr_zone_minutes`` sums the four exported heart-rate zone-minute fields;
    it remains a proxy pending confirmation that the fields exhaustively encode
    observed heart-rate minutes in the source CDR.
    """
    path_obj = Path(path)
    paths = sorted(path_obj.glob("*.parquet")) if path_obj.is_dir() else [path_obj]
    return pd.concat(
        [
            _prepare_daily_fitbit_file(
                file_path, person_ids, min_hr_zone_minutes, min_steps
            )
            for file_path in paths
        ],
        ignore_index=True,
    )


def _prepare_daily_fitbit_file(
    path: Path,
    person_ids: pd.Series,
    min_hr_zone_minutes: int | None,
    min_steps: int | None,
) -> pd.DataFrame:
    """Normalize one summary shard and apply optional valid-day thresholds."""
    df = pd.read_parquet(path)
    df = df.loc[df["person_id"].isin(person_ids)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_main_sleep"] = df["is_main_sleep"].replace({"false": False, "true": True})
    df["hr_zone_minutes"] = df[
        ["fat_burn_min", "cardio_min", "peak_min", "out_of_range_min"]
    ].sum(axis=1)
    df["hr_zone_coverage"] = df["hr_zone_minutes"] / 1_440

    naps = (
        df.loc[df["is_main_sleep"] == False, ["person_id", "date", "minute_in_bed"]]
        .groupby(["person_id", "date"], as_index=False)["minute_in_bed"]
        .sum()
        .rename(columns={"minute_in_bed": "nap_min"})
    )
    df = df.loc[df["is_main_sleep"] == True].merge(
        naps, on=["person_id", "date"], how="left"
    )
    df["nap_min"] = df["nap_min"].fillna(0)

    if min_hr_zone_minutes is not None:
        df = df.loc[df["hr_zone_minutes"] >= min_hr_zone_minutes]
    if min_steps is not None:
        df = df.loc[df["steps"] >= min_steps]

    for col in ["very_active_minutes", "fairly_active_minutes", "lightly_active_minutes"]:
        df[col] = df[col].fillna(0)
    df["sleep_efficiency"] = df["minute_asleep"] / df["minute_in_bed"]
    df["deep_perc"] = df["minute_deep"] / df["minute_in_bed"]
    df["rem_perc"] = df["minute_rem"] / df["minute_in_bed"]
    df["light_perc"] = df["minute_light"] / df["minute_in_bed"]
    df["wake_perc"] = df["minute_wake"] / df["minute_in_bed"]
    df["fat_burn_hr_to_light_active_ratio"] = (
        df["fat_burn_min"] / df["lightly_active_minutes"]
    ).fillna(0)
    df["cardio_hr_to_fairly_active_ratio"] = (
        df["cardio_min"] / df["fairly_active_minutes"]
    ).fillna(0)
    df["peak_hr_to_very_active_ratio"] = (
        df["peak_min"] / df["very_active_minutes"]
    ).fillna(0)
    df = df.drop(columns=["calories_bmr", "elevation", "floors", "minute_restless"])
    df = df.drop(columns="is_main_sleep").dropna()
    if df.shape[0] != df[["person_id", "date"]].drop_duplicates().shape[0]:
        raise ValueError("Expected one eligible main-sleep row per person and date.")
    return df


def count_raw_fitbit_people(path: str) -> int:
    """Count source participants without materializing all Fitbit records."""
    path_obj = Path(path)
    paths = sorted(path_obj.glob("*.parquet")) if path_obj.is_dir() else [path_obj]
    person_ids: set[int] = set()
    for file_path in paths:
        person_ids.update(pd.read_parquet(file_path, columns=["person_id"])["person_id"].unique())
    return len(person_ids)


def find_days_within_window(df: pd.DataFrame, min_days: int, window_days: int) -> pd.DataFrame:
    """Find each person's earliest qualifying window having ``min_days`` records.

    A 30-day window is inclusive of both endpoints: the first and 21st
    observations must therefore be at most 29 calendar days apart.
    """
    daily = df[["person_id", "date"]].sort_values(["person_id", "date"])
    rows: list[dict[str, object]] = []
    max_span = pd.Timedelta(days=window_days - 1)
    for person_id, group in daily.groupby("person_id", sort=False):
        dates = group["date"].to_numpy()
        for start in range(len(dates) - min_days + 1):
            end = start + min_days - 1
            if dates[end] - dates[start] <= max_span:
                rows.append(
                    {
                        "person_id": person_id,
                        "streak_start": dates[start],
                        "streak_end": dates[end],
                        "length": min_days,
                        "definition": f"{min_days}_of_{window_days}",
                    }
                )
                break
    return pd.DataFrame(rows)


def make_cohort(
    streaks: pd.DataFrame,
    daily_attrition: dict[str, int],
    config: dict,
    person_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    definition: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all primary disease, recency, and demographic eligibility rules."""
    attrition = dict(daily_attrition)
    streaks = filter_streak_by_age(streaks.copy(), person_df.copy())
    attrition["Age 18 years or older at qualifying window start"] = streaks["person_id"].nunique()

    cdf = condition_df.loc[condition_df["person_id"].isin(streaks["person_id"])].copy()
    cdf["subcohort"] = cdf["description"].map(config["condition_map"])
    cases = cdf.loc[
        cdf["subcohort"].notna(), ["person_id", "subcohort", "condition_start_date"]
    ].drop_duplicates()
    controls = cdf.loc[~cdf["person_id"].isin(cases["person_id"])].copy()
    controls = exclude_by_icd(controls, config["icd_excl"])[["person_id"]].drop_duplicates()
    attrition["Disease subcohort or eligible control"] = (
        cases["person_id"].nunique() + controls["person_id"].nunique()
    )

    recent_cases = filter_by_recency(cases, streaks, cdf, config["recency_filter"])
    attrition["After disease-specific recency criteria plus controls"] = (
        recent_cases["person_id"].nunique() + controls["person_id"].nunique()
    )
    recent_cases = recent_cases.groupby(["person_id", "subcohort"], as_index=False).min()
    recent_cases = pd.get_dummies(recent_cases, columns=["subcohort"])
    subcohort_cols = [c for c in recent_cases if c.startswith("subcohort_")]
    for col in subcohort_cols:
        recent_cases[col] = np.where(
            recent_cases[col], recent_cases["condition_start_date"].astype(str), ""
        )
    recent_cases = recent_cases.drop(columns="condition_start_date").groupby("person_id", as_index=False).sum()
    cohort = pd.concat([recent_cases, controls], ignore_index=True, sort=False)
    earliest = streaks.groupby("person_id", as_index=False)["streak_start"].min()
    cohort = cohort.merge(earliest.rename(columns={"streak_start": "index_date"}))
    cohort = get_static_features(
        cohort,
        person_df.copy(),
        config["person_features"],
        pd.read_csv(config["physical_measurement_df"]),
        config["physical_measurement_features"],
    )
    attrition["All required demographic variables (age, sex, race, BMI)"] = len(cohort)
    return cohort, attrition


def model_complete_case_count(
    cohort: pd.DataFrame,
    streaks: pd.DataFrame,
    daily: pd.DataFrame,
    base_config: dict,
) -> int:
    """Return the primary XGBoost complete-case count for an alternate cohort."""
    earliest = streaks.merge(
        streaks.groupby("person_id", as_index=False)["streak_start"].min(),
        on=["person_id", "streak_start"],
    )
    selected = daily.merge(earliest[["person_id", "streak_start", "streak_end"]])
    selected = selected.loc[
        (selected["date"] >= selected["streak_start"])
        & (selected["date"] <= selected["streak_end"])
    ]
    selected = selected.loc[selected["person_id"].isin(cohort["person_id"])]
    sumstats = get_fitbit_sumstats(
        selected, base_config["time_series_features"], base_config["rmssd_features"]
    )
    survey_features = base_config["survey_features"]
    survey = pd.read_csv(base_config["survey_df"])[["person_id"] + survey_features]
    features = base_config["static_features"] + survey_features
    features += [f"{x}_{stat}" for x in base_config["time_series_features"] for stat in ["mean", "std"]]
    features += [f"{x}_rmssd" for x in base_config["rmssd_features"]]
    excluded = [
        f"{x}_{stat}"
        for x in base_config["excl_vars_ts"]
        for stat in ["mean", "std", "rmssd"]
    ] + base_config["excl_vars_static"]
    features = [feature for feature in features if feature not in excluded]
    return int(len(cohort.merge(sumstats).merge(survey, how="left").dropna(subset=features)))


def describe_streaks(streaks: pd.DataFrame, definition: str) -> dict[str, object]:
    """Summarize multiplicity and calendar timing of qualifying strict streaks."""
    counts = streaks.groupby("person_id").size()
    first = streaks.groupby("person_id")["streak_start"].min()
    last = streaks.groupby("person_id")["streak_start"].max()
    span = (last - first).dt.days
    return {
        "definition": definition,
        "people_with_qualifying_streak": int(counts.size),
        "qualifying_streaks": int(len(streaks)),
        "streaks_per_person_mean": float(counts.mean()),
        "streaks_per_person_median_iqr": [
            float(counts.median()), float(counts.quantile(0.25)), float(counts.quantile(0.75))
        ],
        "people_with_more_than_one_streak_n_pct": [
            int((counts > 1).sum()), float((counts > 1).mean() * 100)
        ],
        "first_streak_start_median_iqr": [str(first.median().date()), str(first.quantile(.25).date()), str(first.quantile(.75).date())],
        "last_streak_start_median_iqr": [str(last.median().date()), str(last.quantile(.25).date()), str(last.quantile(.75).date())],
        "first_to_last_start_days_median_iqr": [
            float(span.median()), float(span.quantile(.25)), float(span.quantile(.75))
        ],
    }


def main() -> None:
    """Run the alternate-definition attrition audit and write tabular results."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_primary_config()
    with open(ROOT / "conf" / "base_mdl_fitbit.yaml") as f:
        base_config = yaml.safe_load(f)
    person_df = pd.read_csv(config["person_df"])
    condition_df = get_cond_df(config, person_df)
    daily = prepare_daily_fitbit(config["fitbit_summary_data"], person_df["person_id"])
    daily_attrition = {
        "Having Fitbit data": count_raw_fitbit_people(config["fitbit_summary_data"]),
        "Fitbit data linked to an EHR participant": int(person_df["person_id"].nunique()),
    }
    definitions = {
        "7 consecutive days": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 7),
        "14 consecutive days": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 14),
        "21 consecutive days (primary)": find_strict_streaks(daily[["person_id", "date"]].copy(), "date", 21),
        "21 days within 30 days": find_days_within_window(daily, 21, 30),
    }
    attrition_rows: list[dict[str, object]] = []
    streak_rows: list[dict[str, object]] = []
    cohorts: dict[str, pd.DataFrame] = {}
    for definition, streaks in definitions.items():
        attrition = dict(daily_attrition)
        attrition["Fitbit wear-time qualification"] = int(streaks["person_id"].nunique())
        cohort, downstream = make_cohort(
            streaks, attrition, config, person_df, condition_df, definition
        )
        cohorts[definition] = cohort
        downstream["Final complete-case primary-model feature set"] = model_complete_case_count(
            cohort, streaks, daily, base_config
        )
        for step, n in downstream.items():
            attrition_rows.append({"definition": definition, "step": step, "n": n})
        streak_rows.append(describe_streaks(streaks, definition))
    pd.DataFrame(attrition_rows).to_csv(OUTPUT / "attrition_long.csv", index=False)
    attrition_wide = pd.DataFrame(attrition_rows).pivot(index="step", columns="definition", values="n")
    attrition_wide.to_csv(OUTPUT / "attrition_wide.csv")
    pd.DataFrame(streak_rows).to_csv(OUTPUT / "streak_multiplicity_timing.csv", index=False)
    for definition, cohort in cohorts.items():
        safe = definition.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        cohort.to_csv(OUTPUT / f"cohort_{safe}.csv", index=False)
    with open(OUTPUT / "run_metadata.json", "w") as f:
        json.dump({"config_commit": BASELINE_CONFIG_COMMIT, "definitions": list(definitions)}, f, indent=2)


if __name__ == "__main__":
    main()
