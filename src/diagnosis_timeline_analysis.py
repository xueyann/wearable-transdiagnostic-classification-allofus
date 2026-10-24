"""Summarize diagnosis dates relative to the Fitbit streak used for modelling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("results/diagnosis_timeline")
DIAGNOSIS_PATH = Path("data/02_intermediate/study_cohorts_20251222/diag_alltime.csv")
STREAK_PATH = Path("data/02_intermediate/study_cohorts_20260225/streaks_20260225.csv")
COHORT_PATH = Path("data/02_intermediate/study_cohorts_20260225/study_cohort_20260225.csv")
THRESHOLDS = {
    "CVD": 10,
    "Diabetes": 10,
    "OSA": 3,
    "MDD": 2,
    "Anxiety": 2,
    "ADHD": 3,
    "Bipolar": 5,
}


def first_streak() -> pd.DataFrame:
    """Return the earliest qualifying Fitbit streak for each participant."""
    streaks = pd.read_csv(STREAK_PATH, usecols=["person_id", "streak_start", "streak_end"])
    streaks[["streak_start", "streak_end"]] = streaks[["streak_start", "streak_end"]].apply(pd.to_datetime)
    return (
        streaks.sort_values(["person_id", "streak_start", "streak_end"])
        .groupby("person_id", as_index=False)
        .first()
    )


def diagnosis_dates(conditions: list[str]) -> pd.DataFrame:
    """Return earliest and latest mapped diagnosis dates per person and condition."""
    diagnoses = pd.read_csv(DIAGNOSIS_PATH)
    diagnoses = diagnoses.loc[
        diagnoses["subcohort"].isin(conditions),
        ["person_id", "subcohort", "condition_start_date"],
    ].copy()
    diagnoses["condition_start_date"] = pd.to_datetime(diagnoses["condition_start_date"])
    return diagnoses.groupby(["person_id", "subcohort"], as_index=False).agg(
        earliest_diagnosis=("condition_start_date", "min"),
        latest_diagnosis=("condition_start_date", "max"),
    )


def percent(numerator: int, denominator: int) -> float:
    """Return a percentage with a safe zero-denominator convention."""
    return 100 * numerator / denominator if denominator else float("nan")


def pre_recency_summary(data: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Summarize latest diagnosis dates before applying recency eligibility."""
    rows = []
    for condition in conditions:
        group = data.loc[data["subcohort"].eq(condition)]
        n = len(group)
        latest_days = group["latest_days_from_streak_start"]
        retained = int(group["passes_recency"].sum())
        rows.append(
            {
                "condition": condition,
                "recency_threshold_years": int(group["threshold_years"].iloc[0]),
                "candidate_participants": n,
                "retained_by_recency": retained,
                "retained_pct": percent(retained, n),
                "latest_before_streak_n": int(latest_days.lt(0).sum()),
                "latest_before_streak_pct": percent(int(latest_days.lt(0).sum()), n),
                "latest_same_day_n": int(latest_days.eq(0).sum()),
                "latest_after_streak_n": int(latest_days.gt(0).sum()),
                "latest_after_streak_pct": percent(int(latest_days.gt(0).sum()), n),
                "recency_years_median": group["recency_years"].median(),
                "recency_years_q25": group["recency_years"].quantile(0.25),
                "recency_years_q75": group["recency_years"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def earliest_timing_summary(data: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Summarize earliest diagnosis timing relative to the used Fitbit streak."""
    rows = []
    for condition in conditions:
        group = data.loc[data["subcohort"].eq(condition)]
        timing = group["earliest_days_from_streak_start"]
        before = timing.lt(0)
        during = timing.ge(0) & group["earliest_diagnosis"].le(group["streak_end"])
        after = group["earliest_diagnosis"].gt(group["streak_end"])
        rows.append(
            {
                "condition": condition,
                "participants": len(group),
                "before_streak_n": int(before.sum()),
                "before_streak_pct": percent(int(before.sum()), len(group)),
                "during_streak_n": int(during.sum()),
                "during_streak_pct": percent(int(during.sum()), len(group)),
                "after_streak_n": int(after.sum()),
                "after_streak_pct": percent(int(after.sum()), len(group)),
                "earliest_signed_days_median": timing.median(),
                "earliest_signed_days_q25": timing.quantile(0.25),
                "earliest_signed_days_q75": timing.quantile(0.75),
                "days_before_streak_median": (-timing.loc[before]).median(),
                "days_before_streak_q25": (-timing.loc[before]).quantile(0.25),
                "days_before_streak_q75": (-timing.loc[before]).quantile(0.75),
                "days_to_diagnosis_median": timing.loc[after].median(),
                "days_to_diagnosis_q25": timing.loc[after].quantile(0.25),
                "days_to_diagnosis_q75": timing.loc[after].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a dependency-free Markdown table."""
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def main() -> None:
    """Create diagnosis-timeline CSV summaries and a standalone Markdown report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = list(THRESHOLDS)
    streaks = first_streak()
    timeline = diagnosis_dates(conditions).merge(streaks, on="person_id", how="inner")
    timeline["latest_days_from_streak_start"] = (
        timeline["latest_diagnosis"] - timeline["streak_start"]
    ).dt.days
    timeline["earliest_days_from_streak_start"] = (
        timeline["earliest_diagnosis"] - timeline["streak_start"]
    ).dt.days
    timeline["recency_years"] = -timeline["latest_days_from_streak_start"] / 365
    timeline["threshold_years"] = timeline["subcohort"].map(THRESHOLDS)
    # The historical implementation has no lower bound, so future diagnoses pass.
    timeline["passes_recency"] = timeline["recency_years"] <= timeline["threshold_years"]

    pre_summary = pre_recency_summary(timeline, conditions)
    post_summary = earliest_timing_summary(timeline.loc[timeline["passes_recency"]], conditions)

    cohort = pd.read_csv(COHORT_PATH)
    cohort["index_date"] = pd.to_datetime(cohort["index_date"])
    label_columns = [f"subcohort_{condition}" for condition in conditions]
    final = cohort.melt(
        id_vars=["person_id", "index_date"],
        value_vars=label_columns,
        var_name="label",
        value_name="earliest_diagnosis",
    ).dropna(subset=["earliest_diagnosis"])
    final["subcohort"] = final["label"].str.removeprefix("subcohort_")
    final["earliest_diagnosis"] = pd.to_datetime(final["earliest_diagnosis"])
    final = final.merge(streaks, on="person_id", how="left", validate="many_to_one")
    final["earliest_days_from_streak_start"] = (
        final["earliest_diagnosis"] - final["streak_start"]
    ).dt.days
    final_summary = earliest_timing_summary(final, conditions)

    pre_summary.to_csv(OUTPUT_DIR / "pre_recency_latest_diagnosis_summary.csv", index=False)
    post_summary.to_csv(OUTPUT_DIR / "post_recency_earliest_diagnosis_summary.csv", index=False)
    final_summary.to_csv(OUTPUT_DIR / "final_primary_earliest_diagnosis_summary.csv", index=False)

    pre_columns = [
        "condition", "recency_threshold_years", "candidate_participants",
        "retained_by_recency", "retained_pct", "latest_before_streak_n",
        "latest_before_streak_pct", "latest_after_streak_n",
        "latest_after_streak_pct", "recency_years_median", "recency_years_q25",
        "recency_years_q75",
    ]
    post_columns = [
        "condition", "participants", "before_streak_n", "before_streak_pct",
        "during_streak_n", "during_streak_pct", "after_streak_n",
        "after_streak_pct", "earliest_signed_days_median",
        "earliest_signed_days_q25", "earliest_signed_days_q75",
        "days_before_streak_median", "days_to_diagnosis_median",
    ]
    report = f"""# Diagnosis timeline relative to Fitbit data collection

## Definitions and cohort logic

- The historical 20260225 primary cohort uses the earliest qualifying 21-day Fitbit streak for modelling. `streak_start` is the first analysed Fitbit day and `streak_end` is the last day of that selected streak.
- The recency code uses the **latest** mapped diagnosis date per person and condition: `recency = (streak_start - latest diagnosis) / 365`. A participant passes when this value is at most the condition-specific threshold. There is no lower bound, so diagnoses after the Fitbit streak also pass.
- After recency eligibility, the cohort retains the **earliest** mapped diagnosis date for each included condition. That is the date used below for the diagnosis-timeline description.
- Diagnosis dates are dates of mapped condition records; they should not be interpreted as verified incident disease-onset dates.

## 1. Latest diagnosis before applying the recency filter

Positive recency years mean the latest diagnosis preceded Fitbit collection; negative values mean it followed Fitbit collection. The distribution is calculated among participants with a mapped diagnosis and at least one qualifying 21-day streak, before the recency criterion was applied.

{markdown_table(pre_summary.loc[:, pre_columns].round(1))}

## 2. Earliest diagnosis after applying the recency filter

This table is immediately after recency eligibility. “During” means the earliest diagnosis occurred on or after `streak_start` and on or before `streak_end`; “after” means after the analysed 21-day streak ended. Signed timing is `earliest diagnosis − streak_start` in days (negative = past diagnosis). `days_before_streak_median` and `days_to_diagnosis_median` are calculated only within the relevant before/after group.

{markdown_table(post_summary.loc[:, post_columns].round(1))}

## Final primary cohort check

This check uses the 9,301-participant reconstructed primary cohort after its downstream demographic and measurement eligibility steps.

{markdown_table(final_summary.loc[:, post_columns].round(1))}

## Interpretation

The implemented recency definition does not ensure a historical diagnosis. For several cohorts, many participants have their latest—and, after eligibility, their earliest—mapped diagnosis after the Fitbit streak. The current disease-versus-control analysis therefore contains a mixture of temporally past and future diagnosis records. A future sensitivity analysis could require `latest diagnosis <= streak_start` or estimate performance separately for historical versus future diagnosis timing.

## Files

- `pre_recency_latest_diagnosis_summary.csv`
- `post_recency_earliest_diagnosis_summary.csv`
- `final_primary_earliest_diagnosis_summary.csv`
- Analysis code: `src/diagnosis_timeline_analysis.py`

No manuscript files were changed.
"""
    (OUTPUT_DIR / "diagnosis_timeline_results.md").write_text(report)


if __name__ == "__main__":
    main()
