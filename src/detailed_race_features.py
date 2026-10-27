"""Shared detailed-race feature coding for QC model reruns."""
from __future__ import annotations

import pandas as pd

RACE_REFERENCE = "White"
RACE_FEATURES = [
    "race_black",
    "race_asian",
    "race_multiple",
    "race_other_small_unknown",
]


def add_detailed_race_features(
    frame: pd.DataFrame, features: list[str], person: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Replace binary non-White race with requested White-reference indicators."""
    race = person[["person_id", "race"]].copy()
    race["race_group"] = race["race"].replace(
        {
            "White": "white",
            "Black or African American": "black",
            "Asian": "asian",
            "More than one population": "multiple",
        }
    )
    race.loc[
        ~race["race_group"].isin(["white", "black", "asian", "multiple"]),
        "race_group",
    ] = "other_small_unknown"
    for group in ("black", "asian", "multiple", "other_small_unknown"):
        race[f"race_{group}"] = race["race_group"].eq(group).astype(int)
    updated = frame.drop(columns=RACE_FEATURES, errors="ignore").merge(
        race[["person_id", *RACE_FEATURES]], on="person_id", how="left", validate="one_to_one"
    )
    if updated[RACE_FEATURES].isna().any().any():
        raise ValueError("Detailed race indicators are missing for analytic participants.")
    updated[RACE_FEATURES] = updated[RACE_FEATURES].astype(int)
    updated_features = [feature for feature in features if feature != "race_nonwhite"]
    return updated, [*updated_features, *RACE_FEATURES]
