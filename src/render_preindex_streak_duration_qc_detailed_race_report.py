"""Render tables and coefficient-direction summaries for duration sensitivity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "preindex_streak_duration_qc_detailed_race"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a compact Markdown table without optional packages."""
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def main() -> None:
    """Write a human-readable report and coefficient concordance table."""
    samples = pd.read_csv(OUTPUT / "sample_sizes.csv")
    performance = pd.read_csv(OUTPUT / "performance.csv")
    comparisons = pd.read_csv(OUTPUT / "matched_test_delong.csv")
    coefficients = pd.read_csv(OUTPUT / "logreg_coefficients.csv")

    coefficient_summary: list[dict[str, object]] = []
    for condition, group in coefficients.loc[coefficients.feature.ne("const")].groupby("condition"):
        pivot = group.pivot(index="feature", columns="streak_days", values="coefficient")
        fdr = group.pivot(index="feature", columns="streak_days", values="p_fdr_bh")
        shared = pivot.dropna(subset=[7, 14, 21])
        same_direction = (shared[7].gt(0) == shared[14].gt(0)) & (
            shared[7].gt(0) == shared[21].gt(0)
        )
        significant_all = fdr.loc[shared.index].lt(0.05).all(axis=1)
        coefficient_summary.append(
            {
                "condition": condition,
                "features_with_all_three_estimates": len(shared),
                "same_coefficient_direction_all_three": int(same_direction.sum()),
                "direction_concordance_percent": round(100 * same_direction.mean(), 1),
                "fdr_significant_all_three": int(significant_all.sum()),
                "same_direction_among_fdr_significant_all_three": int(
                    same_direction.loc[significant_all].sum()
                ),
            }
        )
    concordance = pd.DataFrame(coefficient_summary)
    concordance.to_csv(OUTPUT / "coefficient_direction_concordance.csv", index=False)

    lines = [
        "# QC-filtered pre-index past-diagnosis streak-duration sensitivity",
        "",
        "Every day in each selected continuous run has at least 600 HR-zone minutes "
        "and 100 steps. HR-zone minutes are a provisional summary-level proxy, not "
        "validated observed-HR wear time. Cases have a qualifying diagnosis on or before "
        "their selected latest eligible streak; controls are lifetime-free of all target "
        "conditions and use their latest eligible streak. All models use the deterministic "
        "seed-42 participant split.",
        "",
        "## Sample sizes and AUROC",
        "",
    ]
    for days in (7, 14, 21):
        lines.extend([f"### {days} consecutive days", ""])
        display_samples = samples.loc[samples.streak_days.eq(days)].copy()
        display_samples["cases / controls"] = display_samples.apply(
            lambda row: f"{row.n_cases:,} / {row.n_controls:,}", axis=1
        )
        lines.extend([markdown_table(display_samples[["condition", "cases / controls"]]), ""])
        result_rows = []
        for condition in display_samples["condition"]:
            row: dict[str, object] = {"condition": condition}
            for model, title in (("logistic", "LogReg"), ("xgboost", "XGBoost")):
                result = performance.loc[
                    performance.wear_time_definition.eq(f"{days}_consecutive_days")
                    & performance.condition.eq(condition)
                    & performance.model.eq(model)
                    & performance.metric.eq("auroc")
                ].iloc[0]
                row[f"{title} AUROC (95% CI)"] = (
                    f"{result.estimate:.3f} ({result.ci_lower:.3f}\u2013{result.ci_upper:.3f})"
                )
            result_rows.append(row)
        lines.extend([markdown_table(pd.DataFrame(result_rows)), ""])

    lines.extend(
        [
            "## Matched 14/21 versus 7-day AUROC comparisons",
            "",
            "Two-sided paired DeLong tests compare duration-specific models on participants "
            "held out in both analysis frames. Benjamini-Hochberg adjustment is across all "
            "28 comparisons; consult the reported raw and adjusted P-values.",
            "",
            markdown_table(comparisons),
            "",
            "## Logistic coefficient direction concordance",
            "",
            "Raw signs include weak/null coefficients and are not substantive evidence. "
            "All features meeting FDR < 0.05 at every duration had the same sign across "
            "7, 14, and 21 days; raw magnitudes should not be compared across cohorts.",
            "",
            markdown_table(concordance),
            "",
            "## Files",
            "",
            "- `sample_sizes.csv`: analytic cases and controls.",
            "- `performance.csv`: AUROC and Brier skill-score results.",
            "- `matched_test_delong.csv`: paired duration comparisons.",
            "- `logreg_coefficients.csv`: seed-42 training-partition logistic coefficients.",
            "- `coefficient_direction_concordance.csv`: coefficient sign concordance.",
        ]
    )
    (OUTPUT / "preindex_streak_duration_sensitivity_results.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
