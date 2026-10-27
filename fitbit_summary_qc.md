# Fitbit summary-data quality control

## Scope

This QC layer uses only the exported daily Fitbit summaries. It does not read
or retain minute-level Fitbit records. It is applied before the existing daily
feature engineering, complete-case filtering, and continuous-streak search.

## Planned valid-day definition

1. Normalize daily sleep records by retaining the main-sleep row and summing
   non-main sleep intervals as naps for the same participant-date.
2. Calculate `hr_zone_minutes` as the sum of `fat_burn_min`, `cardio_min`,
   `peak_min`, and `out_of_range_min`, and divide by 1,440 for the displayed
   coverage proportion.
3. Provisionally require at least 600 HR-zone minutes and at least 100 steps
   for every day in a continuous streak of at least seven days.
4. Apply existing feature transformations and complete-case rules only after
   the QC-valid daily records have been selected.

`hr_zone_minutes` is a summary-level proxy. It will be described as
HR-zone-minute coverage, rather than confirmed observed-HR-minute adherence,
until CDR metadata confirms that the four zone categories are exhaustive and
mutually exclusive for observed heart-rate data.

## Planned reporting

- Source fields, missingness, and daily plausibility summaries.
- Attrition from processed daily summaries through the HR, step, combined
  valid-day, seven-day streak, adult, disease-specific, and final complete-case
  stages.
- In final selected continuous runs (each at least seven days): median, IQR, 5th/95th percentiles of HR-zone
  coverage and the proportion meeting each threshold.
- Extreme percentiles and implausible-value counts for steps, activity minutes,
  HR-zone minutes, total sleep, and sleep-stage minutes.
- A follow-up model-performance sensitivity after validating the semantics of
  `hr_zone_minutes` against CDR documentation.

## CDR quality-field status

The current local daily extract does not contain a sleep quality field. The
current environment has no attached `WORKSPACE_CDR` dataset, so
`sleep_daily_summary_ext` cannot yet be inspected. If it becomes available,
its supplied quality field will be added as a descriptive check before deciding
whether it should be part of a sensitivity definition.

## Results

### Implementation and scope

The QC hook is implemented in `src/streak_attrition_analysis.py` and is invoked
by `src/fitbit_summary_qc.py`. It is placed after main-sleep/nap reconciliation
and before the prior feature engineering, `dropna()` complete-case filtering, and
continuous-streak search. The existing model uses the full selected continuous run,
not a fixed-width seven-day slice; therefore QC is required for every day of that run. The analysis uses `data/02_intermediate/fitbit_data_summary/`.
Machine-readable outputs are in `results/fitbit_summary_qc/`.

### Source-summary inventory

The frozen extract comprises 100 Parquet files with 26,154,403 rows from
52,334 participants (2014-12-05 through 2023-10-01). Before daily
normalization, 12.68% of rows repeat a participant-date within a source shard;
86.51% are marked `is_main_sleep=true` and 13.49% `false`. The processing
therefore retains the main-sleep row and aggregates non-main sleep duration as
naps before applying QC. The heart-rate zone fields and sleep total fields have
no missing values in the raw export; sleep-stage fields are missing in 26.08%
of raw rows, and `very_active_minutes` is missing in 13.45%. The existing
complete-case preprocessing removes records with missing analytic fields after
this QC layer.

### Overall retention

| Stage | Current primary | After provisional QC | Loss | Retention |
|---|---:|---:|---:|---:|
| Raw exported daily-summary rows | 26,154,403 | 26,154,403 | 0 | 100.0% |
| People with any raw summary row | 52,334 | 52,334 | 0 | 100.0% |
| Daily records after current sleep normalization/preprocessing | 11,460,160 | 11,460,160 | 0 | 100.0% |
| Daily records with HR-zone minutes >=600 | 11,460,160 | 11,397,241 | 62,919 | 99.5% |
| Daily records with >=100 steps | 11,460,160 | 11,407,221 | 52,939 | 99.5% |
| Daily records meeting both thresholds | 11,460,160 | 11,358,447 | 101,713 | 99.1% |
| People with >=1 QC-valid day | 31,949 | 31,901 | 48 | 99.8% |
| People with a seven-day streak | 25,826 | 25,688 | 138 | 99.5% |
| People with an adult qualifying streak | 25,825 | 25,687 | 138 | 99.5% |

### Disease-specific final complete-case cohort retention

| Disease area | Primary cases/controls | QC cases/controls | Total loss |
|---|---:|---:|---:|
| CVD | 3,536 / 4,283 | 3,528 / 4,274 | 17 |
| Diabetes | 3,050 / 4,283 | 3,040 / 4,274 | 19 |
| OSA | 3,346 / 4,283 | 3,335 / 4,274 | 20 |
| MDD | 3,014 / 4,283 | 3,003 / 4,274 | 20 |
| Anxiety | 3,578 / 4,283 | 3,561 / 4,274 | 26 |
| Bipolar | 1,028 / 4,283 | 1,027 / 4,274 | 10 |
| ADHD | 783 / 4,283 | 778 / 4,274 | 14 |

### Adherence-proxy distribution in final selected primary continuous runs

Across disease-specific final continuous runs (each at least seven days), the median HR-zone-minute total was
1,440 minutes/day and the 95th percentile was also 1,440 minutes/day. The
5th percentile ranged from 1,364 (MDD) to 1,391 (diabetes) minutes/day.
The proportion meeting the provisional >=600-minute threshold ranged from
99.52% (MDD) to 99.79% (ADHD); the proportion meeting >=100 steps ranged from
99.51% (bipolar) to 99.53% (ADHD). Full median/IQR/5th/95th/99th percentile
results for every disease and both definitions are in
`results/fitbit_summary_qc/final_window_adherence_distribution.csv`.

### Plausibility checks in final selected primary continuous runs

The 99th percentile of steps ranged from 23,761 to 24,875/day and no final
selected day exceeded 100,000 steps. Activity-minute totals had 99th
percentiles of 1,183–1,195 minutes/day and did not exceed 1,440. Total time in
bed had 99th percentiles of 700–718 minutes/day and no selected day exceeded
1,440 minutes. There were no days with `minute_asleep > minute_in_bed`.
Sleep-stage minutes exceeded time in bed in only 0.005%–0.007% of selected
person-days. Full values and flags are in
`results/fitbit_summary_qc/final_window_plausibility.csv`.

### Critical interpretation and next action

The HR-zone-minute value cannot yet be described as observed HR wear time. It
is exactly 1,440 for the median and upper quartile of every final cohort, and
0.48%–0.63% of selected person-days have a total greater than 1,440. Therefore,
the 600-minute threshold is nearly non-discriminating and its apparent 99.1%
daily retention should **not** be used as evidence of ten hours of observed HR.
It is retained in code/results only as a transparent provisional proxy analysis.

No QC-restricted model-performance sensitivity has been run: doing so would
misrepresent the above proxy as validated HR adherence. The next step is to
obtain CDR8 metadata or `sleep_daily_summary_ext` access, verify the
`heart_rate_summary.minute_in_zone` semantics, and then either (a) rerun the
model sensitivity under a validated summary-level coverage rule or (b) report
this descriptive QC and explicitly acknowledge that verified wear-time coverage
was not available in the frozen extract.
