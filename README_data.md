# Data and configuration — AGD-Sentinel (Cortinas)

## cortinas_campaign_data.csv
Measured shear-strength parameters of the colluvial sliding layer at the Cortinas
landslide (Toledo, Colombia). Consolidated-drained direct shear tests.
Source: Buenahora et al. (2026a), Bull. Eng. Geol. Environ. 85:98, Table 4.
Documented failure: August 2021. The 2021 values were measured on already-failed
(moved) material two months after the event. Note: `depth_m` is the SAMPLING depth
of each specimen, not the failure-surface depth used in the model (see config).

## cortinas_config.json
Model configuration. `z_reference_m` (13) is the approximate depth reached by the
2021 failure surface (model geometry), sampled over `z_m` = [8, 16]; it must not be
confused with the specimen sampling depth above. `beta_reference_deg` (30) is the
maximum terrain slope, sampled over [26, 34]. The field-implied linear degradation
rates (alpha = 1.60 kPa/yr; delta = 2.375 deg/yr) derive from the measured
2017->2021 states and match the revised manuscript (Section 2.3).

## Seeds (reproducibility)
Each analysis script uses its own documented seed; figures are reproduced by:
- master_run.py (seed 42): Figs. 2, 3, 6 + master_metrics.json
- reviewer_analyses.py (seed 7): Figs. 4, 8, 9; baselines (Table 2); Sobol values
- missing_analyses.py (seed 11): Fig. 7 (PDP); critical-region, coverage, timings
- agd_corrected.py (seed 2026): Supplementary Figs. S1-S3
- deterioration_laws.py (deterministic, no seed): Fig. 5
