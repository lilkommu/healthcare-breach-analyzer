# Healthcare Breach Impact Analyzer

Analysis of the **HHS Office for Civil Rights (OCR) Breach Portal** — every reported breach of
unsecured protected health information affecting **500+ individuals** since reporting began.
7,795 breaches, October 2009 – July 2026, ~1.04 billion individual records affected.

**Live dashboard:** https://healthcare-breach-analyzer.streamlit.app

## What the data actually shows

1. **Hacking replaced physical theft as the dominant breach mechanism.** In 2010, ~50% of
   reported breaches were theft (stolen laptops, paper files) and ~4% were hacking. By 2024,
   hacking/IT incidents are ~83% of reports and theft is nearly extinct (~1–2%).
2. **Hacking is also the most severe breach type** — median 8,000 individuals per incident,
   vs ~1,700–2,200 for every other type.
3. **Vendor (Business Associate) risk concentrates severity.** BAs file only ~15% of breach
   reports but account for **~50% of all individuals ever affected** (≈520M of ≈1.04B), with the
   highest median breach size of any entity type.
4. **The severity distribution is extremely heavy-tailed.** Median breach: ~3,900 individuals;
   mean: ~133,000. The top 5 breaches — led by Change Healthcare (192.7M, 2024) and Anthem
   (78.8M, 2015) — account for roughly a third of all individuals ever affected.
5. **Breach counts more than doubled in a decade** — from ~300/yr in the mid-2010s to ~700–790/yr in the 2020s (and ~3.7× the ~200/yr of the early 2010s).
6. **State totals reflect entity registration, not patient exposure.** MN "leads" individuals
   affected (212M) almost entirely because Change Healthcare is registered there.

## Breach segments (K-Means, k=5)

Clustered on log-severity, submission year, business-associate flag, breach type, and entity type
(`notebooks/clustering.ipynb`; model persisted in `models/`):

| # | Segment | Share | Median affected | Total affected | Character |
|---|---------|-------|-----------------|----------------|-----------|
| 0 | Modern provider hacking | 44% | ~8.2k | 293M | Providers, 100% hacking, centered ~2022 |
| 1 | Provider human error / insider | 18% | ~1.7k | 24M | Unauthorized disclosure, loss, improper disposal |
| 2 | Health-plan breaches | 12% | ~3.0k | 189M | Mixed hacking + disclosure; includes Anthem |
| 3 | Business associate / vendor | 15% | ~5.7k | **520M** | Half of all affected records, incl. Change Healthcare |
| 4 | Physical theft era (legacy) | 11% | ~2.0k | 13M | ~100% theft, mean year 2014; segment has died out |

State was tested as a clustering feature (frequency encoding) and **excluded** — it produced no
geographic structure, only mirrored big-state breach counts.

## How breaches happen (narrative text mining)

OCR writes a case narrative when it closes an investigation; `notebooks/text_mining.ipynb`
extracts attack vectors and post-breach actions from 6,379 narratives with a rule-based
classifier (dashboard tab: "How breaches happen"):

- **Hacking-family vectors are over half of narrated breaches**: other network intrusion ~21%,
  ransomware ~19%, phishing/email compromise ~18%.
- **Ransomware is recent and the most damaging vector**: essentially absent before 2016, then
  200+ narrated cases/yr through 2020–2023; median 12,060 individuals per breach — 5–7x any
  physical or human-error vector.
- **Human-error vectors persist at small scale**: misdirected mail/faxes/postings ~10%,
  insider snooping ~4%, lost records ~3% — frequent in older years, low severity.
- **Post-breach responses are boilerplate-heavy**: "additional safeguards" in ~67% of
  narratives, staff retraining ~32%, credit monitoring ~29%, discipline/termination ~12%;
  encryption is specifically mentioned in only ~3%.
- Caveats: narratives exist only for closed cases (2025–26 mostly missing), ~17% state no
  clear mechanism, and extraction is regex-based, not a trained classifier.

## Severity prediction (XGBoost + SHAP)

`notebooks/severity_model.ipynb` trains a classifier to predict whether a newly reported breach
will affect 10,000+ individuals using only at-report-time fields (entity type, breach type,
location, business-associate flag, state, date). Narrative-derived features are excluded to
avoid leakage. Performance is real but partial — AUC 0.73 (random split) / 0.67 (temporal
split, train <2023) — and SHAP analysis shows hacking incidents, network-server locations, and
recent years as the dominant risk drivers. The dashboard's "Risk model" tab includes a live
what-if scorer.

## Real dollars: HHS enforcement joined to breaches

`notebooks/enforcement.ipynb` matches 65 officially published OCR settlements and civil money
penalties (amounts stated on the [HHS Resolution Agreements page](https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/agreements/index.html);
never guessed) to breach reports — 42 audited organization-level matches. Findings: penalties
correlate only weakly with breach size (Spearman 0.14), fewer than 1% of breaches draw a
published penalty, and Anthem's record $16M settlement equals about $0.20 per breached record.
This is the only non-estimated dollar data in the project.

## Self-updating data pipeline

`.github/workflows/refresh-data.yml` runs `scripts/refresh_data.py` monthly on GitHub Actions:
it attempts to pull fresh portal exports and new enforcement actions from HHS, rebuilds every
derived dataset (clean, vectors, clusters, penalties), and commits changes — Streamlit Cloud
then redeploys automatically. Fetch failures degrade gracefully (existing data is kept).

## Known data limits (honesty notes)

- **No financial field exists in this data.** The dashboard's "Cost estimate" tab multiplies
  individuals affected by an adjustable external benchmark (default $408/record, from IBM's
  [Cost of a Data Breach report](https://www.ibm.com/reports/data-breach)) and is clearly
  labeled an estimate — directional only, since per-record scaling overstates mega-breach costs.
- **No rural/critical-access hospital flag exists.** `Covered Entity Type` has only broad
  categories, so no rural-hospital claims are made.
- Individuals affected is record-level: one person breached twice counts twice.
- The raw exports have two broken `javax.faces.component.UIPanel@...` column headers, repaired
  on load to `Name of Covered Entity` and `Business Associate Present` (HHS field order).

## Repository structure

```
data/       breach_active.csv, breach_archived.csv (raw OCR exports)
            breach_clean.csv, breach_clustered.csv, breach_vectors.csv,
            ocr_enforcement.csv, breach_penalties.csv (generated / curated)
notebooks/  eda.ipynb -> clustering.ipynb -> text_mining.ipynb
            -> severity_model.ipynb (XGBoost + SHAP) -> enforcement.ipynb (penalty join)
models/     kmeans_breach_clusters.joblib, xgb_severity.joblib, shap_importance.csv,
            feature/column manifests, cluster_profiles.csv
app/        streamlit_app.py (8-tab dashboard)
scripts/    refresh_data.py (monthly auto-refresh, run by GitHub Actions)
```

## Run locally (Windows)

```powershell
venv\Scripts\activate
streamlit run app\streamlit_app.py
```

Re-run the analysis end-to-end: execute the notebooks in order (`eda` -> `clustering` ->
`text_mining` -> `severity_model` -> `enforcement`), or run `python scripts/refresh_data.py`
to rebuild all derived datasets in one step.

## Sources

- [HHS OCR Breach Portal](https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf) (public data; entity-level reports, no PHI)
- [IBM Cost of a Data Breach](https://www.ibm.com/reports/data-breach) (cost-per-record benchmark used for labeled estimates only)
