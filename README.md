# Explainable AI for Humanitarian Food Insecurity Early Warning

**Benchmarking Spatiotemporal Machine Learning Approaches in the Greater Horn of Africa**

Code and derived data for the MSc Data Science thesis of Robert Odingo.

This repository is the reproducibility artifact cited in the thesis. It contains the
feature-extraction pipeline, the modelling harness, the models, the three frozen
country panels the results are computed from, and the results files behind the
reported tables and figures.

---

## What the study does

Food-insecurity early warning in the Greater Horn of Africa rests on expert
assessments issued roughly three times a year. The thesis tests whether
machine-learning forecasts of IPC Phase 3+ can complement them while being
calibrated, explained, and explicit about their own reliability. XGBoost and
LightGBM are trained on satellite-derived anomalies and phase history for FEWS NET
reporting units in Kenya, Ethiopia and Somalia (2018–2023), and evaluated one and
three months ahead against persistence and the FEWS NET projection on identical
rows. A Temporal Fusion Transformer was evaluated under a pre-registered
model-selection gate and excluded.

---

## Quickstart

```bash
git clone https://github.com/Robo3225/gha-food-insecurity-thesis
cd gha-food-insecurity-thesis
pip install -r requirements.txt

# Verify the frozen panels against the hashes published in the thesis
shasum -a 256 data/processed/v2/*_feature_panel.csv | cut -c1-16

# The modelling scripts import the harness by bare module name, so the harness
# directory must be on PYTHONPATH. Run from the repository root:
PYTHONPATH=scripts/07_modelling/harness python scripts/07_modelling/model_xgboost.py
```

The modelling scripts take no command-line arguments; their configuration is in the
files. Without the `PYTHONPATH` setting the import fails with
`ModuleNotFoundError: No module named 'harness_data'`.

Loading is checked: `harness_data.load_all()` reads the three panels, asserts the
21-column canonical predictor set, and reports `135,144` unit-months across the
three countries (Kenya 46,008 / Ethiopia 74,880 / Somalia 14,256).

---

## Frozen artifacts and their hashes

The three country panels are **frozen**: they were built once, content-hashed, and
every model reads them unchanged. The hashes below are stated in the thesis
(Appendix B.6) and are the first 16 hex characters of the file's SHA-256.

| Artifact | `sha256[:16]` | In this repository |
|---|---|---|
| `data/processed/v2/kenya_feature_panel.csv` | `461e6cec10b5ee17` | yes |
| `data/processed/v2/ethiopia_feature_panel.csv` | `ac209e23ce2b2088` | yes |
| `data/processed/v2/somalia_feature_panel.csv` | `793e22608a4ec94f` | yes |
| ACLED reduced extract | `c7eaef347363972e` | no — see *Data not redistributed* |
| Köppen-Geiger source raster `koppen_geiger_0p00833333.tif` (1991_2020) | `2130f0071dfb2904947d8ec3a0d807fac71004df76e769262004f1602e4d6a13` (full SHA-256) | no — see *Data sources* |

Please do not modify the panel files. Their content hashes are published facts
about this study; editing a panel would invalidate them.

### The freeze record

`data/processed/v2/MANIFEST.txt` is the freeze record written by
`scripts/03_crosswalks/build_feature_panel_v2.py` at build time. It records the
build timestamp, the git commit, the row/unit/month counts, the full column
schema, the SHA-256 of every input file, and the output hash.

**It covers Somalia only.** The build script writes a single manifest at a fixed
path (`data/processed/v2/MANIFEST.txt`) rather than one per country, so each
country's build overwrote the previous one and only the last survives. The Kenya
and Ethiopia freeze hashes are recorded in the table above and are verifiable
directly against the panel files published here. The dated commits that froze
each panel are in the private development repository, available to reviewers on
request; the manifest's `git commit` line refers to a commit in that repository.

---

## Repository layout

```
config/                        country configuration and Earth Engine asset ids
data/
  crosswalks/                  FNID <-> admin_code crosswalks (639 / 1,040 / 198 units)
  processed/v2/                the three frozen feature panels + MANIFEST.txt
  external/acled/metadata/     ACLED export provenance (no event data)
outputs/diagnostics/           results files behind the reported tables and figures
scripts/
  01_validation/               source-data validation
  02_inspection/               one-off inspection of the FSC shapefiles
  03_crosswalks/               crosswalk construction; panel assembly and freeze
  04_diagnostics/              the checks behind specific thesis claims
  05_smoke_test/               milestone smoke test (disposable; retained for the record)
  06_feature_extraction/       ACLED fetch, unit centroids, conflict features (Config D)
  07_modelling/                harness, models, explainability, uncertainty, transfer
    harness/                   load-time assertions, splits, baselines, FEWS benchmark
  08_visualization/            figure generators
  gee/                         Google Earth Engine feature extraction
```

### The modelling harness

Every model reads the panels through the harness in `scripts/07_modelling/harness/`,
which fails loudly rather than silently proceeding:

| Assertion | Where |
|---|---|
| the canonical 21-predictor list is present and dtype-compatible across all panels | `harness_data.py` — `assert_canonical_predictors` |
| each unit's series is a contiguous monthly run before the horizon shift is applied | `harness_target.py` |
| no training observation post-dates any test observation | `harness_splits.py` — raises `CHRONOLOGY LEAK` |
| a missing frozen panel | `harness_data.py` — raises `Frozen panel missing: …` |

A single global random seed (42) is used for every model.

---

## Results files

`outputs/diagnostics/` holds the derived metrics behind the reported results. These
are small aggregate tables — no unit-level records.

| File | Contents |
|---|---|
| `m10_metric_grid.csv` | the metric grid: PR-AUC, ROC-AUC, Brier by model, year, horizon, arm and row-set |
| `m10_xgboost_4arm.csv`, `m10_lightgbm_4arm.csv` | the four-arm ablation |
| `m10_xgboost_configs.csv`, `m10_baselines.csv` | configuration comparison; persistence and FEWS NET baselines |
| `m10_p12_rowmatched.csv` | row-matched re-score |
| `m10_tft_gate.csv` | the model-selection gate outcome for the attention-based candidate |
| `c2_shap.csv` | SHAP attribution shares |
| `c2_conformal.csv`, `c2_conformal_weighted.csv` | split-conformal coverage; the weighted extension |
| `c3_transfer.csv`, `c3_transfer_baselines.csv` | transfer to Somalia, including the ablation arms |
| `c3_conformal_transfer.csv` | conformal coverage under transfer |
| `configD.csv` | the Config D conflict arm |
| `p9_observed_phase3.csv` | Phase 3+ prevalence among observed and all labelled unit-months |

Other diagnostic output (crosswalk matching reports, exploratory summaries, scratch
test files) is not published; it is intermediate working output, not evidence for a
reported result.

---

## Figures

Figures were rendered with matplotlib. The generators for four of them are in
`scripts/08_visualization/`; for two the script was not retained. **In all cases the
underlying data are in the files listed below**, so every reported figure can be
reconstructed from published data.

| Figure | Generator | Data |
|---|---|---|
| Biophysical value-added and SHAP share | not retained | `c2_shap.csv`, `m10_metric_grid.csv` |
| Interval coverage | not retained | `c2_conformal.csv`, `c3_conformal_transfer.csv` |
| Three-share SHAP bars at h = 3 | `build_figure_d.py` | `c2_shap.csv` |
| Prevalence, benchmark and ablation panels | `build_figures.py` | `p9_observed_phase3.csv`, `m10_p12_rowmatched.csv`, `m10_metric_grid.csv`, `c3_transfer.csv` |
| Program-officer risk card | — | illustrative design, not data-derived; the caption says so |

```bash
python scripts/08_visualization/build_figures.py \
    scripts/08_visualization/ch5_v8_crosscheck_input.txt outdir/
python scripts/08_visualization/build_figure_d.py \
    outputs/diagnostics/c2_shap.csv \
    scripts/08_visualization/ch5_v8_crosscheck_input.txt outdir/
```

Both generators cross-check every value they plot against the thesis tables and
**abort rather than plot** on any mismatch. `ch5_v8_crosscheck_input.txt` is the
fixture they check against; it uses the chapter's table labels as they stood when
the scripts were written, and is committed unmodified because the scripts' checks
reference those labels.

No plotting code has been reconstructed after the fact. Where a script was not
retained, this README says so rather than supplying a lookalike.

---

## Data sources

Earth-observation features are extracted through Google Earth Engine; collection
identifiers carry the product version, which is a firmer pin than a citation alone.

| Source | Version / identifier | Reference |
|---|---|---|
| CHIRPS rainfall | `UCSB-CHG/CHIRPS/DAILY` (v2.0), aggregated to monthly | Funk et al. (2015), https://doi.org/10.1038/sdata.2015.66 · https://www.chc.ucsb.edu/data/chirps |
| MODIS NDVI | `MODIS/061/MOD13Q1` — collection 6.1, 16-day, 250 m | https://lpdaac.usgs.gov/products/mod13q1v061/ |
| MODIS land surface temperature | `MODIS/061/MOD11A2` — collection 6.1, 8-day, 1 km | Wan (2014) · https://lpdaac.usgs.gov/products/mod11a2v061/ |
| WorldPop population | `WorldPop/GP/100m/pop`, fixed 2020 reference raster | https://hub.worldpop.org/ |
| Köppen-Geiger climate class | present-day 1991–2020, 1 km (`koppen_geiger_0p00833333.tif`), ingested as a private Earth Engine asset; source raster SHA-256 in the table above | Beck et al. (2023), https://doi.org/10.1038/s41597-023-02549-6 · https://www.gloh2o.org/koppen/ |
| FEWS NET food-security assessments and projections | reporting-unit boundaries and classifications, accessed 17 August 2026 | https://fews.net/data/geographic-boundaries |
| Harmonized FEWS NET dataset (World Bank) | World Bank Data Catalog dataset 0064614; release `FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv`; CC-BY 4.0 | https://datacatalog.worldbank.org/search/dataset/0064614/harmonized-sub-national-food-security-data |
| ACLED conflict events | research-tier access, retrieved 18 September 2026, 33,964 events; reduced-extract hash `c7eaef347363972e` | Raleigh et al. (2010), https://doi.org/10.1177/0022343310378914 · https://acleddata.com/ |

`data/external/acled/metadata/acled_export_metadata.txt` records the ACLED export in
full: timestamp, endpoint, fields, per-country and per-year event counts, and the
extract hash.

The harmonized FEWS NET dataset (World Bank Data Catalog, dataset 0064614,
*Harmonized Sub-National Food Security Data — Processed Integrated Phase
Classification (IPC) sourced from FEWS NET*, CC-BY 4.0) is at
https://datacatalog.worldbank.org/search/dataset/0064614/harmonized-sub-national-food-security-data.
The pipeline consumes the release named above
(`FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv`). Its reporting units are
FEWS NET livelihood-zone × administrative intersections, each assigned the majority
IPC phase by spatial join.

---

## Data not redistributed

Some inputs are not in this repository, for licence reasons or size:

- **ACLED event data.** Research-tier access does not permit redistribution. Only the
  export metadata and the content hash are published. `scripts/06_feature_extraction/download_acled.py`
  reproduces the extract from the ACLED API given your own credentials, supplied via
  the `ACLED_USERNAME` and `ACLED_PASSWORD` environment variables.
- **FEWS NET shapefiles, the harmonized dataset, WFP and World Bank source files.**
  Obtain these from the providers above.
- **Intermediate per-source panels, climatology artifacts and lagged feature tables.**
  Rebuildable from the extraction scripts; the frozen v2 panels they feed are published.
- **Model checkpoints and training logs.**

The modelling results reproduce from the published panels and `requirements.txt`
alone; the raw sources are needed only to rebuild the panels from scratch.

---

## Requirements

Two files, deliberately separate:

- **`requirements.txt`** — the ten versions stated in Appendix B.6, pinned verbatim.
  These are the versions used for the reported results. It is not a `pip freeze`
  and lists no transitive dependencies.
- **`requirements-extraction.txt`** — the additional packages the extraction,
  crosswalk, fetch and figure code imports, **unpinned**. Their versions were not
  recorded and the thesis does not state them; a version printed here would be a
  reproducibility claim that is not backed by the record.

Python 3.13.12.

---

## Notes

**Internal identifiers in comments.** Code comments carry labels from the author's
working record: numbered decisions (`D19`, `D22`, `D27`), milestones (`M10`) and
tasks (`P12`), along with occasional references to internal documents such as the
chapter draft a cross-check fixture was written against. That record — a decision
register and project log — is not published here, so these labels are not
resolvable from this repository. They are retained because they mark where a choice
was made and recorded rather than assumed. The decisions that bear on the method
are described in the thesis.

**Earth Engine asset paths.** `config/config.py` names assets under the author's own
Earth Engine project (`thesis-project-502010`). These are genuine provenance but are
not readable by others; re-running extraction requires ingesting the equivalent
assets into your own project.

**Working records and development history.** The author's working records (a
decision register, pre-registrations, and per-country technical logs) are not part
of this repository; they are retained privately. The cryptographic freeze record
and the original development timeline are available to reviewers on request.

---

## Licence and citation

MIT — see [LICENSE](LICENSE). The licence covers the code and the author's derived
outputs in this repository; source datasets remain subject to their providers' terms.

To cite the study, cite the thesis:

> Odingo, R. (2026). *Explainable AI for Humanitarian Food Insecurity Early Warning:
> Benchmarking Spatiotemporal Machine Learning Approaches in the Greater Horn of
> Africa* [MSc thesis]. IU International University of Applied Sciences.
