# 2025_Plant_Costs

Data and code for **"Power Plant Cost Data Availability for the Energy Transition"** by Steven Dahlke (https://doi.org/10.2139/ssrn.6103410).

The project builds a longitudinal, plant-level dataset (1994–2023) linking financial data from the **Federal Energy Regulatory Commission (FERC) Form 1** with generator characteristics from the **U.S. Energy Information Administration (EIA) Form 860**. It uses the data to describe the coverage and content of public power plant cost data, and to estimate coal plant capital investment during successive EPA environmental compliance periods.

## Data sources

All raw inputs were downloaded from the [Public Utility Data Liberation (PUDL)](https://catalyst.coop/pudl/) project by Catalyst Cooperative and are stored in `data/`:

| File | Source |
|---|---|
| `core_ferc1__yearly_steam_plants_sched402.csv` | FERC Form 1, Schedule 402 (plant-level costs) |
| `core_ferc1__yearly_balance_sheet_assets_sched110.csv` | FERC Form 1, Schedule 110 |
| `core_eia860__scd_generators.csv` | EIA Form 860 generator data |
| `core_eia__entity_plants.csv`, `core_eia__entity_utilities.csv` | EIA plant and utility entities |
| `out_eia__yearly_generators.parquet`, `out_eia__yearly_utilities.csv` | EIA yearly generator and utility data |

Other files in `data/` are produced by the scripts below. Spreadsheets ending in `_cleaningSandbox.xlsx` are working files from the data cleaning process.

## Repository structure

| Folder | Contents |
|---|---|
| `code/` | Notebooks and scripts, numbered in run order |
| `data/` | Raw PUDL inputs and the linked datasets |
| `intermediate data/` | Cleaned datasets and diagnostic outputs |
| `output/` | Figures and tables used in the paper |

## Setup

The scripts were run with Python 3.12 and the package versions in `requirements.txt`. Script 12 requires Python 3.10 or later.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS or Linux
pip install -r requirements.txt
```

The notebooks also need Jupyter. Figures use the Cambria font; on systems without it, matplotlib falls back to a default font and prints a warning.

## Run order

Run each script from the `code/` folder. Each script locates the project root from its own location.

| Step | File | Reads | Writes |
|---|---|---|---|
| 1 | `1 clean large steam plants.ipynb` | PUDL inputs in `data/` | `data/eia_gen.csv`, `data/ferc1_plants_utilitynames.csv` |
| 2 | `2 match datasets.ipynb` | Step 1 outputs | `data/cost_data_v1.csv`, `data/plant_costs_with_history.csv` |
| 3 | `3 make map.py` | `cost_data_v1.csv` | `data/cost_data_2023.csv`, map in `output/` |
| 4 | `4 Overall Descriptive Statistics.py` | `cost_data_v1.csv` | Observation counts in `output/` |
| 5 | `5 Characteristics by Technology Type.py` | `cost_data_2023.csv`, `eia_gen.csv` | Sample composition and coverage tables in `output/` |
| 6 | `6 Capacity by technology and year.py` | `cost_data_v1.csv` | `intermediate data/cleaned_cost_data.csv`, capacity figures |
| 7 | `7 clean retirement tags.py` | `cleaned_cost_data.csv` | `intermediate data/cleaned_cost_dataV2.csv` |
| 8–11 | Capital cost, age and subcategory scripts | `cleaned_cost_dataV2.csv` | Descriptive figures and tables |
| 12 | `12 Capex econometrics data prep.py` | `cleaned_cost_dataV2.csv` | `cleaned_cost_dataV3.csv`, `capex_econometric/df_model_ready.csv` |
| 13 | `13 capex econometrics model.py` | `df_model_ready.csv` | Coal investment model results, figures and tables |
| 14–17 | Capital, new plant, operating cost and employee scripts | `cleaned_cost_dataV3.csv` | Descriptive figures and tables |
| 18 | `18 post2001 coal plants.py` | `df_model_ready.csv` | Post-2001 coal plant check and all-vintage robustness results |

## Coal investment model (scripts 12, 13 and 18)

The model is a panel difference-in-differences regression with year-specific treatment effects, estimated with `linearmodels.PanelOLS`:

```
capex_flow ~ is_coal_ge25mw_pre2001 + event_1 + ... + event_28 + d_capacity_mw_x + TimeEffects
```

- **Outcome:** the annual change in each plant's reported capital balance.
- **Treated plants:** coal plants of 25 MW or more installed in 2000 or earlier.
- **Comparison plants:** natural gas combined-cycle and combustion turbine plants, nuclear plants, and coal plants outside the treatment definition.
- **Year effects** (`TimeEffects`) absorb changes common to all plants in a given year. The `event_t` coefficients measure the treated-comparison difference in year 1995 + t, relative to the difference in 1995.
- **Standard errors** are clustered by Form 1 plant entry.

**Sample rules:**
- Plants must have at least 10 years of reported data under the same Form 1 plant name (script 12).
- Script 12 interpolates or removes specific plant-years with reporting errors; each correction is listed in the script.
- Changes in capital balance that span missing reporting years are excluded (scripts 13 and 18).

`linearmodels` does not add an intercept to formulas by default; this specification relies on `TimeEffects` instead.

**Main outputs in `output/`:**

| Output | Files |
|---|---|
| Coal investment premium by year | `coal_capex_event_study.png`, `coal_capex_event_study_table_complete_tab_separated.txt` |
| Natural gas and nuclear comparison groups | `coal_capex_event_study_NGcontrol*`, `coal_capex_event_study_NUKEcontrol*` |
| All-vintage treatment group | `coal_capex_RC_allvintage_vs_main.png`, `coal_capex_RC_allvintage_complete_tab_separated.txt` |
| Post-2001 coal plant capital balances | `post2001_coal_individual_trajectories.png` |

## Citation

> Dahlke, Steven, Power Plant Cost Data Availability for the Energy Transition (January 20, 2026). https://doi.org/10.2139/ssrn.6103410

## License

Code and processed data are released under CC0 1.0 (see `LICENSE`). Raw data are from PUDL; see the [PUDL documentation](https://catalystcoop-pudl.readthedocs.io/) for its terms.
