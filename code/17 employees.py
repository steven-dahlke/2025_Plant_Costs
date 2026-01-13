import os
from pathlib import Path
import pandas as pd
import numpy as np


# 1. Define the project root directory
# -----------------------------------------------------------------------------
try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")


# Load cleaned cost data into a DataFrame
# -----------------------------------------------------------------------------
# The CSV lives at {root}/intermediate data/cleaned_cost_dataV3.csv
csv_path = Path(root) / "intermediate data" / "cleaned_cost_dataV3.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Could not find cleaned cost CSV at: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows from {csv_path}")

# Drop rows where avg_num_employees is missing
# -----------------------------------------------------------------------------
before_count = len(df)
df = df.dropna(subset=["avg_num_employees"]) if "avg_num_employees" in df.columns else df
after_count = len(df)
print(f"Dropped {before_count - after_count:,} rows missing 'avg_num_employees' ({after_count:,} remain)")

# Filter to report_year == 2023 and compute employees per MW per plant
# -----------------------------------------------------------------------------
if "report_year" not in df.columns:
    raise KeyError("Column 'report_year' not found in data")

df_2023 = df[df["report_year"] == 2023].copy()
print(f"Rows for 2023: {len(df_2023):,}")

# Determine capacity column
capacity_col = "capacity_mw_x"
if capacity_col not in df_2023.columns:
    raise KeyError(f"Expected capacity column '{capacity_col}' not found in data")
print(f"Using capacity column: {capacity_col}")

# Avoid division by zero
df_2023[capacity_col] = pd.to_numeric(df_2023[capacity_col], errors="coerce")
df_2023.loc[df_2023[capacity_col] == 0, capacity_col] = np.nan

if "avg_num_employees" not in df_2023.columns:
    raise KeyError("Column 'avg_num_employees' not found in data")

df_2023["employees_per_MW"] = df_2023["avg_num_employees"] / df_2023[capacity_col]

id_col = "plant_name_ferc1"

# Determine technology column (fall back to a likely alternative)
tech_col = "technology_description" if "technology_description" in df_2023.columns else next((c for c in df_2023.columns if "techn" in c.lower()), None)
if tech_col is None:
    raise KeyError("No technology-description-like column found (expected 'technology_description')")

# Compute average employees_per_MW per plant
df_plants = (
    df_2023.dropna(subset=["employees_per_MW"]) 
    .groupby(id_col, as_index=False)
    .agg({"employees_per_MW": "mean", tech_col: "first"})
)

print(f"Computed employees_per_MW for {len(df_plants):,} plants (2023)")

# Compute average of per-plant employees_per_MW by technology
tech_avg = (
    df_plants.groupby(tech_col, as_index=False)["employees_per_MW"].mean()
    .rename(columns={"employees_per_MW": "avg_employees_per_MW"})
)

print("Average employees per MW by technology (top rows):")
tech_avg["avg_employees_per_100MW"] = tech_avg["avg_employees_per_MW"] * 100
print("Average employees per 100 MW by technology (top rows):")
print(tech_avg.sort_values("avg_employees_per_100MW", ascending=False).head(20).to_string(index=False))



