import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re

# 1. Define the project root directory
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Construct the full path to the data file
data_path = os.path.join(root, "intermediate data", "cleaned_cost_dataV2.csv")
print(f"Reading data from: {data_path}")

# 3. Read the data into a pandas DataFrame
df = pd.read_csv(data_path, low_memory=False)
print(f"Initial data shape: {df.shape}")


# 4. Filter data for the model
# -----------------------------------------------------------------------------

# Define the technologies we want to include in the model
technologies_to_keep = [
    "Coal",
    "Natural Gas Fired CC",
    "Natural Gas Fired CT",
    "Nuclear"
]

# Filter the DataFrame to only include rows with the specified technologies
df_filtered = df[df['technology_description'].isin(technologies_to_keep)].copy()

print(f"Shape after filtering for technologies: {df_filtered.shape}")

# Handle duplicate plant entries by adding installation year to plant names
# -----------------------------------------------------------------------------
print("\nHandling duplicate plant entries - Round 1: Adding installation years...")

# Create masks for first round
duplicates_mask = df_filtered.duplicated(subset=['plant_name_ferc1', 'report_year'], keep=False)
valid_years_mask = df_filtered['installation_year'].notna()
combined_mask = duplicates_mask & valid_years_mask

# Add installation year to plant names
df_filtered.loc[combined_mask, 'plant_name_ferc1'] = (
    df_filtered.loc[combined_mask, 'plant_name_ferc1'] + '_' + 
    df_filtered.loc[combined_mask, 'installation_year'].astype(int).astype(str)
)

# Second round: Handle remaining duplicates by adding utility ID
print("\nHandling duplicate plant entries - Round 2: Adding utility IDs...")

# Create new mask for remaining duplicates after first round
remaining_duplicates_mask = df_filtered.duplicated(subset=['plant_name_ferc1', 'report_year'], keep=False)

# Add utility ID to plant names for remaining duplicates
df_filtered.loc[remaining_duplicates_mask, 'plant_name_ferc1'] = (
    df_filtered.loc[remaining_duplicates_mask, 'plant_name_ferc1'] + '_' + 
    df_filtered.loc[remaining_duplicates_mask, 'utility_id_ferc1'].astype(str)
)

# Third round: Handle remaining duplicates by adding capacity
print("\nHandling duplicate entries - Round 3: Adding capacity values...")

# Create new mask for remaining duplicates after second round
final_duplicates_mask = df_filtered.duplicated(subset=['plant_name_ferc1', 'report_year'], keep=False)

# Add capacity to plant names for remaining duplicates
df_filtered.loc[final_duplicates_mask, 'plant_name_ferc1'] = (
    df_filtered.loc[final_duplicates_mask, 'plant_name_ferc1'] + '_' + 
    df_filtered.loc[final_duplicates_mask, 'capacity_mw_x'].astype(int).astype(str) + 
    'MW'
)

# Sort the DataFrame by plant name and report year
df_filtered = df_filtered.sort_values(['plant_name_ferc1', 'report_year'])

# Print information about the changes
num_modified_first = combined_mask.sum()
num_skipped = duplicates_mask.sum() - num_modified_first
num_modified_second = remaining_duplicates_mask.sum()
num_modified_third = final_duplicates_mask.sum()

print(f"Round 1: Modified {num_modified_first} plant names with installation years")
print(f"Skipped {num_skipped} entries due to missing installation years")
print(f"Round 2: Modified {num_modified_second} plant names with utility IDs")
print(f"Round 3: Modified {num_modified_third} plant names with capacity values")
print("\nExample of modified plant names:")
if num_modified_third > 0:
    # Create a safer way to show examples by using loc[] with the boolean mask
    examples = df_filtered.loc[final_duplicates_mask, ['plant_name_ferc1', 'report_year', 'installation_year', 'capacity_mw_x']].head()
    print(examples)

# Filter plants based on minimum number of entries
# -----------------------------------------------------------------------------
print("\nFiltering plants with insufficient time series data...")

# Count entries per plant
plant_counts = df_filtered['plant_name_ferc1'].value_counts()

# Get plants with at least 10 entries
valid_plants = plant_counts[plant_counts >= 10].index

# Filter the DataFrame to keep only those plants
original_shape = df_filtered.shape
df_filtered = df_filtered[df_filtered['plant_name_ferc1'].isin(valid_plants)]

# Print summary of the filtering
plants_removed = len(plant_counts) - len(valid_plants)
rows_removed = original_shape[0] - df_filtered.shape[0]
print(f"Removed {plants_removed} plants with fewer than 10 entries")
print(f"Removed {rows_removed} total rows")
print(f"Shape after filtering: {df_filtered.shape}")

# Validation check for remaining duplicates
# -----------------------------------------------------------------------------
print("\nChecking for any remaining duplicates...")

# Find any remaining duplicates
final_duplicates = df_filtered[df_filtered.duplicated(subset=['plant_name_ferc1', 'report_year'], keep=False)]

if len(final_duplicates) > 0:
    print("ERROR: Found remaining duplicates after all cleaning steps!")
    print("\nDuplicate entries found for:")
    
    # Group by plant name and report year to show the problematic cases
    duplicate_summary = final_duplicates.groupby(['plant_name_ferc1', 'report_year']).agg({
        'installation_year': 'count',
        'utility_id_ferc1': lambda x: ', '.join(map(str, set(x)))
    }).reset_index()
    
    print(duplicate_summary)
    
    # Stop the code execution
    raise ValueError("Duplicate entries found. Please investigate the plants listed above.")

print("Validation passed: No remaining duplicates found.")

# Filter out rows where capex or capacity are zero or non-positive
pre_filter_shape = df_filtered.shape
df_filtered = df_filtered[df_filtered['capex_total'] > 0]
df_filtered = df_filtered[df_filtered['capacity_mw_x'] > 0]
print(f"Shape after removing non-positive capex/capacity rows: {df_filtered.shape}")
print(f"Number of rows removed: {pre_filter_shape[0] - df_filtered.shape[0]}")


# Define the directory path for the output files
output_dir = os.path.join(root, "intermediate data")

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define the full path for the output file
output_path = os.path.join(output_dir, "cleaned_cost_dataV3.csv")

# Export the DataFrame to CSV
df_filtered.to_csv(output_path, index=False)



# Define the columns that are essential for the model
# -----------------------------------------------------------------------------

# Define the columns that are essential for the model
required_columns = [
    "installation_year",
    "capacity_mw_x",
    "capex_total"
]

# Remove rows where any of the required columns have missing data and create an explicit copy
df_filtered = df_filtered.dropna(subset=required_columns).copy()

print(f"Shape after removing rows with missing essential data: {df_filtered.shape}")


# # Create plots of capex_total over time for each plant
# # -----------------------------------------------------------------------------
# print("\nGenerating capex time series plots for each plant...")


# def safe_filename(name: str, replacement: str = "_", maxlen: int = 150) -> str:
#     # Replace characters invalid on Windows: <>:"/\|?* and control chars
#     s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, str(name))
#     s = s.strip().strip(".")  # no trailing spaces/dots
#     s = re.sub(rf"{re.escape(replacement)}+", replacement, s)  # collapse repeats
#     return s[:maxlen]

# plot_dir = os.path.join(root, "intermediate data", "capex_econometric", "capex_total_plots_before_cleaning")
# os.makedirs(plot_dir, exist_ok=True)

# plant_names = df_filtered['plant_name_ferc1'].unique()

# # Track names to avoid accidental collisions after sanitizing
# name_counts = {}

# for plant in plant_names:
#     plant_data = df_filtered[df_filtered['plant_name_ferc1'] == plant]

#     plt.figure(figsize=(10, 6))
#     plt.plot(plant_data['report_year'], plant_data['capex_total'], 'b-o')
#     plt.title(f"Capex Time Series: {plant}")
#     plt.xlabel("Report Year")
#     plt.ylabel("Capex Total ($)")
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     base = safe_filename(plant)
#     filename = f"{base}.png"
#     # Ensure uniqueness if different plants sanitize to the same name
#     if filename in name_counts:
#         name_counts[filename] += 1
#         filename = f"{base}_{name_counts[filename]}.png"
#     else:
#         name_counts[filename] = 0

#     plot_path = os.path.join(plot_dir, filename)
#     plt.savefig(plot_path)
#     plt.close()

# print(f"Generated plots saved to: {plot_dir}")



# Helper: interpolate capex for a specific plant or span of years
# -----------------------------------------------------------------------------
def interpolate_capex_for_year(
    dataframe: pd.DataFrame,
    plant_name: str,
    first_year: int,
    last_year: int | None = None
) -> pd.DataFrame:
    """
    Interpolate capex_total for a plant.

    Modes
    -----
    1. Single-year mode (last_year is None or == first_year):
       Replace that year's capex_total with the average of the nearest
       previous and next non-null capex_total values.
    2. Span mode (last_year provided and > first_year):
       Linear interpolation between capex_total in first_year and last_year,
       overwriting capex_total for existing intermediate years only.

    Parameters
    ----------
    dataframe : DataFrame with columns: plant_name_ferc1, report_year, capex_total
    plant_name : str
    first_year : int
    last_year : int | None

    Returns
    -------
    DataFrame (copy) with modifications.
    """
    required = {'plant_name_ferc1', 'report_year', 'capex_total'}
    if not required.issubset(dataframe.columns):
        raise KeyError(f"DataFrame must contain columns: {required}")

    if last_year is None:
        last_year = first_year

    if first_year > last_year:
        raise ValueError("first_year must be <= last_year")

    df_out = dataframe.copy()
    plant_mask = df_out['plant_name_ferc1'] == plant_name
    plant_df = df_out[plant_mask]

    if plant_df.empty:
        print(f"No rows found for plant '{plant_name}'; nothing to interpolate.")
        return df_out

    # Single-year mode
    if first_year == last_year:
        target_year = first_year
        mask_target = plant_mask & (df_out['report_year'] == target_year)
        if not mask_target.any():
            print(f"No rows for plant '{plant_name}' in {target_year}; skipping.")
            return df_out

        nonnull = plant_df.dropna(subset=['capex_total'])
        prev_candidates = nonnull[nonnull['report_year'] < target_year]
        next_candidates = nonnull[nonnull['report_year'] > target_year]

        if prev_candidates.empty or next_candidates.empty:
            print(
                f"Cannot interpolate single year {target_year} for '{plant_name}': "
                f"missing {'previous' if prev_candidates.empty else ''}"
                f"{' and ' if prev_candidates.empty and next_candidates.empty else ''}"
                f"{'next' if next_candidates.empty else ''} neighbor."
            )
            return df_out

        prev_year = prev_candidates['report_year'].max()
        next_year = next_candidates['report_year'].min()
        capex_prev = prev_candidates.loc[prev_candidates['report_year'] == prev_year, 'capex_total'].mean()
        capex_next = next_candidates.loc[next_candidates['report_year'] == next_year, 'capex_total'].mean()

        if pd.isna(capex_prev) or pd.isna(capex_next):
            print(f"Neighbor capex NaN; skipping interpolation for '{plant_name}' {target_year}.")
            return df_out

        new_val = float((capex_prev + capex_next) / 2.0)
        old_vals = df_out.loc[mask_target, 'capex_total'].values
        df_out.loc[mask_target, 'capex_total'] = new_val
        print(
            f"Interpolated '{plant_name}' {target_year} using {prev_year} ({capex_prev:.2f}) "
            f"and {next_year} ({capex_next:.2f}) -> {new_val:.2f}. "
            f"Updated {mask_target.sum()} row(s); old values: {old_vals}."
        )
        return df_out

    # Span mode
    mask_first = plant_mask & (df_out['report_year'] == first_year)
    mask_last = plant_mask & (df_out['report_year'] == last_year)

    if not mask_first.any() or not mask_last.any():
        print(
            f"Cannot span-interpolate '{plant_name}' {first_year}-{last_year}: "
            f"missing {'first_year' if not mask_first.any() else ''}"
            f"{' and ' if (not mask_first.any() and not mask_last.any()) else ''}"
            f"{'last_year' if not mask_last.any() else ''} anchor row(s)."
        )
        return df_out

    capex_first = df_out.loc[mask_first, 'capex_total'].mean()
    capex_last = df_out.loc[mask_last, 'capex_total'].mean()

    if pd.isna(capex_first) or pd.isna(capex_last):
        print(
            f"Anchor capex NaN (first={capex_first}, last={capex_last}); "
            f"skipping interpolation for '{plant_name}' {first_year}-{last_year}."
        )
        return df_out

    total_years = last_year - first_year
    intermediate_years = list(range(first_year + 1, last_year))
    if not intermediate_years:
        print(f"No intermediate years between {first_year} and {last_year}; nothing to do.")
        return df_out

    updates = 0
    skipped_missing = []
    for y in intermediate_years:
        mask_y = plant_mask & (df_out['report_year'] == y)
        if not mask_y.any():
            skipped_missing.append(y)
            continue
        frac = (y - first_year) / total_years
        interp_val = float(capex_first + frac * (capex_last - capex_first))
        old_vals = df_out.loc[mask_y, 'capex_total'].values
        df_out.loc[mask_y, 'capex_total'] = interp_val
        updates += mask_y.sum()
        print(
            f"Span interpolation '{plant_name}' {y}: "
            f"{capex_first:.2f} + {frac:.3f} * ({capex_last:.2f} - {capex_first:.2f}) = {interp_val:.2f}. "
            f"Updated {mask_y.sum()} row(s); old values: {old_vals}."
        )

    print(
        f"Completed span interpolation for '{plant_name}' {first_year}-{last_year}. "
        f"Anchors kept (first={capex_first:.2f}, last={capex_last:.2f}). Updated {updates} row(s)."
        + (f" Missing years (no rows): {skipped_missing}" if skipped_missing else "")
    )
    return df_out

# Helper: remove specific year(s) for a plant
# -----------------------------------------------------------------------------
def remove_years(
    dataframe: pd.DataFrame,
    plant_name: str,
    years: int | list | tuple | set
) -> pd.DataFrame:
    """
    Remove rows for a given plant and one or more report years.

    Parameters
    ----------
    dataframe : DataFrame (must have plant_name_ferc1, report_year)
    plant_name : str
    years : int | iterable of int

    Returns
    -------
    DataFrame (copy) with rows removed.
    """
    if 'plant_name_ferc1' not in dataframe.columns or 'report_year' not in dataframe.columns:
        raise KeyError("DataFrame must contain 'plant_name_ferc1' and 'report_year' columns")

    if isinstance(years, (int, np.integer)):
        years_set = {int(years)}
    else:
        years_set = {int(y) for y in years}

    df_out = dataframe.copy()
    plant_mask = df_out['plant_name_ferc1'] == plant_name
    if not plant_mask.any():
        print(f"No rows for plant '{plant_name}'; nothing removed.")
        return df_out

    before = df_out.shape[0]
    remove_mask = plant_mask & df_out['report_year'].isin(years_set)
    found_years = set(df_out.loc[remove_mask, 'report_year'].unique())
    missing_years = sorted(years_set - found_years)

    df_out = df_out.loc[~remove_mask].copy()
    removed = before - df_out.shape[0]

    print(
        f"Removed {removed} row(s) for '{plant_name}' in years {sorted(found_years)}."
        + (f" Missing years (no rows): {missing_years}" if missing_years else "")
    )
    return df_out


df_filtered = interpolate_capex_for_year(df_filtered, "59th st gt-1", 2019)
df_filtered = remove_years(df_filtered, "a.b. brown station", [1994,2000, 2023])
df_filtered = remove_years(df_filtered, "afton turbine", [2019,2020,2021,2022,2023])
df_filtered = remove_years(df_filtered, "bartow_1963", [2009])
df_filtered = interpolate_capex_for_year(df_filtered, "belle river (total)", 2007)
df_filtered = interpolate_capex_for_year(df_filtered, "belle river (total)", 2011)
df_filtered = remove_years(df_filtered, "beluga", [2001,2002])
df_filtered = interpolate_capex_for_year(df_filtered, "beluga", 2011)
df_filtered = remove_years(df_filtered, "ben french station_1960", [2014])
df_filtered = remove_years(df_filtered, "boardman_1980_216", [2020])
df_filtered = interpolate_capex_for_year(df_filtered, "bonanza (dgt share)", 2009)
df_filtered = remove_years(df_filtered, "bonanza (dgt share)", [2005])
df_filtered = remove_years(df_filtered, "cape canaveral", [2010])
df_filtered = remove_years(df_filtered, "ceredo", [2005,2006])
df_filtered = interpolate_capex_for_year(df_filtered, "daniel_1981_296", 2019)
df_filtered = remove_years(df_filtered, "demoss petrie", [1998])
df_filtered = remove_years(df_filtered, "edwardsport", [2011])
df_filtered = interpolate_capex_for_year(df_filtered, "energy center", 2007)
df_filtered = remove_years(df_filtered, "four corners (1)", [2019,2020,2021,2022,2023])
df_filtered = remove_years(df_filtered, "gallagher", [2021,2022,2023])
df_filtered = interpolate_capex_for_year(df_filtered, "hutchinson_1965", 2003)
df_filtered = interpolate_capex_for_year(df_filtered, "jeffrey 20%", 1999)
df_filtered = remove_years(df_filtered, "karn 1 & 2", [2023])
df_filtered = remove_years(df_filtered, "lauderdale_1993", [2016,2017,2018])
df_filtered = remove_years(df_filtered, "lee_1978", [2007])
df_filtered = remove_years(df_filtered, "lordsburg turbine", [2019,2020,2021,2022,2023])
df_filtered = remove_years(df_filtered, "luna_2006_263", [2019,2020,2021,2022,2023])
df_filtered = remove_years(df_filtered, "lv generation", [2014])
df_filtered = remove_years(df_filtered, "meramec", [2022,2023])
df_filtered = remove_years(df_filtered, "miami fort 6", [2015,2016,2019])
df_filtered = remove_years(df_filtered, "mill creek", [2021,2022,2023])
df_filtered = remove_years(df_filtered, "mint farm", [2008])
df_filtered = remove_years(df_filtered, "monroe_1968", [2011,2012,2013,2014,2015])
df_filtered = interpolate_capex_for_year(df_filtered, "monroe_1969", 2003)
df_filtered = remove_years(df_filtered, "montrose", [2018,2019])
df_filtered = remove_years(df_filtered, "osawatomie", [2003,2004])
df_filtered = remove_years(df_filtered, "palo verde (1)", [2019,2020,2021,2022,2023])
df_filtered = interpolate_capex_for_year(df_filtered, "pea ridge", 2019)
df_filtered = interpolate_capex_for_year(df_filtered, "pleasant hill", 1995)
df_filtered = remove_years(df_filtered, "port everglades_1965", [2013])
df_filtered = remove_years(df_filtered, "rahtdrum", [1999,2000,2001,2002,2003,2004])
df_filtered = remove_years(df_filtered, "rio bravo", [2019,2020,2021,2022,2023])
df_filtered = interpolate_capex_for_year(df_filtered, "river bend", 2008)
df_filtered = remove_years(df_filtered, "river bend_1954", [2020])
df_filtered = remove_years(df_filtered, "river bend_1969", [2012,2013])
df_filtered = interpolate_capex_for_year(df_filtered, "riverside 3 & 4", 2019)
df_filtered = remove_years(df_filtered, "riverside_1956", 2005)
df_filtered = interpolate_capex_for_year(df_filtered, "sheboygan falls", 2012)
df_filtered = remove_years(df_filtered, "sibley6", [2018,2019])
df_filtered = remove_years(df_filtered, "sun peak 3, 4, 5", [1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014])
df_filtered = remove_years(df_filtered, "taconite harbor", [2023])
df_filtered = interpolate_capex_for_year(df_filtered, "urquhart #2 peaking", 1996,2002)
df_filtered = remove_years(df_filtered, "wabash river_1968", [2015,2016,2017])
df_filtered = remove_years(df_filtered, "wabash river_1968", [2003,2004])
df_filtered = interpolate_capex_for_year(df_filtered, "whitehorn", 2009,2012)



# 5. Create variables for the econometric model
# ----------------------------------------------------------------------------

df_cleaned = df_filtered.copy()

# Create the 'Age' variable
df_cleaned['age'] = df_cleaned['report_year'] - df_cleaned['installation_year']

# Create a dummy variable for plants 20 years or older
df_cleaned['is_old_plant'] = (df_cleaned['installation_year'] < 2000).astype(int)
print("Created new column: 'is_old_plant'")

# Create a dummy variable for Coal technology
df_cleaned['is_coal'] = (df_cleaned['technology_description'] == 'Coal').astype(int)
print("Created new column: 'is_coal'")

# Updated: Coal plants with capacity >= 25 MW installed on or before 2000
df_cleaned['is_coal_ge25mw_pre2001'] = (
    (df_cleaned['technology_description'] == 'Coal') &
    (df_cleaned['capacity_mw_x'] >= 25) &
    (df_cleaned['installation_year'] <= 2000)
).astype(int)
print("Created new column: 'is_coal_ge25mw_pre2001' (legacy coal ≥25 MW, installed ≤2000)")

# Define compliance period

df_cleaned['is_compliance_period'] = (
    (df_cleaned['report_year'] >= 2007) & (df_cleaned['report_year'] <= 2018)
).astype(int)
print("Created: 'is_compliance_period' (2007-18)")

# Create the log-transformed variables for the model
df_model_ready = df_cleaned.copy()
df_model_ready['ln_capital_balance'] = np.log(df_model_ready['capex_total'])
df_model_ready['ln_capacity'] = np.log(df_model_ready['capacity_mw_x'])

print("Created new columns: 'age', 'is_old_plant', 'ln_capital_balance', 'ln_capacity'")
print(
    df_model_ready[
        ['technology_description', 'report_year', 'age', 'is_old_plant', 'is_coal', 'is_coal_ge25mw_pre2001', 'ln_capital_balance', 'ln_capacity']
    ].head()
)




# 6. Export the final DataFrame to a CSV file
# -----------------------------------------------------------------------------

# Define the directory path for the output files
output_dir = os.path.join(root, "intermediate data", "capex_econometric")

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define the full path for the output file
output_path = os.path.join(output_dir, "df_model_ready.csv")

# Export the DataFrame to CSV
df_model_ready.to_csv(output_path, index=False)

print(f"\nSuccessfully exported the model-ready data to: {output_path}")

