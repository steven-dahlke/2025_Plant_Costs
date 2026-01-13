import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend to prevent GUI errors
import matplotlib.pyplot as plt

# --- 1. Load the Data ---
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd()))

print(f"Project root: {root}")

# Construct the full path to the data file
# Assuming the script is in a 'scripts' folder and data is in 'intermediate data'
data_path = os.path.join(root, "intermediate data", "cleaned_cost_dataV2.csv")
print(f"Reading data from: {data_path}")

# Read the data into a pandas DataFrame
try:
    df = pd.read_csv(data_path, low_memory=False)
    print("CSV loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}")
    print(
        "Please ensure the 'intermediate data' folder with 'cleaned_cost_dataV2.csv' exists at the same level as your script folder.")
    exit()  # Exit the script if the file doesn't exist.

# --- 2. Data Preparation ---

# Convert 'report_year' to numeric, coercing errors to NaN
df['report_year'] = pd.to_numeric(df['report_year'], errors='coerce')

# Filter for the years 1995-2023
df_filtered = df[(df['report_year'] >= 1995) & (df['report_year'] <= 2023)].copy()

# Define the capex categories and technologies of interest
capex_cols = ["capex_land", "capex_structures", "capex_equipment"]
tech_descriptions = ["Coal", "Natural Gas Fired CC", "Natural Gas Fired CT", "Nuclear"]
cols_to_aggregate = capex_cols + ['capacity_mw_x']

# Filter for the specified technologies
df_filtered = df_filtered[df_filtered['technology_description'].isin(tech_descriptions)]

# Drop rows where any of the capex or capacity values are missing
df_filtered.dropna(subset=cols_to_aggregate, inplace=True)

# --- 3. Data Aggregation ---

# Group by technology and year, then sum the capex and capacity columns
# Using .agg() for explicit aggregation
df_agg = df_filtered.groupby(['technology_description', 'report_year'])[cols_to_aggregate].sum().reset_index()

print("\nAggregated Data Head:")
print(df_agg.head())

# --- 4. Calculate and Save Percentages for 1995 and 2023 ---
output_dir_base = os.path.join(root, 'output')
os.makedirs(output_dir_base, exist_ok=True)
percentages_filepath = os.path.join(output_dir_base, 'capex_percentages.txt')

print("\n--- Capex Percentage Breakdown for 1995 and 2023 ---")
print(f"Saving percentage breakdown to: {percentages_filepath}")

with open(percentages_filepath, 'w') as f:
    f.write("--- Capex Percentage Breakdown for 1995 and 2023 ---\n")
    # Filter the aggregated data for the two specific years
    df_percent_years = df_agg[df_agg['report_year'].isin([1995, 2023])]

    for tech in tech_descriptions:
        tech_header = f"\nTechnology: {tech}"
        print(tech_header)
        f.write(tech_header + "\n")

        for year in [1995, 2023]:
            # Get the data for the specific technology and year
            data = df_percent_years[
                (df_percent_years['technology_description'] == tech) & (df_percent_years['report_year'] == year)]

            if not data.empty:
                # Calculate the total capex for the year
                total_capex = data[capex_cols].sum(axis=1).iloc[0]

                if total_capex > 0:
                    # Calculate the percentage for each capex category
                    percent_land = (data['capex_land'].iloc[0] / total_capex) * 100
                    percent_structures = (data['capex_structures'].iloc[0] / total_capex) * 100
                    percent_equipment = (data['capex_equipment'].iloc[0] / total_capex) * 100

                    # Build the output string
                    output_block = (
                        f"  Year: {year}\n"
                        f"    - Capex Land: {percent_land:.2f}%\n"
                        f"    - Capex Structures: {percent_structures:.2f}%\n"
                        f"    - Capex Equipment: {percent_equipment:.2f}%"
                    )
                    print(output_block)
                    f.write(output_block + "\n")
                else:
                    output_line = f"  Year: {year} - Total Capex is zero, cannot calculate percentages."
                    print(output_line)
                    f.write(output_line + "\n")
            else:
                output_line = f"  Year: {year} - No data available."
                print(output_line)
                f.write(output_line + "\n")

# --- 4b. Calculate and Save Weighted Average Percentages ---
# First, calculate percentages for all years (needed for weighting)
df_agg_percent_all = df_agg.copy()
df_agg_percent_all['total_capex'] = df_agg_percent_all[capex_cols].sum(axis=1)

# Vectorized percentage calculation for better performance
for col in capex_cols:
    df_agg_percent_all[col] = np.where(
        df_agg_percent_all['total_capex'] > 0,
        (df_agg_percent_all[col] / df_agg_percent_all['total_capex']) * 100,
        0
    )

# Now calculate the weighted average across technologies for each year using a robust, vectorized approach to avoid apply()
# This resolves the DeprecationWarning.
# Create the product (numerator) columns for the weighted average calculation
df_agg_percent_all['weighted_land'] = df_agg_percent_all['capex_land'] * df_agg_percent_all['capacity_mw_x']
df_agg_percent_all['weighted_structures'] = df_agg_percent_all['capex_structures'] * df_agg_percent_all['capacity_mw_x']
df_agg_percent_all['weighted_equipment'] = df_agg_percent_all['capex_equipment'] * df_agg_percent_all['capacity_mw_x']

# Group by report_year and sum the weighted values and the weights themselves
df_sums = df_agg_percent_all.groupby('report_year', as_index=False).agg({
    'weighted_land': 'sum',
    'weighted_structures': 'sum',
    'weighted_equipment': 'sum',
    'capacity_mw_x': 'sum'  # This is the sum of weights (denominator)
})

# Calculate the final weighted average, handling division by zero
df_sums['weighted_avg_land'] = np.where(df_sums['capacity_mw_x'] > 0,
                                        df_sums['weighted_land'] / df_sums['capacity_mw_x'], 0)
df_sums['weighted_avg_structures'] = np.where(df_sums['capacity_mw_x'] > 0,
                                              df_sums['weighted_structures'] / df_sums['capacity_mw_x'], 0)
df_sums['weighted_avg_equipment'] = np.where(df_sums['capacity_mw_x'] > 0,
                                             df_sums['weighted_equipment'] / df_sums['capacity_mw_x'], 0)

# The resulting DataFrame has the weighted averages per year
df_weighted_avg = df_sums[['report_year', 'weighted_avg_land', 'weighted_avg_structures', 'weighted_avg_equipment']]

# Append the weighted average results to the text file
with open(percentages_filepath, 'a') as f:
    header = "\n\n--- Weighted Average Capex Percentages Across Technologies (1995 & 2023) ---\n"
    print(header)
    f.write(header)

    for year in [1995, 2023]:
        data = df_weighted_avg[df_weighted_avg['report_year'] == year]
        if not data.empty:
            output_block = (
                f"\nYear: {year}\n"
                f"  - Weighted Avg Land: {data['weighted_avg_land'].iloc[0]:.2f}%\n"
                f"  - Weighted Avg Structures: {data['weighted_avg_structures'].iloc[0]:.2f}%\n"
                f"  - Weighted Avg Equipment: {data['weighted_avg_equipment'].iloc[0]:.2f}%"
            )
            print(output_block)
            f.write(output_block + "\n")
        else:
            output_line = f"\nYear: {year} - No data available for weighted average calculation."
            print(output_line)
            f.write(output_line + "\n")

# --- 5. Chart Generation ---

# Create a directory to save the charts
output_dir_charts = os.path.join(output_dir_base, 'capex_stacked_area_charts')
os.makedirs(output_dir_charts, exist_ok=True)
print(f"\nCharts will be saved in: {output_dir_charts}")

# Define colors for the stacked areas
colors = ['#4c72b0', '#55a868', '#c44e52']
labels = ['Land', 'Structures', 'Equipment']

# Loop through each technology and create a plot
for tech in tech_descriptions:
    # Filter the aggregated data for the current technology
    tech_data = df_agg[df_agg['technology_description'] == tech]

    if tech_data.empty:
        print(f"\nNo data available for {tech} after filtering and aggregation. Skipping chart.")
        continue

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get the years and the capex values
    years = tech_data['report_year']
    capex_values = [tech_data['capex_land'], tech_data['capex_structures'], tech_data['capex_equipment']]

    # Create the stacked area chart
    ax.stackplot(years, capex_values, labels=labels, colors=colors, alpha=0.8)

    # --- 6. Chart Customization ---
    ax.set_title(f'Capital Expenditure Categories for {tech} (1995-2023)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Capital Expenditure ($)', fontsize=12)

    # Format y-axis to be more readable (e.g., in millions)
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Set x-axis to show integer years and rotate labels for clarity
    ax.set_xticks(np.arange(min(years), max(years) + 1, 2))
    plt.xticks(rotation=45)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Adjust layout and save the figure
    plt.tight_layout()
    # Sanitize filename
    safe_filename = tech.replace(" ", "_").replace("/", "_")
    fig_path = os.path.join(output_dir_charts, f'{safe_filename}_capex_chart.png')
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Chart for {tech} saved to {fig_path}")

# --- 7. Chart Generation (Percentages) ---

# --- 7a. Prepare data for percentage charts ---
df_agg_percent = df_agg.copy()
# Calculate total capex for each row to use as a denominator
df_agg_percent['total_capex'] = df_agg_percent[capex_cols].sum(axis=1)

# Calculate the percentage for each category, handling division by zero
for col in capex_cols:
    df_agg_percent[col] = df_agg_percent.apply(
        lambda row: (row[col] / row['total_capex']) * 100 if row['total_capex'] > 0 else 0,
        axis=1
    )

# --- 7b. Generate and save percentage charts ---
output_dir_charts_percent = os.path.join(output_dir_base, 'capex_stacked_area_charts_percentage')
os.makedirs(output_dir_charts_percent, exist_ok=True)
print(f"\nPercentage charts will be saved in: {output_dir_charts_percent}")

for tech in tech_descriptions:
    # Filter the percentage data for the current technology
    tech_data_percent = df_agg_percent[df_agg_percent['technology_description'] == tech]

    if tech_data_percent.empty:
        print(f"\nNo data available for {tech} to generate percentage chart. Skipping.")
        continue

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get the years and the capex percentage values
    years = tech_data_percent['report_year']
    capex_percent_values = [tech_data_percent['capex_land'], tech_data_percent['capex_structures'],
                            tech_data_percent['capex_equipment']]

    # Create the stacked area chart
    ax.stackplot(years, capex_percent_values, labels=labels, colors=colors, alpha=0.8)

    # --- 7c. Chart Customization (Percentages) ---
    ax.set_title(f'Capital Expenditure Percentage for {tech} (1995-2023)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)

    # Format y-axis to show percentages
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    ax.set_ylim(0, 100)  # Percentages should be on a 0-100 scale

    # Set x-axis to show integer years and rotate labels for clarity
    min_year, max_year = min(years), max(years)
    ax.set_xticks(np.arange(min_year, max_year + 1, 2))
    plt.xticks(rotation=45)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Adjust layout and save the figure
    plt.tight_layout()
    safe_filename = tech.replace(" ", "_").replace("/", "_")
    fig_path = os.path.join(output_dir_charts_percent, f'{safe_filename}_capex_percentage_chart.png')
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Percentage chart for {tech} saved to {fig_path}")

