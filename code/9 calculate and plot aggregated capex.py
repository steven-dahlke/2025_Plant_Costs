import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend to prevent GUI errors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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

# 4. Drop rows with missing values in the columns essential for aggregation
cols_to_check = ['capex_per_mw', 'capacity_mw_x', 'report_year', 'technology_description']
df_cleaned = df.dropna(subset=cols_to_check)
print(f"\nOriginal rows: {len(df)}, Rows after dropping NA: {len(df_cleaned)}")


# 5. Define a function to calculate the weighted average
def weighted_average(group):
    """Calculates the weighted average of capex_per_mw."""
    weights = group['capacity_mw_x']
    values = group['capex_per_mw']
    return np.average(values, weights=weights)


# 6. Group by year and technology, then apply the weighted average function
aggregated_capex = df_cleaned.groupby(['report_year', 'technology_description'])[
    ['capex_per_mw', 'capacity_mw_x']].apply(weighted_average).reset_index(
    name='weighted_avg_capex_per_mw')

# 7. Display the first few rows of the resulting aggregated data
print("\nAggregated Capex (Weighted Average):")
print(aggregated_capex.head())

# 8. Sort the data by technology description, then report year
aggregated_capex_sorted = aggregated_capex.sort_values(by=['technology_description', 'report_year'])
print("\nSorted Data Preview:")
print(aggregated_capex_sorted.head())

# 9. Define the output path
output_dir = os.path.join(root, "intermediate data")
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
output_path = os.path.join(output_dir, "aggregated_capex_year_tech.csv")

# 10. Save the sorted DataFrame to a CSV file
aggregated_capex_sorted.to_csv(output_path, index=False)
print(f"\nSorted data successfully saved to: {output_path}")

# 11. Calculate and export the average annual rate of change
print("\n--- Calculating and Exporting Average Annual Rate of Change in Capex ---")


def calculate_cagr(start_value, end_value, num_years):
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    if start_value is None or end_value is None or num_years <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / num_years) - 1


# Define technologies and their specific time periods
tech_periods = {
    'Coal': (1995, 2023),
    'Natural Gas Fired CC': (1995, 2023),
    'Natural Gas Fired CT': (1995, 2023),
    'Nuclear': (1995, 2023),
    'Onshore Wind Turbine': (2010, 2023),
    'Solar Photovoltaic': (2016, 2023)
}

# Define the output text file path
text_output_path = os.path.join(output_dir, "Capex_avg_change_by_tech.txt")

# Open the file to write the results
with open(text_output_path, 'w') as f:
    f.write("--- Average Annual Rate of Change in Capex ---\n")

    # Loop through each technology and calculate its rate of change
    for tech, (start_year, end_year) in tech_periods.items():
        tech_df = aggregated_capex_sorted[aggregated_capex_sorted['technology_description'] == tech]

        start_val_row = tech_df[tech_df['report_year'] == start_year]
        end_val_row = tech_df[tech_df['report_year'] == end_year]

        start_capex = start_val_row['weighted_avg_capex_per_mw'].iloc[0] if not start_val_row.empty else None
        end_capex = end_val_row['weighted_avg_capex_per_mw'].iloc[0] if not end_val_row.empty else None

        if start_capex is not None and end_capex is not None:
            num_years = end_year - start_year
            cagr = calculate_cagr(start_capex, end_capex, num_years)

            f.write(f"\nTechnology: {tech}\n")
            f.write(f"Period: {start_year}-{end_year}\n")
            f.write(f"  Start Capex ({start_year}): ${start_capex:,.2f}/MW\n")
            f.write(f"  End Capex ({end_year}): ${end_capex:,.2f}/MW\n")
            f.write(f"  Average Annual Rate of Change: {cagr:.2%}\n")
        else:
            f.write(f"\nCould not calculate rate of change for '{tech}' for the period {start_year}-{end_year}.\n")
            if start_capex is None:
                f.write(f"  Missing data for the start year: {start_year}\n")
            if end_capex is None:
                f.write(f"  Missing data for the end year: {end_year}\n")

print(f"Rate of change analysis successfully saved to: {text_output_path}")

# 12. Normalize data and create plots
print("\n--- Normalizing Data and Plotting Time Series ---")

# --- Set Font ---
plt.rcParams['font.family'] = 'Cambria'

# --- Universal Plotting Parameters ---
# Change these values to adjust the appearance of both plots
plot_font_size = 34
plot_line_width = 4
plot_marker_size = 10

# --- Color Scheme ---
color_map = {
    'Coal': '#3c3c3c',
    'Natural Gas Fired CC': '#A0522D',
    'Natural Gas Fired CT': '#FF8C00',
    'Nuclear': '#C8A2C8',
    'Onshore Wind Turbine': '#5D9732',
    'Solar Photovoltaic': '#FFC423'
}

# --- PLOT 1: CONVENTIONAL TECHNOLOGIES ---
conventional_techs = {
    'Coal': (1995, 2023),
    'Natural Gas Fired CC': (1995, 2023),
    'Natural Gas Fired CT': (1995, 2023),
    'Nuclear': (1995, 2023)
}

normalized_conventional_list = []
for tech, (start_year, _) in conventional_techs.items():
    tech_df = aggregated_capex_sorted[
        (aggregated_capex_sorted['technology_description'] == tech) &
        (aggregated_capex_sorted['report_year'] >= start_year)
        ].copy()
    start_value_row = tech_df[tech_df['report_year'] == start_year]
    if not start_value_row.empty:
        start_value = start_value_row['weighted_avg_capex_per_mw'].iloc[0]
        tech_df['normalized_capex'] = tech_df['weighted_avg_capex_per_mw'] / start_value
        normalized_conventional_list.append(tech_df)
    else:
        print(f"Warning: No conventional data for {tech} in {start_year}.")

if normalized_conventional_list:
    normalized_conventional_df = pd.concat(normalized_conventional_list)
    fig, ax = plt.subplots(figsize=(14, 8))
    for tech in normalized_conventional_df['technology_description'].unique():
        plot_df = normalized_conventional_df[normalized_conventional_df['technology_description'] == tech]
        ax.plot(plot_df['report_year'], plot_df['normalized_capex'], marker='o', linestyle='-',
                linewidth=plot_line_width, markersize=plot_marker_size, label=tech, color=color_map.get(tech))

    ax.set_xlabel('Year', fontsize=plot_font_size)
    ax.set_ylabel('Normalized Capex', fontsize=plot_font_size)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=plot_font_size - 2)
    ax.set_ylim(bottom=0.75)  # Set y-axis minimum for linear scale
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=plot_font_size - 2, frameon=False)
    plot_output_dir = os.path.join(root, "output")
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_path = os.path.join(plot_output_dir, "normalized_capex_conventional.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Conventional technologies plot saved to: {plot_path}")
    plt.close(fig)  # Close the figure to free up memory

# --- PLOT 2: RENEWABLE TECHNOLOGIES ---
renewable_techs = {
    'Onshore Wind Turbine': 2010,
    'Solar Photovoltaic': 2016
}
normalization_year_renewables = 2016

normalized_renewable_list = []
for tech, start_year in renewable_techs.items():
    tech_df = aggregated_capex_sorted[
        (aggregated_capex_sorted['technology_description'] == tech) &
        (aggregated_capex_sorted['report_year'] >= start_year)
        ].copy()

    base_value_row = tech_df[tech_df['report_year'] == normalization_year_renewables]
    if not base_value_row.empty:
        base_value = base_value_row['weighted_avg_capex_per_mw'].iloc[0]
        tech_df['normalized_capex'] = tech_df['weighted_avg_capex_per_mw'] / base_value
        normalized_renewable_list.append(tech_df)
    else:
        print(f"Warning: No renewable data for {tech} in normalization year {normalization_year_renewables}.")

if normalized_renewable_list:
    normalized_renewable_df = pd.concat(normalized_renewable_list)
    fig, ax = plt.subplots(figsize=(14, 8))
    for tech in normalized_conventional_df['technology_description'].unique():
        plot_df = normalized_conventional_df[normalized_conventional_df['technology_description'] == tech]
        ax.plot(plot_df['report_year'], plot_df['normalized_capex'], marker='o', linestyle='-',
                linewidth=plot_line_width, markersize=plot_marker_size, label=tech, color=color_map.get(tech))

    ax.set_xlabel('Year', fontsize=plot_font_size)
    ax.set_ylabel(f'Normalized Capex', fontsize=plot_font_size)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8)
    # Dynamically set the y-axis limit to remove white space
    min_y_val = normalized_renewable_df['normalized_capex'].min()
    ax.set_ylim(bottom=min_y_val * 0.95)
    ax.tick_params(axis='both', which='major', labelsize=plot_font_size - 2)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=plot_font_size - 2, frameon=False)
    plot_output_dir = os.path.join(root, "output")
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_path = os.path.join(plot_output_dir, "normalized_capex_renewables.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Renewable technologies plot saved to: {plot_path}")
    plt.close(fig)  # Close the figure to free up memory

# --- PLOT 3: ABSOLUTE CAPEX - COAL VS. COMBINED AVERAGE ---
print("\n--- Creating Plot: Absolute Capex - Coal vs. Combined Average (Gas & Nuclear) ---")

# 1. Isolate the technologies to be averaged
techs_to_average = ['Natural Gas Fired CC', 'Natural Gas Fired CT', 'Nuclear']
combined_df = df_cleaned[df_cleaned['technology_description'].isin(techs_to_average)].copy()

# 2. Calculate the annual weighted average capex for the combined group
combined_avg_capex = combined_df.groupby('report_year')[['capex_per_mw', 'capacity_mw_x']].apply(
    weighted_average).reset_index(name='weighted_avg_capex')

# 3. Get the absolute capex data for Coal from the previously aggregated dataframe
coal_abs_df = aggregated_capex_sorted[aggregated_capex_sorted['technology_description'] == 'Coal'].copy()

# 4. Filter data to the desired range (1995-2023)
plot_start_year = 1995
plot_end_year = 2023
coal_plot_df = coal_abs_df[
    (coal_abs_df['report_year'] >= plot_start_year) & (coal_abs_df['report_year'] <= plot_end_year)]
combined_plot_df = combined_avg_capex[
    (combined_avg_capex['report_year'] >= plot_start_year) & (combined_avg_capex['report_year'] <= plot_end_year)]

# 5. Create the plot
if not coal_plot_df.empty and not combined_plot_df.empty:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Add the individual tech series in the background
    for tech in techs_to_average:
        tech_df = aggregated_capex_sorted[
            (aggregated_capex_sorted['technology_description'] == tech) &
            (aggregated_capex_sorted['report_year'] >= plot_start_year) &
            (aggregated_capex_sorted['report_year'] <= plot_end_year)
            ]
        if not tech_df.empty:
            ax.plot(tech_df['report_year'], tech_df['weighted_avg_capex_per_mw'],
                    linestyle='--',
                    linewidth=plot_line_width - 1.5,
                    label=tech,
                    color=color_map.get(tech),
                    alpha=0.5)  # Make semi-transparent

    # Plot Coal series
    ax.plot(coal_plot_df['report_year'], coal_plot_df['weighted_avg_capex_per_mw'], marker='o', linestyle='-',
            linewidth=plot_line_width, markersize=plot_marker_size, label='Coal', color=color_map.get('Coal'))

    # Plot the combined average series
    ax.plot(combined_plot_df['report_year'], combined_plot_df['weighted_avg_capex'], marker='o', linestyle='-',
            linewidth=plot_line_width, markersize=plot_marker_size, label='Weighted Avg.',
            color='#4682B4')  # SteelBlue for contrast

    # 6. Style the plot
    ax.set_xlabel('Year', fontsize=plot_font_size)
    ax.set_ylabel('Capital Balance ($million/MW)', fontsize=plot_font_size)
    ax.tick_params(axis='both', which='major', labelsize=plot_font_size - 2)

    # Format the y-axis to display as currency in millions
    formatter = FuncFormatter(lambda y, _: f'${y / 1_000_000:.1f}')
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=plot_font_size - 2, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 7. Save the plot
    plot_output_dir = os.path.join(root, "output")
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_path = os.path.join(plot_output_dir, "absolute_capex_coal_vs_avg.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Absolute Capex plot saved to: {plot_path}")

    # --- CALCULATE AND PRINT CAPEX DIFFERENCES ---
    print("\n--- Capex Difference Analysis (Coal vs. Weighted Average) ---")

    # Get 1995 values
    capex_coal_1995 = coal_plot_df[coal_plot_df['report_year'] == 1995]['weighted_avg_capex_per_mw'].iloc[0]
    capex_avg_1995 = combined_plot_df[combined_plot_df['report_year'] == 1995]['weighted_avg_capex'].iloc[0]

    # Get 2023 values
    capex_coal_2023 = coal_plot_df[coal_plot_df['report_year'] == 2023]['weighted_avg_capex_per_mw'].iloc[0]
    capex_avg_2023 = combined_plot_df[combined_plot_df['report_year'] == 2023]['weighted_avg_capex'].iloc[0]

    # Print individual values first
    print(f"\nCapital Balance in 1995:")
    print(f"  - Coal: ${capex_coal_1995:,.2f}/MW")
    print(f"  - Weighted Avg. (Gas & Nuclear): ${capex_avg_1995:,.2f}/MW")

    print(f"\nCapital Balance in 2023:")
    print(f"  - Coal: ${capex_coal_2023:,.2f}/MW")
    print(f"  - Weighted Avg. (Gas & Nuclear): ${capex_avg_2023:,.2f}/MW")

    # Calculate and print differences
    diff_1995 = capex_coal_1995 - capex_avg_1995
    diff_2023 = capex_coal_2023 - capex_avg_2023
    change_in_diff = diff_2023 - diff_1995

    print(f"\nDifference (Coal - Avg.) in 1995: ${diff_1995:,.2f}/MW")
    print(f"Difference (Coal - Avg.) in 2023: ${diff_2023:,.2f}/MW")
    print(f"Change in the difference from 1995 to 2023: ${change_in_diff:,.2f}/MW")

    # Calculate average capacity of individual coal plants in the sample
    coal_plants_sample = df_cleaned[
        (df_cleaned['technology_description'] == 'Coal') &
        (df_cleaned['report_year'] >= plot_start_year) &
        (df_cleaned['report_year'] <= plot_end_year)
        ]
    # Use .mean() for a robust average of individual plant entries
    avg_coal_capacity = coal_plants_sample['capacity_mw_x'].sum()/9
    print(f"\nTotal capacity of individual coal plants during sample (1995-2023): {avg_coal_capacity:,.2f} MW")

    plt.close(fig)
else:
    print("Warning: Could not create Absolute Capex plot due to missing data for Coal or the combined average.")


