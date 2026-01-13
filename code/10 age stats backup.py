import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

# --- Chart Customization Parameters ---
font_size = 34
font_name = 'Cambria'

# 5) Check if the desired font is available, otherwise fall back to a default
try:
    # Attempt to find the font. If not found, it will raise a ValueError.
    fm.findfont(font_name, fallback_to_default=False)
    plt.rcParams['font.family'] = font_name
    print(f"Successfully set font to '{font_name}'.")
except ValueError:
    print(f"Font '{font_name}' not found. Falling back to the default sans-serif font.")
    plt.rcParams['font.family'] = 'sans-serif'

# 1. Define the project root directory
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd()))

print(f"Project root: {root}")

# 2. Define input and output file paths
input_path = os.path.join(root, 'intermediate data', 'cleaned_cost_dataV2.csv')
output_dir = os.path.join(root, 'output')
output_path = os.path.join(output_dir, 'age_stats.csv')
plot_output_path = os.path.join(output_dir, 'age_capacity_by_year.png')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Reading data from: {input_path}")
print(f"Will save table to: {output_path}")
print(f"Will save plot to: {plot_output_path}")

# 3. Read the data into a pandas DataFrame
try:
    df = pd.read_csv(input_path, low_memory=False)

    # 4. Drop rows with missing values in the relevant columns and create an explicit copy
    cols_to_check = ['installation_year', 'capacity_mw_x', 'technology_description']
    df_cleaned = df.dropna(subset=cols_to_check).copy()

    # Ensure data types are correct for calculation
    df_cleaned.loc[:, 'installation_year'] = pd.to_numeric(df_cleaned['installation_year'])
    df_cleaned.loc[:, 'capacity_mw_x'] = pd.to_numeric(df_cleaned['capacity_mw_x'])

    # 5. Calculate the weighted average installation year for each technology
    df_cleaned['weighted_year_numerator'] = df_cleaned['installation_year'] * df_cleaned['capacity_mw_x']
    grouped_sums = df_cleaned.groupby('technology_description').agg(
        numerator_sum=('weighted_year_numerator', 'sum'),
        denominator_sum=('capacity_mw_x', 'sum')
    )
    age_stats = (grouped_sums['numerator_sum'] / grouped_sums['denominator_sum']).to_frame(
        'weighted_avg_installation_year').reset_index()

    # 6. Save the resulting table to a CSV file
    age_stats.to_csv(output_path, index=False)

    print("\nSuccessfully calculated weighted averages and saved the results.")
    print("First 5 rows of the output:")
    print(age_stats.head())

    # 7. Create a plot to visualize the data
    print("\nGenerating plot...")
    # Reshape data for plotting: years as index, tech as columns, capacity as values
    plot_data = df_cleaned.groupby(['installation_year', 'technology_description'])['capacity_mw_x'].sum().unstack(
        fill_value=0)

    # 1) Update the y axis units to GW (divide MW by 1000)
    plot_data = plot_data / 1000

    # 4) Recolor the technology groups
    technology_color_map = {
        # Fossil Fuels
        'Conventional Steam Coal': '#3c3c3c',
        'Natural Gas Fired CC': '#CB4335',
        'Natural Gas Fired CT': '#D35400',
        'Other Natural Gas': '#E67E22',
        # Baseload / Other
        'Nuclear': '#6a0dad',
        'Other': '#a9a9a9',
        # Renewables
        'Onshore Wind Turbine': '#5D9732',
        'Solar Photovoltaic': '#FFC423',
        'Geothermal': '#8C564B',
        'Conventional Hydroelectric': '#457B9D',
    }
    # Create a list of colors in the order of the dataframe columns
    colors = [technology_color_map.get(tech, '#000000') for tech in plot_data.columns]

    # Create the stacked bar chart
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    # 3) Reduce space between bars
    plot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.9)

    # Customize the plot
    # 6) Remove the title from the chart
    # ax.set_title('Installed Capacity by Year and Technology', fontsize=18, fontweight='bold')
    ax.set_xlabel('Installation Year', fontsize=font_size)
    ax.set_ylabel('Total Capacity (GW)', fontsize=font_size)
    # 6) Remove the legend title
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=font_size)

    # Fix x-axis labels to prevent overlap
    tick_frequency = 5  # Show a label every 5 years
    years = plot_data.index.astype(int)
    tick_positions = np.arange(len(years))

    # Create labels: show the year if it's a multiple of the frequency, otherwise show an empty string
    tick_labels = [year if i % tick_frequency == 0 else '' for i, year in enumerate(years)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # Apply rotation here

    # 4) Set font size for tick labels
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    ax.grid(axis='x', which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # 8. Save the plot to a file
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {plot_output_path}")


except FileNotFoundError:
    print(f"Error: The file was not old_path at {input_path}")
except KeyError as e:
    print(f"Error: A required column was not old_path in the CSV file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

