import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt


# --- Chart Customization Parameters ---
font_size = 34
font_name = 'Cambria'

# plt.rcParams['font.family'] = 'Cambria' # This is often unreliable

# Get a list of all font names known to matplotlib
font_names = sorted(set([f.name for f in fm.fontManager.ttflist]))

# Check if "Cambria" is in the list
print("Is 'Cambria' available?", "Cambria" in font_names)

# Get the path to the font file
# This is a more robust way to ensure the correct font is used
try:
    font_path = fm.findfont("Cambria")
    # --- Include the font size directly in the FontProperties object ---
    my_font = fm.FontProperties(fname=font_path, size=font_size)
    print(f"Successfully loaded Cambria font from: {font_path}")
except ValueError:
    print("Cambria font not found. Using default font.")
    # --- Also include size in the fallback font ---
    my_font = fm.FontProperties(size=font_size) # Use default font as a fallback


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
    cols_to_check = ['installation_year', 'capacity_mw_x', 'technology_description', 'report_year']
    df_cleaned = df.dropna(subset=cols_to_check).copy()

    # Ensure data types are correct for calculation
    df_cleaned.loc[:, 'installation_year'] = pd.to_numeric(df_cleaned['installation_year'])
    df_cleaned.loc[:, 'capacity_mw_x'] = pd.to_numeric(df_cleaned['capacity_mw_x'])
    df_cleaned.loc[:, 'report_year'] = pd.to_numeric(df_cleaned['report_year'])

    # --- NEW: Filter for plants existing in 2023 ---
    df_cleaned = df_cleaned[df_cleaned['report_year'] == 2023].copy()

    # --- Filter data to start from the year 1950 ---
    df_cleaned = df_cleaned[df_cleaned['installation_year'] >= 1950].copy()

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
        'Natural Gas Fired CT': '#E67E22',
        'Other Natural Gas': '#800000', # Changed from orange ('#E67E22') to maroon for better contrast
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
    # The fontproperties object now controls both font family and size
    # Add `labelpad` to create space between the label and the tick numbers
    ax.set_xlabel('Installation Year', fontproperties=my_font, labelpad=20)
    ax.set_ylabel('Total Capacity (GW)', fontproperties=my_font)

    # Use the 'prop' argument, which now contains size info, for the legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', prop=my_font)

    # Fix x-axis labels to prevent overlap
    tick_frequency = 5  # Show a label every 5 years
    years = plot_data.index.astype(int)
    tick_positions = np.arange(len(years))

    # Create labels: show the year if it's a multiple of the frequency, otherwise show an empty string
    tick_labels = [year if i % tick_frequency == 0 else '' for i, year in enumerate(years)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # Apply rotation here

    # The `labelsize` argument is no longer needed as plt.setp handles it below
    # ax.tick_params(axis='x', labelsize=font_size)
    # ax.tick_params(axis='y', labelsize=font_size)

    # Set the font properties for all tick labels on both axes.
    # This now correctly applies both the font family and size.
    plt.setp(ax.get_xticklabels(), fontproperties=my_font)
    plt.setp(ax.get_yticklabels(), fontproperties=my_font)

    ax.grid(axis='x', which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # 8. Save the plot to a file
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {plot_output_path}")


except FileNotFoundError:
    print(f"Error: The file was not found at {input_path}")
except KeyError as e:
    print(f"Error: A required column was not found in the CSV file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

