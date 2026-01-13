import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# Define the project root directory
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(f"Project root: {root}")

# Define the path to the data file
path = os.path.join(root, 'data', 'cost_data_2023.csv')
df_2023 = pd.read_csv(path, low_memory=False)


# --- Filter out entries with zero or missing capacity before analysis ---
print(f"\nOriginal number of plant entries in 2023: {len(df_2023)}")

# Keep only rows where capacity is greater than zero
df_2023 = df_2023[df_2023['capacity_mw_x'] > 0].copy()

rows_after_filter = len(df_2023)
print(f"Number of entries after removing those with zero capacity: {rows_after_filter}")

# Overwrite the original CSV file with the cleaned DataFrame
print(f"Overwriting 'cost_data_2023.csv' with {rows_after_filter} cleaned entries...")
df_2023.to_csv(path, index=False)
print("Export complete.")


# --- Table 1: Composition of FERC Form 1 Sample by Technology Type (2023) ---
print("\nGenerating Table 1: Composition of FERC Form 1 Sample by Technology Type (2023)...")

# Group by technology and perform aggregations using the corrected 'capacity_mw_x' column
table1_df = df_2023.groupby('technology_description').agg(
    number_of_plants=('technology_description', 'size'),
    total_capacity_mw=('capacity_mw_x', 'sum')
).reset_index()

# Calculate the total capacity for the entire 2023 sample
total_sample_capacity = table1_df['total_capacity_mw'].sum()

# Calculate each technology's percentage of the total capacity
table1_df['percent_of_total_capacity'] = (table1_df['total_capacity_mw'] / total_sample_capacity) * 100

# Sort the table by total capacity in descending order for better readability
table1_df = table1_df.sort_values(by='total_capacity_mw', ascending=False)

# Rename the columns to be more descriptive for the final table
table1_df.rename(columns={
    'technology_description': 'Technology Group',
    'number_of_plants': 'Number of Unique Plants',
    'total_capacity_mw': 'Total Capacity (MW)',
    'percent_of_total_capacity': "% of Sample's Total Capacity"
}, inplace=True)


# --- Display and Export the Final Table ---

# Create a display version with formatting for better console output
display_df = table1_df.copy()
display_df['Total Capacity (MW)'] = display_df['Total Capacity (MW)'].map('{:,.0f}'.format)
display_df["% of Sample's Total Capacity"] = display_df["% of Sample's Total Capacity"].map('{:.2f}%'.format)


# Print the formatted table to the console
print("\n" + "="*80)
print("Table 1: Composition of FERC Form 1 Sample by Technology Type (2023)")
print("="*80)
print(display_df.to_string(index=False))
print("="*80)


# Export the raw numerical data to a CSV file
table1_output_path = os.path.join(root, 'output', 'table_1_composition_by_technology_2023.csv')
table1_df.to_csv(table1_output_path, index=False, float_format='%.2f')

print(f"\nSuccessfully exported the data for Table 1 to: {table1_output_path}")


# --- Table 2: Descriptive Statistics for Plant Capacity (MW) by Technology (2023) ---
print("\nGenerating Table 2: Descriptive Statistics for Plant Capacity by Technology...")

# The Interquartile Range (IQR) is the difference between the 75th and 25th percentile.
# We can define a function to calculate it during the aggregation.
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

# Group by technology and calculate the specified descriptive statistics for 'capacity_mw_x'
table2_df = df_2023.groupby('technology_description').agg(
    mean_capacity_mw=('capacity_mw_x', 'mean'),
    median_capacity_mw=('capacity_mw_x', 'median'),
    min_capacity_mw=('capacity_mw_x', 'min'),
    max_capacity_mw=('capacity_mw_x', 'max'),
    std_dev_capacity_mw=('capacity_mw_x', 'std'),
    iqr_capacity_mw=('capacity_mw_x', iqr)
).reset_index()

# Sort the table by mean capacity in descending order for better presentation
table2_df = table2_df.sort_values(by='mean_capacity_mw', ascending=False)

# Rename the columns to match the desired output
table2_df.rename(columns={
    'technology_description': 'Technology Group',
    'mean_capacity_mw': 'Mean Capacity (MW)',
    'median_capacity_mw': 'Median Capacity (MW)',
    'min_capacity_mw': 'Min Capacity (MW)',
    'max_capacity_mw': 'Max Capacity (MW)',
    'std_dev_capacity_mw': 'Std. Dev. Capacity (MW)',
    'iqr_capacity_mw': 'IQR Capacity (MW)'
}, inplace=True)

# --- Display and Export the Final Table ---

# Create a display version with formatting for console output
# We'll format the numbers to one decimal place with commas for readability
display_df2 = table2_df.copy()
for col in display_df2.columns:
    if col != 'Technology Group':
        display_df2[col] = display_df2[col].map('{:,.1f}'.format)

# Print the formatted table to the console
print("\n" + "="*120)
print("Table 2: Descriptive Statistics for Plant Capacity (MW) by Technology (2023)")
print("="*120)
print(display_df2.to_string(index=False))
print("="*120)

# Export the raw numerical data to a CSV file
table2_output_path = os.path.join(root, 'output', 'table_2_capacity_stats_2023.csv')
table2_df.to_csv(table2_output_path, index=False, float_format='%.2f')

print(f"\nSuccessfully exported the data for Table 2 to: {table2_output_path}")

# --- Figure 1: Box Plot of Plant Capacity (MW) by Technology (2023) ---
print("\nGenerating Figure 1: Box plot of plant capacity distribution...")

# Import the necessary formatters and FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import FontProperties

# --- Define direct paths to the REGULAR and BOLD font files ---
# Regular font for tick labels
font_path_regular = 'C:/Windows/Fonts/cambria.ttc'
custom_font = FontProperties(fname=font_path_regular)

# Bold font for axis titles
font_path_bold = 'C:/Windows/Fonts/cambriab.ttf'
bold_font = FontProperties(fname=font_path_bold)


# Define shorter names for plot labels
label_rename_map = {
    'Natural Gas Fired Combined Cycle': 'Natural Gas CC',
    'Natural Gas Fired Combustion Turbine': 'Natural Gas CT',
    'Conventional Steam Coal': 'Coal',
    'Onshore Wind Turbine':'Onshore Wind'
}
df_2023['technology_description'] = df_2023['technology_description'].replace(label_rename_map)

# Define the original custom color map
technology_color_map = {
    'Conventional Steam Coal': '#3c3c3c', 'Natural Gas Fired Combined Cycle': '#CB4335',
    'Natural Gas Fired Combustion Turbine': '#D35400', 'Other Natural Gas': '#E67E22',
    'Nuclear': '#6a0dad', 'Other': '#a9a9a9', 'Onshore Wind Turbine': '#5D9732',
    'Solar Photovoltaic': '#FFC423', 'Geothermal': '#8C564B', 'Conventional Hydroelectric': '#457B9D',
}
updated_color_map = {label_rename_map.get(k, k): v for k, v in technology_color_map.items()}

# Set the style for the plot for better aesthetics
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))

# Create the box plot
sorted_order = df_2023.groupby('technology_description')['capacity_mw_x'].median().sort_values(ascending=False).index
ax = sns.boxplot(
    x='technology_description', y='capacity_mw_x', data=df_2023,
    order=sorted_order, palette=updated_color_map, hue='technology_description', legend=False
)

# Set the y-axis to a logarithmic scale and format ticks
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())

# --- Apply custom fonts, sizes, and weights to plot elements ---
font_size = 24

# Set X and Y axis titles using the BOLD font property
plt.xlabel('Technology Group', fontproperties=bold_font, fontsize=font_size)
plt.ylabel('Capacity (MW) - Log Scale', fontproperties=bold_font, fontsize=font_size)

# Set font and size for the tick labels using the REGULAR font property
for label in ax.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_size)
    label.set_rotation(45)
    label.set_ha('right')

for label in ax.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_size)

# Adjust layout to make sure everything fits without being cut off
plt.tight_layout()

# Save the figure to the output folder
figure_output_path = os.path.join(root, 'output', 'figure_1_capacity_boxplot_2023.png')
plt.savefig(figure_output_path, dpi=300)

print(f"Successfully saved box plot figure with custom colors to: {figure_output_path}")