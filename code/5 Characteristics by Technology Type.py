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

# Define the direct path to the font file ---
font_path = 'C:/Windows/Fonts/cambria.ttc'
custom_font = FontProperties(fname=font_path)

# --- Define shorter names for plot labels ---
label_rename_map = {
    'Natural Gas Fired Combined Cycle': 'Natural Gas CC',
    'Natural Gas Fired Combustion Turbine': 'Natural Gas CT',
    'Conventional Steam Coal': 'Coal',
    'Onshore Wind Turbine':'Onshore Wind'
}
# Apply the renaming to the DataFrame column used for plotting
df_2023['technology_description'] = df_2023['technology_description'].replace(label_rename_map)


# Define the original custom color map
technology_color_map = {
    # Fossil Fuels
    'Conventional Steam Coal': '#3c3c3c',
    'Natural Gas Fired Combined Cycle': '#CB4335',
    'Natural Gas Fired Combustion Turbine': '#D35400',
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

# Update the color map keys to match the new shorter labels
updated_color_map = {label_rename_map.get(k, k): v for k, v in technology_color_map.items()}


# Set the style for the plot for better aesthetics
sns.set_style("whitegrid")

# Create the figure and axes for the plot
plt.figure(figsize=(16, 8))

# Create the box plot using the updated data and color map
sorted_order = df_2023.groupby('technology_description')['capacity_mw_x'].median().sort_values(ascending=False).index
ax = sns.boxplot(
    x='technology_description',
    y='capacity_mw_x',
    data=df_2023,
    order=sorted_order,
    palette=updated_color_map, # <-- Use the updated color map
    hue='technology_description',
    legend=False
)

# Set the y-axis to a logarithmic scale
ax.set_yscale('log')

# Add a formatter to prevent the y-axis from using scientific notation
ax.yaxis.set_major_formatter(ScalarFormatter())

# --- Apply the custom font to each text element ---
font_size=24
plt.xlabel('Technology Group', fontproperties=custom_font, fontsize=font_size, fontweight='bold')
plt.ylabel('Capacity (MW) - Log Scale', fontproperties=custom_font, fontsize=font_size, fontweight='bold')

# To set the font for the tick labels, we iterate through them
for label in ax.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_size)

for label in ax.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_size)

# Improve the plot's labels and title
#plt.title('Distribution of Plant Capacity (MW) by Technology (2023)', fontsize=16)
plt.xlabel('Technology Group', fontsize=font_size)
plt.ylabel('Capacity (MW) - Log Scale', fontsize=font_size)

# Rotate the x-axis labels AND set their font size
plt.xticks(rotation=45, ha='right', fontsize=font_size)
# Set the Y-axis label font size
plt.yticks(fontsize=font_size)


# Adjust layout to make sure everything fits without being cut off
plt.tight_layout()

# Save the figure to the output folder
figure_output_path = os.path.join(root, 'output', 'figure_1_capacity_boxplot_2023.png')
plt.savefig(figure_output_path, dpi=300)

print(f"Successfully saved box plot figure with custom colors to: {figure_output_path}")



# --- Table 3: Comparison of FERC Sample Capacity to EIA National Totals ---
print("\nGenerating Table 3: Comparison of FERC Sample Capacity to EIA National Totals...")

# --- Step 1: Load and Filter the National EIA Generator Data ---
try:
    eia_path = os.path.join(root, 'data', 'eia_gen.csv')
    df_eia = pd.read_csv(eia_path, low_memory=False)
    print(f"Successfully loaded EIA data from: {eia_path}. Initial rows: {len(df_eia)}")
except FileNotFoundError:
    print(f"Error: The file was not found at {eia_path}")
    print("Please ensure 'eia_gen.csv' is in the 'data' directory.")
    # Exit or handle the error appropriately if the file is essential
    exit()

# Filter EIA data to include only generators with an "existing" operational status
df_eia = df_eia[df_eia['operational_status'] == 'existing'].copy()
print(f"Filtered EIA data to keep 'existing' generators. Rows remaining: {len(df_eia)}")


# --- Step 2: Define a mapping from EIA technologies to your study's Technology Groups ---
# This is a sample mapping to align EIA's detailed descriptions with your broader categories.
eia_to_ferc_tech_map = {
    'Conventional Steam Coal': 'Conventional Steam Coal',
    'Natural Gas Fired Combined Cycle': 'Natural Gas Fired Combined Cycle',
    'Natural Gas Fired Combustion Turbine': 'Natural Gas Fired Combustion Turbine',
    'Natural Gas Steam Turbine': 'Other Natural Gas',
    'Other Natural Gas': 'Other Natural Gas',
    'Natural Gas Internal Combustion Engine': 'Other Natural Gas',
    'Natural Gas with Compressed Air Storage': 'Other Natural Gas',
    'Nuclear': 'Nuclear',
    'Onshore Wind Turbine': 'Onshore Wind Turbine',
    'Solar Photovoltaic': 'Solar Photovoltaic',
    'Geothermal': 'Geothermal',
    'Petroleum Liquids': 'Other',
    'Wood/Wood Waste Biomass': 'Other',
    'Coal Integrated Gasification Combined Cycle': 'Other',
    'Other Gases': 'Other',
    'Municipal Solid Waste': 'Other',
    'All Other': 'Other',
    'Batteries':'Batteries',
    'Conventional Hydroelectric': 'Hydroelectric',
    'Hydroelectric Pumped Storage': 'Hydroelectric',
    'Solar Thermal without Energy Storage': 'Solar Thermal',
    'Solar Thermal with Energy Storage': 'Solar Thermal'
}

# Apply the mapping to create a new column in the EIA dataframe
# Unmapped technologies will result in NaN
df_eia['Technology Group'] = df_eia['technology_description'].map(eia_to_ferc_tech_map)

# Fill any unmapped technologies with 'Other' using recommended syntax
df_eia['Technology Group'] = df_eia['Technology Group'].fillna('Other')
df_eia.to_csv(os.path.join(root, 'data', 'eia_gen_with_ferc_mapping.csv'), index=False)

# --- Step 3: Calculate total national capacity for each technology group from EIA data ---
eia_national_capacity = df_eia.groupby('Technology Group').agg(
    eia_national_capacity_mw=('capacity_mw', 'sum')
).reset_index()


# --- Step 4: Merge the FERC sample data with the EIA national data ---
# We use the 'table1_df' which already contains the aggregated FERC sample capacity
# A left merge ensures we keep all technologies from our FERC sample.
comparison_df = pd.merge(
    table1_df,
    eia_national_capacity,
    on='Technology Group',
    how='outer'
)


# Rename columns for clarity in the final table
comparison_df.rename(columns={
    'Total Capacity (MW)': 'FERC Sample: Total Capacity (MW)',
    'eia_national_capacity_mw': 'EIA National: Total Capacity (MW)'
}, inplace=True)



# --- Step 5: Calculate the coverage percentage ---
# An outer merge creates NaN for columns where a technology group is missing from one of the dataframes.
# We fill the FERC capacity column with 0 for those cases.
comparison_df['FERC Sample: Total Capacity (MW)'] = comparison_df['FERC Sample: Total Capacity (MW)'].fillna(0)

comparison_df['FERC as % of National Capacity (Coverage)'] = \
    (comparison_df['FERC Sample: Total Capacity (MW)'] / comparison_df['EIA National: Total Capacity (MW)']) * 100

# Fill any potential NaN values using recommended syntax
comparison_df['FERC as % of National Capacity (Coverage)'] = comparison_df['FERC as % of National Capacity (Coverage)'].fillna(0)


# --- Step 6: Add a "Total" Row ---
total_ferc_capacity = comparison_df['FERC Sample: Total Capacity (MW)'].sum()
total_eia_capacity = comparison_df['EIA National: Total Capacity (MW)'].sum()
total_coverage = (total_ferc_capacity / total_eia_capacity) * 100 if total_eia_capacity else 0

total_row = pd.DataFrame([{
    'Technology Group': 'Total',
    'FERC Sample: Total Capacity (MW)': total_ferc_capacity,
    'EIA National: Total Capacity (MW)': total_eia_capacity,
    'FERC as % of National Capacity (Coverage)': total_coverage
}])

comparison_df = pd.concat([comparison_df, total_row], ignore_index=True)


# --- Step 7: Display and Export the Final Comparison Table ---

# Create a display version with formatting for better console output
display_df3 = comparison_df.copy()
display_df3['FERC Sample: Total Capacity (MW)'] = display_df3['FERC Sample: Total Capacity (MW)'].map('{:,.0f}'.format)
display_df3['EIA National: Total Capacity (MW)'] = display_df3['EIA National: Total Capacity (MW)'].map('{:,.0f}'.format)
display_df3['FERC as % of National Capacity (Coverage)'] = display_df3['FERC as % of National Capacity (Coverage)'].map('{:.2f}%'.format)

# Reorder columns for the final presentation
display_df3 = display_df3[[
    'Technology Group',
    'FERC Sample: Total Capacity (MW)',
    'EIA National: Total Capacity (MW)',
    'FERC as % of National Capacity (Coverage)'
]]


# Print the formatted table to the console
print("\n" + "="*120)
print("Table 3: Comparison of FERC Sample Capacity to EIA National Totals by Technology (2023)")
print("="*120)
print(display_df3.to_string(index=False))
print("="*120)

# Export the raw numerical data to a CSV file
table3_output_path = os.path.join(root, 'output', 'table_3_ferc_vs_eia_capacity_2023.csv')
comparison_df.to_csv(table3_output_path, index=False, float_format='%.2f')

print(f"\nSuccessfully exported the data for Table 3 to: {table3_output_path}")
