import os
import pandas as pd
import matplotlib.pyplot as plt


# 1. Define the project root directory
# -----------------------------------------------------------------------------
try:
    # This works when running the script from a subfolder
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Load the cleaned cost data
# -----------------------------------------------------------------------------
data_path = os.path.join(root, "intermediate data", "cleaned_cost_dataV3.csv")
df = pd.read_csv(data_path, low_memory=False)

# 3. Filter for plants with installation year > 1994
# -----------------------------------------------------------------------------
df_filtered = df[df['installation_year'] > 1994]

print(f"Original dataset: {len(df)} rows")
print(f"Filtered dataset (installation_year > 1994): {len(df_filtered)} rows")
print(f"Removed {len(df) - len(df_filtered)} rows")

# 4. Filter for the lowest report_year for each plant
# -----------------------------------------------------------------------------
df_earliest = df_filtered.loc[df_filtered.groupby('plant_name_ferc1')['report_year'].idxmin()]

print(f"After filtering for earliest report_year per plant: {len(df_earliest)} rows")
print(f"Removed {len(df_filtered) - len(df_earliest)} rows (duplicate plants with later report years)")

# 4.5. Fix capex_total for specific plants
# -----------------------------------------------------------------------------
df_earliest.loc[df_earliest['plant_name_ferc1'] == 'west gardner', 'capex_total'] = 118813492
print("\nFixed capex_total for 'west gardner' plant")

# 5. Calculate capital cost per kW
# -----------------------------------------------------------------------------
# Convert MW to kW (multiply by 1000)
df_earliest['capacity_kw'] = df_earliest['capacity_mw_x'] * 1000

# Calculate capital cost per kW ($/kW)
df_earliest['capex_per_kw'] = df_earliest['capex_total'] / df_earliest['capacity_kw']

# Display summary statistics
print("\nCapital Cost per kW Summary:")
print(f"Mean: ${df_earliest['capex_per_kw'].mean():.2f}/kW")
print(f"Median: ${df_earliest['capex_per_kw'].median():.2f}/kW")
print(f"Min: ${df_earliest['capex_per_kw'].min():.2f}/kW")
print(f"Max: ${df_earliest['capex_per_kw'].max():.2f}/kW")

# Check for any plants with missing or zero capacity
missing_capacity = df_earliest[df_earliest['capacity_mw_x'].isna() | (df_earliest['capacity_mw_x'] <= 0)]
if len(missing_capacity) > 0:
    print(f"\nWarning: {len(missing_capacity)} plants have missing or zero capacity and cannot calculate $/kW")

# 6. Select specific columns and export
# -----------------------------------------------------------------------------
columns_to_keep = [
    'plant_name_ferc1',
    'plant_id_eia', 
    'technology_description', 
    'report_year', 
    'installation_year', 
    'capacity_mw_x', 
    'capacity_kw',
    'capex_total', 
    'capex_per_kw', 
    'state', 
    'latitude', 
    'longitude'
]

# Select only the specified columns
df_final = df_earliest[columns_to_keep].copy()

# 6.5. Manual data cleaning and corrections
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("MANUAL DATA CLEANING")
print("="*80)

# Plants to remove
plants_to_remove = [
    'rathdrum', 'gila river_2003_258', 'ceredo', 'horseshoe lake', 
    'rockingham', 'blue lake', 'burlington ct', 'attala', 'wheatland', 
    'vestaburg', 'northeastern 1&2', 'gaylord', 'vermillion', 'energy center', 
    "paddy's run ct", 'glendive', 'beaver'
]

initial_count = len(df_final)
df_final = df_final[~df_final['plant_name_ferc1'].isin(plants_to_remove)]
removed_count = initial_count - len(df_final)
print(f"Removed {removed_count} plants from dataset")

# Capex corrections (apply before recalculating capex_per_kw)
capex_corrections = {
    'osawatomie': 31488703,
    'possum point com cyc': 376006963,
    'north loop': 27433905,
    'darlington': 113158806,
    'greene county ct': 185883047
}

for plant_name, new_capex in capex_corrections.items():
    mask = df_final['plant_name_ferc1'] == plant_name
    if mask.any():
        df_final.loc[mask, 'capex_total'] = new_capex
        print(f"Updated capex_total for '{plant_name}' to ${new_capex:,}")

# Recalculate capex_per_kw after corrections
df_final['capex_per_kw'] = df_final['capex_total'] / df_final['capacity_kw']
print(f"\nRecalculated capex_per_kw for corrected plants")
print("="*80)

# Export to CSV
output_path = os.path.join(root, "intermediate data", "installation_capex.csv")
df_final.to_csv(output_path, index=False)

print(f"\nExported {len(df_final)} rows to: {output_path}")
print(f"Final dataset columns: {list(df_final.columns)}")

# 7. Create visualization of capex per kW by installation year
# -----------------------------------------------------------------------------
# Sizing parameters
font_size = 28
point_size = 140

plt.figure(figsize=(14, 8))

# Define custom color mapping
color_mapping = {
    'Coal': '#3c3c3c',
    'Natural Gas Fired CC': '#A0522D',
    'Natural Gas Fired CT': '#FF8C00',
    'Nuclear': '#C8A2C8',
}

# Get unique technologies
technologies = df_final['technology_description'].unique()

# Create scatter plot for each technology
for tech in technologies:
    tech_data = df_final[df_final['technology_description'] == tech]
    color = color_mapping.get(tech, '#808080')  # Default to gray if not in mapping
    plt.scatter(tech_data['installation_year'], 
                tech_data['capex_per_kw'], 
                label=tech, 
                alpha=0.6, 
                s=point_size,
                color=color)

plt.xlabel('Installation Year', fontsize=font_size, fontname='Cambria')
plt.ylabel('Capital Cost ($/kW)', fontsize=font_size, fontname='Cambria')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'Cambria', 'size': font_size-2}, frameon=False)
plt.grid(True, alpha=0.3)

# Format x-axis to show integers only
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

# Set x-axis ticks to every 5 years
import numpy as np
x_min = df_final['installation_year'].min()
x_max = df_final['installation_year'].max()
ax.set_xticks(np.arange(np.floor(x_min / 5) * 5, np.ceil(x_max / 5) * 5 + 1, 5))

# Set tick label fonts
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Cambria')
    label.set_fontsize(font_size-2)

plt.tight_layout()

# Save the plot to output folder
plot_path = os.path.join(root, "output", "installed_capex_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

#plt.show()

# 8. Print average capex per kW by year and technology
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("AVERAGE CAPITAL COST PER KW BY INSTALLATION YEAR AND TECHNOLOGY")
print("="*80)

# Calculate average capex per kW for each year and technology
avg_capex_table = df_final.groupby(['installation_year', 'technology_description'])['capex_per_kw'].mean().reset_index()
avg_capex_table = avg_capex_table.pivot(index='installation_year', columns='technology_description', values='capex_per_kw')

# Format the table for display
print(avg_capex_table.to_string(float_format=lambda x: f'${x:,.0f}'))
print("="*80)

# 9. Count number of generators by technology
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("NUMBER OF GENERATORS BY TECHNOLOGY")
print("="*80)

generator_counts = df_final['technology_description'].value_counts().sort_index()
print(generator_counts.to_string())
print(f"\nTotal generators in sample: {len(df_final)}")
print("="*80)
