import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the project root directory
# -----------------------------------------------------------------------------
try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Load the cleaned cost data
# -----------------------------------------------------------------------------
data_path = os.path.join(root, "intermediate data", "cleaned_cost_dataV3.csv")
df = pd.read_csv(data_path, low_memory=False)

print(f"Loaded dataset: {len(df)} rows")

# 3. Define OPEX categories and aggregate
# -----------------------------------------------------------------------------
opex_categories = [
    'opex_operations', 'opex_fuel', 'opex_coolants', 'opex_steam', 
    'opex_steam_other', 'opex_transfer', 'opex_electric', 'opex_misc_power', 
    'opex_rents', 'opex_allowances', 'opex_engineering', 'opex_structures', 
    'opex_boiler', 'opex_plants', 'opex_misc_steam'
]

# Categories to keep separate
categories_to_keep = ['opex_operations', 'opex_fuel', 'opex_steam', 'opex_boiler']

# Create opex_misc from two components
df['opex_misc'] = df['opex_misc_steam'] + df['opex_misc_power']

# All other categories to aggregate into "opex_other"
categories_to_aggregate = [cat for cat in opex_categories if cat not in categories_to_keep]
df['opex_other'] = df[categories_to_aggregate].sum(axis=1)

# Select only the final aggregated categories
final_opex_categories = categories_to_keep + ['opex_misc', 'opex_other']

# 4. Function to analyze and plot OPEX for a given year
# -----------------------------------------------------------------------------
def analyze_opex_by_year(year, df_full):
    print(f"\n\n{'='*80}")
    print(f"ANALYZING OPEX FOR {year}")
    print(f"{'='*80}")
    
    df_year = df_full[df_full['report_year'] == year]
    print(f"Filtered for report_year = {year}: {len(df_year)} rows")
    
    # Group by technology and sum each final OPEX category
    opex_by_tech = df_year.groupby('technology_description')[final_opex_categories].sum()
    
    # Calculate percentage shares for each technology
    opex_by_tech_pct = opex_by_tech.div(opex_by_tech.sum(axis=1), axis=0) * 100
    
    # Calculate average shares across all technologies
    category_avg_shares = opex_by_tech_pct.mean(axis=0).sort_values(ascending=False)
    
    print(f"\nAverage OPEX Category Shares across all technologies (%):")
    print(category_avg_shares.apply(lambda x: f'{x:.1f}%'))
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"OPEX CATEGORY SHARES BY TECHNOLOGY (%) - {year}")
    print(f"{'='*80}")
    print(opex_by_tech_pct.round(1).to_string())
    print(f"{'='*80}")
    
    # Export summary table to CSV
    csv_path = os.path.join(root, "output", f"opex_shares_{year}.csv")
    opex_by_tech_pct.round(1).to_csv(csv_path)
    print(f"Summary table saved to: {csv_path}")
    
    # Create and save plot
    font_size = 12
    plt.rcParams['font.family'] = 'Cambria'
    
    fig, ax = plt.subplots(figsize=(14, 8))
    opex_by_tech_pct.plot(kind='bar', stacked=True, ax=ax, width=0.7)
    
    ax.set_xlabel('Technology', fontsize=font_size, fontname='Cambria')
    ax.set_ylabel('Share of Operational Expenditures (%)', fontsize=font_size, fontname='Cambria')
    ax.set_title(f'Operational Expenditure Shares by Technology and Cost Category ({year})', fontsize=font_size+2, fontname='Cambria', fontweight='bold')
    ax.legend(title='OPEX Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.xticks(rotation=45, ha='right', fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.tight_layout()
    
    plot_path = os.path.join(root, "output", f"opex_shares_{year}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}\n")
    #plt.show()
    
    return opex_by_tech_pct

# 5. Analyze both years
# -----------------------------------------------------------------------------
opex_2023 = analyze_opex_by_year(2023, df)
opex_1995 = analyze_opex_by_year(1995, df)

# 6. Create side-by-side comparison plot
# -----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("CREATING SIDE-BY-SIDE COMPARISON PLOT")
print(f"{'='*80}\n")

font_size = 12
plt.rcParams['font.family'] = 'Cambria'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1995
opex_1995.plot(kind='bar', stacked=True, ax=ax1, width=0.7, legend=False)
ax1.set_xlabel('Technology', fontsize=font_size, fontname='Cambria')
ax1.set_ylabel('Share of Operational Expenditures (%)', fontsize=font_size, fontname='Cambria')
ax1.set_title('1995', fontsize=font_size+2, fontname='Cambria', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 100)
ax1.tick_params(axis='x', rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=font_size-2)
plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=font_size-2)

# Plot 2023
opex_2023.plot(kind='bar', stacked=True, ax=ax2, width=0.7)
ax2.set_xlabel('Technology', fontsize=font_size, fontname='Cambria')
ax2.set_ylabel('Share of Operational Expenditures (%)', fontsize=font_size, fontname='Cambria')
ax2.set_title('2023', fontsize=font_size+2, fontname='Cambria', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)
ax2.tick_params(axis='x', rotation=45)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=font_size-2)
plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=font_size-2)

# Adjust legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, title='OPEX Category', loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=font_size-2)
ax2.legend().remove()

plt.suptitle('Operational Expenditure Shares by Technology: 1995 vs 2023', 
             fontsize=font_size+3, fontname='Cambria', fontweight='bold', y=1.02)
plt.tight_layout()

# Save the comparison plot
plot_path = os.path.join(root, "output", "opex_shares_comparison_1995_2023.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {plot_path}\n")
#plt.show()

# 7. Create combined summary table with both years
# -----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("COMBINED OPEX SHARE SUMMARY: 1995 AND 2023")
print(f"{'='*80}")

# Add year identifier to the index
opex_1995_labeled = opex_1995.copy()
opex_1995_labeled.index = [f"{tech} 1995" for tech in opex_1995_labeled.index]

opex_2023_labeled = opex_2023.copy()
opex_2023_labeled.index = [f"{tech} 2023" for tech in opex_2023_labeled.index]

# Combine the two dataframes
opex_combined = pd.concat([opex_1995_labeled, opex_2023_labeled])

# Sort by technology name (this groups each technology's years together)
opex_combined = opex_combined.sort_index()

print(opex_combined.round(1).to_string())
print(f"{'='*80}")

# Export combined table to CSV
csv_path = os.path.join(root, "output", "opex_shares_combined_1995_2023.csv")
opex_combined.round(1).to_csv(csv_path)
print(f"Combined summary table saved to: {csv_path}\n")

# 8. Create change summary table (optional - keep if useful)
# -----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("OPEX SHARE CHANGES: 2023 vs 1995 (percentage points)")
print(f"{'='*80}")

opex_change = opex_2023 - opex_1995
print(opex_change.round(1).to_string())
print(f"{'='*80}")

# Export change table to CSV
csv_path_change = os.path.join(root, "output", "opex_shares_change_1995_to_2023.csv")
opex_change.round(1).to_csv(csv_path_change)
print(f"Change table saved to: {csv_path_change}\n")

# 9. Calculate capacity and capacity factor by technology for 2023
# -----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("CAPACITY AND CAPACITY FACTOR BY TECHNOLOGY - 2023")
print(f"{'='*80}\n")

# Filter for 2023
df_2023 = df[df['report_year'] == 2023]

# Calculate total capacity in kW for each technology
capacity_by_tech = df_2023.groupby('technology_description')['capacity_mw_x'].sum() * 1000  # Convert MW to kW

print("Total Capacity by Technology (kW):")
print(capacity_by_tech.apply(lambda x: f'{x:,.0f}'))

# Calculate number of units and average capacity per unit
unit_count_by_tech = df_2023.groupby('technology_description').size()
avg_capacity_by_tech = df_2023.groupby('technology_description')['capacity_mw_x'].mean()  # Average in MW

print("\n\nNumber of Units by Technology:")
print(unit_count_by_tech.to_string())

print("\n\nAverage Unit Capacity by Technology (MW):")
print(avg_capacity_by_tech.apply(lambda x: f'{x:,.1f}'))

# Calculate capacity factor for each technology
# Capacity factor = actual generation / (capacity * hours in year)
generation_by_tech = df_2023.groupby('technology_description')['net_generation_mwh'].sum()
capacity_mw_by_tech = df_2023.groupby('technology_description')['capacity_mw_x'].sum()
max_generation_by_tech = capacity_mw_by_tech * 8760  # Maximum possible generation (MWh)

capacity_factor_by_tech = (generation_by_tech / max_generation_by_tech) * 100  # As percentage

print("\n\nCapacity Factor by Technology (%):")
print(capacity_factor_by_tech.apply(lambda x: f'{x:.1f}%'))

# Create summary table
capacity_summary = pd.DataFrame({
    'Number of Units': unit_count_by_tech,
    'Capacity (kW)': capacity_by_tech,
    'Capacity (MW)': capacity_mw_by_tech,
    'Avg Unit Capacity (MW)': avg_capacity_by_tech,
    'Generation (MWh)': generation_by_tech,
    'Max Generation (MWh)': max_generation_by_tech,
    'Capacity Factor (%)': capacity_factor_by_tech
})

# Calculate average total OPEX per plant (excluding fuel)
opex_categories_no_fuel = [cat for cat in final_opex_categories if cat != 'opex_fuel']
total_opex_by_tech = df_2023.groupby('technology_description')[opex_categories_no_fuel].sum().sum(axis=1)
avg_opex_per_plant = total_opex_by_tech / unit_count_by_tech

capacity_summary['Avg Total OPEX per Plant ($)'] = avg_opex_per_plant

print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
print(capacity_summary.to_string())
print(f"{'='*80}")

# Export summary to CSV
csv_path = os.path.join(root, "output", "opex_2023_atb_compare.csv")
capacity_summary.to_csv(csv_path)
print(f"\nCapacity summary saved to: {csv_path}\n")
