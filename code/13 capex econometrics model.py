# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

import os
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 1. Define the project root directory and load the prepared data
# -----------------------------------------------------------------------------
try:
    # This works when running the script from a subfolder
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# Construct the full path to the model-ready data file
data_path = os.path.join(root, "intermediate data", "capex_econometric", "df_model_ready.csv")
print(f"Reading model-ready data from: {data_path}")

# Read the data into a pandas DataFrame
df = pd.read_csv(data_path, low_memory=False)
print(f"Initial data shape: {df.shape}")

# Set up the DataFrame for Panel Data Analysis
# -----------------------------------------------------------------------------
df = df.set_index(['plant_name_ferc1', 'report_year'], inplace=False)
df['report_year'] = df.index.get_level_values('report_year')

# Compute within-plant first differences (flows)
df['capex_flow'] = df.groupby(level=0)['capex_total'].diff()
df['d_capacity_mw_x'] = df.groupby(level=0)['capacity_mw_x'].diff()

# Drop first year per plant (no lag to diff)
df_flow = df.dropna(subset=['capex_flow']).copy()
print(f"Flow sample size: {df_flow.shape[0]} rows across {df_flow.index.get_level_values(0).nunique()} plants.")

# Event Study Implementation 
# ----------------------------------------------------------------------------- 

print("\n\n" + "="*80)
print(" Event Study: Coal Plant Capital Investment Dynamics")
print("="*80)

# Create event time relative to reference year (1995)
df_flow['event_time'] = df_flow['report_year'] - 1995

# Create lead and lag indicators (relative to 1995) - START FROM 1, NOT 0
event_years = range(1, 29)  # 1996 to 2023 (exclude 1995 baseline)

# Check available years and create event variables
available_years = sorted(df_flow['report_year'].unique())
print(f"Available years: {min(available_years)} to {max(available_years)}")

valid_event_years = []
for t in event_years:  # Now starts from t=1
    year = 1995 + t
    if year in available_years:
        # Create variable names: event_1, event_2, etc. (no event_0)
        var_name = f'event_{t}'
        df_flow[var_name] = np.where(
            (df_flow['is_coal_ge25mw_pre2001'] == 1) & (df_flow['event_time'] == t), 1, 0
        )
        valid_event_years.append(t)

print(f"Event study covers years: {[1995 + t for t in valid_event_years]}")
print(f"Baseline year (omitted): 1995")

event_vars = []
for t in valid_event_years:
    var_name = f'event_{t}'
    event_vars.append(var_name)

# Build the event study formula (intercept included by default)
event_formula = f"""
capex_flow ~ {' + '.join(event_vars)}
             + d_capacity_mw_x
"""

print(f"Event study includes {len(event_vars)} event time indicators (1995 omitted as baseline)")
print("Reference group: Gas and nuclear plants across all years")

# Fit the event study model with entity clustering only
model_event = PanelOLS.from_formula(event_formula, data=df_flow)
# Main model - entity clustering only
res_event = model_event.fit(cov_type='clustered', cluster_entity=True)
print(res_event)

# Extract event study coefficients for analysis
event_coeffs = []
event_ses = []
event_times = []

for t in valid_event_years:
    var_name = f'event_{t}'
    if var_name in res_event.params.index:
        event_coeffs.append(res_event.params[var_name])
        event_ses.append(res_event.std_errors[var_name])
        event_times.append(t)

# Convert to arrays for easier manipulation
event_coeffs = np.array(event_coeffs)
event_ses = np.array(event_ses)
event_times = np.array(event_times)

# Calculate confidence intervals
ci_lower = event_coeffs - 1.96 * event_ses
ci_upper = event_coeffs + 1.96 * event_ses

print(f"\nEvent Study Coefficients Summary (relative to 1995 baseline):")
print(f"{'Year':<8} {'Event_t':<8} {'Coeff':<15} {'Std_Err':<15} {'CI_Lower':<15} {'CI_Upper':<15}")
print("-" * 90)
for i, t in enumerate(event_times):
    year = 1995 + t
    print(f"{year:<8} {t:<8} {event_coeffs[i]:<15.0f} {event_ses[i]:<15.0f} {ci_lower[i]:<15.0f} {ci_upper[i]:<15.0f}")

# Calculate average effects by policy-relevant periods
print("\n\n" + "="*80)
print(" POLICY-RELEVANT INSIGHTS FROM EVENT STUDY")
print("="*80)

periods = {
    'Early baseline (1996-2000)': [t for t in event_times if 1 <= t <= 5],
    'Pre-regulation anticipation (2001-2005)': [t for t in event_times if 6 <= t <= 10],
    'Initial compliance (2006-2007)': [t for t in event_times if 11 <= t <= 12], 
    'Peak compliance (2008-2010)': [t for t in event_times if 13 <= t <= 15],
    'Extended compliance (2011-2015)': [t for t in event_times if 16 <= t <= 20],
    'Late compliance/transition (2016-2019)': [t for t in event_times if 21 <= t <= 24],
    'Post-compliance/new era (2020-2023)': [t for t in event_times if 25 <= t <= 28]
}

print("Average investment effects by period:")
for period_name, period_ts in periods.items():
    if len(period_ts) > 0:
        period_coeffs = [event_coeffs[i] for i, t in enumerate(event_times) if t in period_ts]
        if len(period_coeffs) > 0:
            avg_effect = np.mean(period_coeffs)
            min_effect = np.min(period_coeffs)
            max_effect = np.max(period_coeffs)
            print(f"  {period_name:35s}: ${avg_effect/1e6:5.1f}M avg (range: ${min_effect/1e6:4.1f}M - ${max_effect/1e6:4.1f}M)")

# Calculate cumulative investment over key periods
cumulative_2008_2010 = np.sum([event_coeffs[i] for i, t in enumerate(event_times) if 13 <= t <= 15])
cumulative_2001_2005 = np.sum([event_coeffs[i] for i, t in enumerate(event_times) if 6 <= t <= 10])

print(f"\nCumulative investment estimates:")
print(f"  Pre-regulation period (2001-2005): ${cumulative_2001_2005/1e6:.1f}M total")
print(f"  Peak compliance period (2008-2010): ${cumulative_2008_2010/1e6:.1f}M total")
print(f"  Compliance period acceleration: ${(cumulative_2008_2010-cumulative_2001_2005)/1e6:.1f}M additional")

print(f"\nKey takeaways:")
print(f"  • Investment patterns show evolution from 1995 baseline through 2023")
print(f"  • Substantial anticipatory investment began around 2001")
print(f"  • Peak investment occurred during 2008-2010 compliance period") 
print(f"  • Regulations appear to have accelerated existing investment trends")
print(f"  • Investment patterns show complex, multi-year response to regulatory environment")

# Event Study Plot
print(f"\nEvent Study Plot (Coefficients in millions):")
print("=" * 60)
max_coeff = max(abs(event_coeffs)) if len(event_coeffs) > 0 else 1
for i, t in enumerate(event_times):
    year = 1995 + t
    coeff_millions = event_coeffs[i] / 1e6
    # Create a simple bar representation
    bar_length = int(20 * abs(coeff_millions) / (max_coeff / 1e6)) if max_coeff > 0 else 0
    bar = "+" * bar_length if coeff_millions >= 0 else "-" * bar_length
    compliance_marker = " <-- Compliance starts" if t == 11 else ""
    reference_marker = " <-- Reference period" if t == 0 else ""
    print(f"{year} (t={t:2d}): {coeff_millions:6.1f}M |{bar:<20}|{compliance_marker}{reference_marker}")

print("\nNote: Use matplotlib to create a proper event study plot with confidence intervals")

# Create proper event study plot with confidence intervals
# -----------------------------------------------------------------------------
print("\nCreating event study plot...")

# Font size parameter - adjust this to control all text sizes
FONT_SIZE = 26  # Change this value to adjust all text sizes

# Line styling parameter - adjust this to control line thickness and point size
LINE_WIDTH = 5  # Change this value to adjust line thickness
POINT_SIZE = 12  # Change this value to adjust point/marker size

# Set font to Cambria
plt.rcParams['font.family'] = 'Cambria'

# Main event study plot - REMOVE YELLOW SHADING
# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Convert event times to actual years for plotting
years = [1995 + t for t in event_times]
coeffs_millions = event_coeffs / 1e6
ci_lower_millions = ci_lower / 1e6
ci_upper_millions = ci_upper / 1e6

# Plot the coefficient estimates FIRST (so it appears first in legend)
ax.plot(years, coeffs_millions, 'o-', color='darkblue', linewidth=LINE_WIDTH, 
        markersize=POINT_SIZE, label='Coal Investment Premium vs. Gas and Nuclear')

# Plot confidence intervals as shaded area
ax.fill_between(years, ci_lower_millions, ci_upper_millions, 
                alpha=0.3, color='lightblue', label='95% Confidence Interval')

# Add horizontal reference line at zero
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# REMOVED: ax.axvspan(2010, 2017, alpha=0.15, color='yellow', label='EPA Regulation Compliance Period')

# Formatting with font size parameter
ax.set_xlabel('Year', fontsize=FONT_SIZE, fontweight='bold')
ax.set_ylabel('$ Millions', fontsize=FONT_SIZE, fontweight='bold')

# Set x-axis to show every few years with font size control
ax.set_xticks(range(1995, 2024, 3))
ax.set_xticklabels(range(1995, 2024, 3), rotation=45, fontsize=FONT_SIZE)

# Set y-axis tick label font size
ax.tick_params(axis='y', labelsize=FONT_SIZE)

# Add grid
ax.grid(True, alpha=0.3)

# Legend with font size control
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=FONT_SIZE, framealpha=0.6)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_dir = os.path.join(root, "output")
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "coal_capex_event_study.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Event study plot saved to: {plot_path}")

# Also save as PDF for high-quality output
pdf_path = os.path.join(output_dir, "coal_capex_event_study.pdf")
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Event study plot (PDF) saved to: {pdf_path}")

# Close the plot to free memory
plt.close()

# Show the plot
# plt.show()

# Print summary statistics for the plot
print(f"\nPlot Summary Statistics:")
print(f"  Years covered: {min(years)} - {max(years)}")
print(f"  Peak positive effect: ${max(coeffs_millions):.1f}M in {years[np.argmax(coeffs_millions)]}")
print(f"  Largest negative effect: ${min(coeffs_millions):.1f}M in {years[np.argmin(coeffs_millions)]}")
print(f"  Average effect 2008-2010: ${np.mean([coeffs_millions[i] for i, y in enumerate(years) if 2008 <= y <= 2010]):.1f}M")

# Diagnostic check
print("\nControl group (gas/nuclear) investment patterns:")
control_group = df_flow[df_flow['is_coal_ge25mw_pre2001'] == 0].copy()

# Drop the column version of report_year before resetting index, or use level directly
yearly_control_avg = control_group.groupby(level='report_year')['capex_flow'].agg(['mean', 'count'])
print("Year-by-year gas/nuclear change in plant investment (first 10 years):")
print(yearly_control_avg.head(10))
print(f"\nTotal control group observations: {yearly_control_avg['count'].sum()}")
print(f"Average annual control group size: {yearly_control_avg['count'].mean():.1f} plants")

# Robustness Check 1: Natural Gas Control Group Only
# ----------------------------------------------------------------------------

print("\n\n" + "="*80)
print(" ROBUSTNESS CHECK 1: Event Study with Natural Gas Control Group Only")
print("="*80)

# Filter to include only coal and natural gas plants (both CT and CC)
df_flow_ng = df_flow[df_flow['technology_description'].isin(['Coal', 'Natural Gas Fired CT', 'Natural Gas Fired CC'])].copy()
print(f"Natural gas control sample: {df_flow_ng.shape[0]} rows across {df_flow_ng.index.get_level_values(0).nunique()} plants.")

# Create event variables for natural gas control model (NO event_0)
for t in valid_event_years:  # This now excludes t=0
    var_name = f'event_{t}'
    df_flow_ng[var_name] = np.where(
        (df_flow_ng['is_coal_ge25mw_pre2001'] == 1) & (df_flow_ng['event_time'] == t), 1, 0
    )

# Fit the natural gas control model with entity clustering only
model_event_ng = PanelOLS.from_formula(event_formula, data=df_flow_ng)
# Robustness check 1 - entity clustering only
res_event_ng = model_event_ng.fit(cov_type='clustered', cluster_entity=True)
print(res_event_ng)

# Extract coefficients
event_coeffs_ng = []
event_ses_ng = []
event_times_ng = [] 

for t in valid_event_years:
    var_name = f'event_{t}'
    if var_name in res_event_ng.params.index:
        event_coeffs_ng.append(res_event_ng.params[var_name])
        event_ses_ng.append(res_event_ng.std_errors[var_name])
        event_times_ng.append(t) 

event_coeffs_ng = np.array(event_coeffs_ng)
event_ses_ng = np.array(event_ses_ng)
ci_lower_ng = event_coeffs_ng - 1.96 * event_ses_ng
ci_upper_ng = event_coeffs_ng + 1.96 * event_ses_ng

# Create the correct years array for NG model
years_ng = [1995 + t for t in event_times_ng]  

# Create natural gas control plot
fig, ax = plt.subplots(figsize=(14, 8))

coeffs_millions_ng = event_coeffs_ng / 1e6
ci_lower_millions_ng = ci_lower_ng / 1e6
ci_upper_millions_ng = ci_upper_ng / 1e6

ax.plot(years_ng, coeffs_millions_ng, 'o-', color='#1f2937', linewidth=LINE_WIDTH, 
        markersize=POINT_SIZE, label='Coal Investment Premium vs. Natural Gas Only')

ax.fill_between(years_ng, ci_lower_millions_ng, ci_upper_millions_ng, 
                alpha=0.3, color='lightblue', label='95% Confidence Interval')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
# REMOVED: ax.axvspan(2009, 2017, alpha=0.15, color='yellow', label='EPA Regulation Compliance Period')

ax.set_xlabel('Year', fontsize=FONT_SIZE, fontweight='bold')
ax.set_ylabel('$ Millions', fontsize=FONT_SIZE, fontweight='bold')
ax.set_xticks(range(1995, 2024, 3))
ax.set_xticklabels(range(1995, 2024, 3), rotation=45, fontsize=FONT_SIZE)
ax.tick_params(axis='y', labelsize=FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=FONT_SIZE, framealpha=0.6)

plt.tight_layout()

# Save natural gas control plot
plot_path_ng = os.path.join(output_dir, "coal_capex_event_study_NGcontrol.png")
plt.savefig(plot_path_ng, dpi=300, bbox_inches='tight')
print(f"Natural gas control plot saved to: {plot_path_ng}")
plt.close()

# Robustness Check 2: Nuclear Control Group Only - CORRECTED
# ----------------------------------------------------------------------------

print("\n\n" + "="*80)
print(" ROBUSTNESS CHECK 2: Event Study with Nuclear Control Group Only")
print("="*80)

# Filter to include only coal and nuclear plants
df_flow_nuke = df_flow[df_flow['technology_description'].isin(['Coal', 'Nuclear'])].copy()
print(f"Nuclear control sample: {df_flow_nuke.shape[0]} rows across {df_flow_nuke.index.get_level_values(0).nunique()} plants.")

# Create event variables for nuclear control model (NO event_0)
for t in valid_event_years:  # This now excludes t=0
    var_name = f'event_{t}'
    df_flow_nuke[var_name] = np.where(
        (df_flow_nuke['is_coal_ge25mw_pre2001'] == 1) & (df_flow_nuke['event_time'] == t), 1, 0
    )

# Fit the nuclear control model with entity clustering only
model_event_nuke = PanelOLS.from_formula(event_formula, data=df_flow_nuke)
# Robustness check 2 - entity clustering only  
res_event_nuke = model_event_nuke.fit(cov_type='clustered', cluster_entity=True)
print(res_event_nuke)

# Extract coefficients
event_coeffs_nuke = []
event_ses_nuke = []
event_times_nuke = [] 

for t in valid_event_years:
    var_name = f'event_{t}'
    if var_name in res_event_nuke.params.index:
        event_coeffs_nuke.append(res_event_nuke.params[var_name])
        event_ses_nuke.append(res_event_nuke.std_errors[var_name])
        event_times_nuke.append(t)

event_coeffs_nuke = np.array(event_coeffs_nuke)
event_ses_nuke = np.array(event_ses_nuke)
ci_lower_nuke = event_coeffs_nuke - 1.96 * event_ses_nuke
ci_upper_nuke = event_coeffs_nuke + 1.96 * event_ses_nuke

# Create the correct years array for nuclear model  
years_nuke = [1995 + t for t in event_times_nuke]

# Create nuclear control plot
fig, ax = plt.subplots(figsize=(14, 8))

coeffs_millions_nuke = event_coeffs_nuke / 1e6
ci_lower_millions_nuke = ci_lower_nuke / 1e6
ci_upper_millions_nuke = ci_upper_nuke / 1e6

ax.plot(years_nuke, coeffs_millions_nuke, 'o-', color='#065f46', linewidth=LINE_WIDTH, 
        markersize=POINT_SIZE, label='Coal Investment Premium vs. Nuclear Only')

ax.fill_between(years_nuke, ci_lower_millions_nuke, ci_upper_millions_nuke, 
                  alpha=0.3, color='lightblue', label='95% Confidence Interval')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
# REMOVED: ax.axvspan(2009, 2017, alpha=0.15, color='yellow', label='EPA Regulation Compliance Period')

ax.set_xlabel('Year', fontsize=FONT_SIZE, fontweight='bold')
ax.set_ylabel('$ Millions', fontsize=FONT_SIZE, fontweight='bold')               
ax.set_xticks(range(1995, 2024, 3))
ax.set_xticklabels(range(1995, 2024, 3), rotation=45, fontsize=FONT_SIZE)
ax.tick_params(axis='y', labelsize=FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=FONT_SIZE, framealpha=0.6)

plt.tight_layout()

# Save nuclear control plot
plot_path_nuke = os.path.join(output_dir, "coal_capex_event_study_NUKEcontrol.png")
plt.savefig(plot_path_nuke, dpi=300, bbox_inches='tight')
print(f"Nuclear control plot saved to: {plot_path_nuke}")
plt.close()

plt.show()

# Summary comparison of all three models
print("\n\n" + "="*80)
print(" COMPARISON OF ROBUSTNESS CHECKS")
print("="*80)

print("Peak compliance period (2008-2010) average effects:")
peak_years_idx = [i for i, y in enumerate(years) if 2008 <= y <= 2010]

main_avg = np.mean([coeffs_millions[i] for i in peak_years_idx])
ng_avg = np.mean([coeffs_millions_ng[i] for i in peak_years_idx])
nuke_avg = np.mean([coeffs_millions_nuke[i] for i in peak_years_idx])

print(f"  Main model (vs. Gas + Nuclear):  ${main_avg:.1f}M")
print(f"  Natural gas control only:        ${ng_avg:.1f}M")
print(f"  Nuclear control only:            ${nuke_avg:.1f}M")

print(f"\nSample sizes:")
print(f"  Main model:       {df_flow.shape[0]} observations")
print(f"  Natural gas only: {df_flow_ng.shape[0]} observations")
print(f"  Nuclear only:     {df_flow_nuke.shape[0]} observations")

# Coal plants in 2008
coal_2008 = df_flow[(df_flow['is_coal_ge25mw_pre2001'] == 1) & 
                   (df_flow['report_year'] == 2008)]['capex_flow'].mean()

# Non-coal plants in 2008  
noncoal_2008 = df_flow[(df_flow['is_coal_ge25mw_pre2001'] == 0) & 
                      (df_flow['report_year'] == 2008)]['capex_flow'].mean()

manual_beta13 = coal_2008 - noncoal_2008
print(f"\nManual check of 2008 coal premium vs. non-coal: ${manual_beta13/1e6:.1f}M (should match event_13 coefficient)")

# Create table of coal investment premiums 2007-2019
print("\n" + "="*60)
print(" COAL INVESTMENT PREMIUM TABLE: 2007-2019")
print("="*60)

# Filter for years 2007-2019 (event times 12-24)
table_years = range(2007, 2020)
table_data = []
total_premium = 0

print(f"{'Year':<6} {'Event_t':<8} {'Premium ($M)':<15} {'Premium ($)':<20}")
print("-" * 50)

for year in table_years:
    event_t = year - 1995  # Convert year to event time
    
    # Find the coefficient for this year
    try:
        idx = list(event_times).index(event_t)
        premium_millions = coeffs_millions[idx]
        premium_dollars = event_coeffs[idx]
        total_premium += premium_dollars
        
        print(f"{year:<6} {event_t:<8} ${premium_millions:<14.1f} ${premium_dollars:<19,.0f}")
        table_data.append({
            'Year': year,
            'Event_t': event_t,
            'Premium_Millions': premium_millions,
            'Premium_Dollars': premium_dollars
        })
        
    except ValueError:
        # Year not found in event times
        print(f"{year:<6} {event_t:<8} {'N/A':<14} {'N/A':<19}")

print("-" * 50)
print(f"{'TOTAL':<6} {'':<8} ${total_premium/1e6:<14.1f} ${total_premium:<19,.0f}")
print(f"\nAverage annual premium 2007-2019: ${(total_premium/len(table_data))/1e6:.1f}M")

# Calculate Total Coal Investment Premium Across All Plants
print("\n" + "="*70)
print(" TOTAL COAL INVESTMENT PREMIUM ACROSS ALL PLANTS: 2007-2019")
print("="*70)

# Count coal plants by year - use the index level explicitly
coal_counts_by_year = df_flow[df_flow['is_coal_ge25mw_pre2001'] == 1].groupby(level='report_year').size()

# Calculate total premiums
total_table_data = []
grand_total_premium = 0

print(f"{'Year':<6} {'Coal Plants':<12} {'Per-Plant ($M)':<15} {'Total Premium ($M)':<18} {'Total Premium ($)':<20}")
print("-" * 75)

for year in range(2007, 2020):
    event_t = year - 1995
    
    try:
        idx = list(event_times).index(event_t)
        per_plant_premium = event_coeffs[idx]
        per_plant_millions = per_plant_premium / 1e6
        
        # Get number of coal plants in this year
        coal_count = coal_counts_by_year.get(year, 0)
        
        # Calculate total premium across all coal plants
        total_premium = per_plant_premium * coal_count
        total_millions = total_premium / 1e6
        grand_total_premium += total_premium
        
        print(f"{year:<6} {coal_count:<12} ${per_plant_millions:<14.1f} ${total_millions:<17.1f} ${total_premium:<19,.0f}")
        
        total_table_data.append({
            'Year': year,
            'Coal_Plants': coal_count,
            'Per_Plant_Premium': per_plant_premium,
            'Total_Premium': total_premium
        })
        
    except ValueError:
        print(f"{year:<6} {'N/A':<12} {'N/A':<14} {'N/A':<17} {'N/A':<19}")

print("-" * 75)
print(f"{'TOTAL':<6} {'':<12} {'':<14} ${grand_total_premium/1e6:<17.1f} ${grand_total_premium:<19,.0f}")

# Calculate some summary statistics
avg_coal_plants = np.mean([data['Coal_Plants'] for data in total_table_data])
total_plant_years = sum([data['Coal_Plants'] for data in total_table_data])

print(f"\nSummary Statistics 2007-2019:")
print(f"  Average number of coal plants per year: {avg_coal_plants:.1f}")
print(f"  Total coal plant-years in period: {total_plant_years}")
print(f"  Grand total investment premium: ${grand_total_premium/1e6:.1f}M")
print(f"  Average annual total premium: ${(grand_total_premium/len(total_table_data))/1e6:.1f}M")

# Show the range of coal plant counts across years
min_coal_year = min(total_table_data, key=lambda x: x['Coal_Plants'])
max_coal_year = max(total_table_data, key=lambda x: x['Coal_Plants'])
print(f"  Coal plant count range: {min_coal_year['Coal_Plants']} ({min_coal_year['Year']}) to {max_coal_year['Coal_Plants']} ({max_coal_year['Year']})")

# Calculate cumulative premium 2007-2019 for Natural Gas control
cumulative_ng_2007_2019 = 0
ng_table_data = []

for year in range(2007, 2020):
    event_t = year - 1995
    try:
        idx = list(event_times_ng).index(event_t)
        premium = event_coeffs_ng[idx]
        cumulative_ng_2007_2019 += premium
        ng_table_data.append({'Year': year, 'Premium': premium})
    except ValueError:
        pass

print(f"Natural Gas Control - Cumulative premium 2007-2019: ${cumulative_ng_2007_2019/1e6:.1f}M")

# Calculate cumulative premium 2007-2019 for Nuclear control
cumulative_nuke_2007_2019 = 0
nuke_table_data = []

for year in range(2007, 2020):
    event_t = year - 1995
    try:
        idx = list(event_times_nuke).index(event_t)
        premium = event_coeffs_nuke[idx]
        cumulative_nuke_2007_2019 += premium
        nuke_table_data.append({'Year': year, 'Premium': premium})
    except ValueError:
        pass

print(f"Nuclear Control - Cumulative premium 2007-2019: ${cumulative_nuke_2007_2019/1e6:.1f}M")

# Export detailed model results to text files for Word documents
print("\n" + "="*70)
print(" EXPORTING MODEL RESULTS TO TAB-SEPARATED FILES")
print("="*70)

def export_complete_model_tab(model_results, model_name, output_dir):
    """Export complete model results including key statistics in tab-separated format"""
    
    filename = f"{model_name}_complete_tab_separated.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        # Write model summary statistics first
        f.write("MODEL SUMMARY STATISTICS\n")
        f.write("Statistic\tValue\n")
        # REMOVED: f.write(f"Model Name\t{model_name}\n")
        f.write(f"Observations\t{model_results.nobs:,}\n")
        f.write(f"Entities\t{model_results.entity_info['total']}\n")
        f.write(f"Time Periods\t{model_results.time_info['total']}\n")
        f.write(f"R-squared\t{model_results.rsquared:.4f}\n")
        f.write(f"R-squared (Within)\t{model_results.rsquared_within:.4f}\n")
        f.write(f"R-squared (Between)\t{model_results.rsquared_between:.4f}\n")
        f.write(f"R-squared (Overall)\t{model_results.rsquared_overall:.4f}\n")
        f.write(f"F-statistic\t{model_results.f_statistic.stat:.4f}\n")
        f.write(f"F-statistic p-value\t{model_results.f_statistic.pval:.6f}\n")
        f.write(f"Log-likelihood\t{model_results.loglik:.2e}\n")
        f.write(f"Average Obs per Entity\t{model_results.nobs/model_results.entity_info['total']:.1f}\n")
        f.write(f"Average Obs per Time\t{model_results.nobs/model_results.time_info['total']:.1f}\n\n")
        
        # Write all coefficients table
        f.write("ALL COEFFICIENT ESTIMATES\n")
        f.write("Variable\tCoefficient\tStd Error\tt-stat\tP-value\tCI Lower\tCI Upper\n")
        
        for var in model_results.params.index:
            coeff = model_results.params[var]
            se = model_results.std_errors[var]
            tstat = model_results.tstats[var]
            pval = model_results.pvalues[var]
            ci_lower = model_results.conf_int().loc[var, 'lower']
            ci_upper = model_results.conf_int().loc[var, 'upper']
            
            f.write(f"{var}\t{coeff:.2e}\t{se:.2e}\t{tstat:.3f}\t{pval:.4f}\t{ci_lower:.2e}\t{ci_upper:.2e}\n")
        
        f.write("\n")
        
        # Write event study coefficients in millions (if applicable)
        if any('event_' in var for var in model_results.params.index):
            f.write("EVENT STUDY COEFFICIENTS (MILLIONS)\n")
            f.write("Year\tEvent_t\tCoefficient ($M)\tStd Error ($M)\tt-stat\tP-value\tCI Lower ($M)\tCI Upper ($M)\n")
            
            for var in model_results.params.index:
                if 'event_' in var:
                    event_t = int(var.split('_')[1])
                    year = 1995 + event_t
                    coeff = model_results.params[var]
                    se = model_results.std_errors[var]
                    tstat = model_results.tstats[var]
                    pval = model_results.pvalues[var]
                    ci_lower = model_results.conf_int().loc[var, 'lower']
                    ci_upper = model_results.conf_int().loc[var, 'upper']
                    
                    f.write(f"{year}\t{event_t}\t{coeff/1e6:.1f}\t{se/1e6:.1f}\t{tstat:.3f}\t{pval:.4f}\t{ci_lower/1e6:.1f}\t{ci_upper/1e6:.1f}\n")
    
    print(f"  {model_name} complete results exported to: {filepath}")
    return filepath

# Export complete results for all three models
output_file_paths = []

main_file = export_complete_model_tab(res_event, "coal_capex_event_study_table", output_dir)
output_file_paths.append(main_file)

ng_file = export_complete_model_tab(res_event_ng, "coal_capex_event_study_NGcontrol_table", output_dir)
output_file_paths.append(ng_file)

nuke_file = export_complete_model_tab(res_event_nuke, "coal_capex_event_study_NUKEcontrol_table", output_dir)
output_file_paths.append(nuke_file)

print(f"\nAll model results exported successfully!")
print(f"Files saved to: {output_dir}")
print("Tab-separated files can be easily converted to Word tables.")
print("\nExported files:")
for i, path in enumerate(output_file_paths, 1):
    print(f"  {i}. {os.path.basename(path)}")

print("\nTo use in Word:")
print("1. Open the .txt file and copy the desired table section")
print("2. Paste into Word document") 
print("3. Select the pasted text")
print("4. Go to Insert > Table > Convert Text to Table")
print("5. Choose 'Tabs' as the separator")