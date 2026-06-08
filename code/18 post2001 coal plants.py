import os
import re
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def display_name(name):
    """Convert raw plant name to clean display label for plot legends.
    Strips deduplication artifacts (trailing _YYYY or _YYYY_NNN suffixes
    added by script 12), then applies title case and fixes Roman numerals.
    """
    # Strip trailing year-based deduplication tags (e.g. _2003, _2011_209)
    name = re.sub(r'_\d{4}(_\d+)*$', '', name)
    name = name.replace('_', ' ')          # remaining underscores → spaces
    name = name.title()                     # title case
    # Fix Roman numerals that title() lowercases (e.g. 'Iii' → 'III')
    name = re.sub(
        r'\b(Xi{0,3}|Ix|Vi{0,3}|Iv|I{1,3})\b',
        lambda m: m.group().upper(),
        name,
        flags=re.IGNORECASE
    )
    return name

# 1. Define the project root directory
# -----------------------------------------------------------------------------
try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Load the model-ready dataset (same as script 13)
# -----------------------------------------------------------------------------
data_path = os.path.join(root, "intermediate data", "capex_econometric", "df_model_ready.csv")
print(f"Reading data from: {data_path}")

df = pd.read_csv(data_path, low_memory=False)
print(f"Data loaded. Shape: {df.shape}")

# Ensure numeric types
df['installation_year'] = pd.to_numeric(df['installation_year'], errors='coerce')
df['report_year']       = pd.to_numeric(df['report_year'],       errors='coerce')
df['capacity_mw_x']     = pd.to_numeric(df['capacity_mw_x'],     errors='coerce')
df['capex_total']       = pd.to_numeric(df['capex_total'],        errors='coerce')

# 3. Identify post-2001 coal plants
#    Mirrors the treatment-group definition in script 13:
#    treatment group = Coal, capacity >= 25 MW, installation_year <= 2000
#    post-2001 coal = Coal, installation_year > 2000
# -----------------------------------------------------------------------------
coal_df = df[df['technology_description'] == 'Coal'].copy()

post2001_coal = coal_df[coal_df['installation_year'] > 2000].copy()
pre2001_coal  = coal_df[coal_df['installation_year'] <= 2000].copy()

# 4. Summary table: one row per post-2001 coal plant
# -----------------------------------------------------------------------------
# Use the most recent observation per plant for stable capacity/install-year values
plant_summary = (
    post2001_coal
    .sort_values('report_year')
    .groupby('plant_name_ferc1')
    .agg(
        installation_year = ('installation_year', 'first'),
        capacity_mw       = ('capacity_mw_x',     'first'),
        years_in_sample   = ('report_year',        'count'),
        first_year        = ('report_year',         'min'),
        last_year         = ('report_year',          'max'),
    )
    .reset_index()
    .sort_values('installation_year')
)

print("\n" + "="*70)
print(" POST-2001 COAL PLANTS IN ECONOMETRIC SAMPLE")
print("="*70)
print(f"{'Plant Name':<40} {'Install Yr':>10} {'Cap (MW)':>10} {'Yrs':>5} {'Range':>12}")
print("-"*70)
for _, row in plant_summary.iterrows():
    print(
        f"{row['plant_name_ferc1']:<40} "
        f"{int(row['installation_year']):>10} "
        f"{row['capacity_mw']:>10.0f} "
        f"{int(row['years_in_sample']):>5} "
        f"{int(row['first_year'])}-{int(row['last_year']):>4}"
    )
print(f"\nTotal post-2001 coal plants: {len(plant_summary)}")
print(f"Total pre-2001  coal plants: {pre2001_coal['plant_name_ferc1'].nunique()}")

# Save summary table to CSV
output_dir = os.path.join(root, "output")
os.makedirs(output_dir, exist_ok=True)

summary_path = os.path.join(root, "intermediate data", "post2001_coal.csv")
plant_summary.to_csv(summary_path, index=False)
print(f"\nSummary table saved to: {summary_path}")

# 5. Plot: individual capital balance trajectories for post-2001 coal plants
# -----------------------------------------------------------------------------
FONT_SIZE  = 34
LINE_WIDTH = 3
POINT_SIZE = 7

plt.rcParams['font.family'] = 'Cambria'

post2001_plants = plant_summary['plant_name_ferc1'].tolist()
n_plants = len(post2001_plants)

if n_plants == 0:
    print("\nNo post-2001 coal plants found in sample — skipping plots.")
else:
    fig, ax = plt.subplots(figsize=(14, 8))

    for plant in post2001_plants:
        pdata = post2001_coal[post2001_coal['plant_name_ferc1'] == plant].sort_values('report_year')
        ax.plot(
            pdata['report_year'],
            pdata['capex_total'] / 1e9,
            'o-',
            linewidth=LINE_WIDTH,
            markersize=POINT_SIZE,
            label=display_name(plant)
        )

    ax.set_xlabel('Year',                        fontsize=FONT_SIZE)
    ax.set_ylabel('Capital Balance ($ Billions)', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=FONT_SIZE - 4, frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "post2001_coal_individual_trajectories.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"\nIndividual trajectories plot saved to: {plot1_path}")
    plt.close()

# 6. Comparison plot: pre-2001 vs post-2001 coal plant average capital trajectories
#    Normalized to each group's 2001 value to put both on the same scale and
#    make the presence/absence of investment waves directly visible.
# -----------------------------------------------------------------------------

# Compute capacity-weighted average capex_total by year for each group
def weighted_avg_by_year(group_df):
    return (
        group_df
        .groupby('report_year')[['capex_total', 'capacity_mw_x']]
        .apply(lambda g: np.average(g['capex_total'], weights=g['capacity_mw_x']),
               include_groups=False)
        .reset_index(name='wtd_avg_capex')
    )

pre_avg  = weighted_avg_by_year(pre2001_coal)
post_avg = weighted_avg_by_year(post2001_coal) if n_plants > 0 else pd.DataFrame()

# Normalize both series to their 2001 value (first common year)
def normalize_to_year(series_df, base_year, capex_col='wtd_avg_capex'):
    base_row = series_df[series_df['report_year'] == base_year]
    if base_row.empty:
        # Fall back to earliest available year
        base_row = series_df.sort_values('report_year').iloc[[0]]
    base_val = base_row[capex_col].values[0]
    out = series_df.copy()
    out['normalized'] = out[capex_col] / base_val
    return out

base_year = 2001
pre_norm  = normalize_to_year(pre_avg,  base_year)
if not post_avg.empty:
    post_norm = normalize_to_year(post_avg, base_year)

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(
    pre_norm['report_year'],
    pre_norm['normalized'],
    'o-',
    color='#3c3c3c',
    linewidth=LINE_WIDTH + 1,
    markersize=POINT_SIZE,
    label=f'Pre-2001 coal plants (n={pre2001_coal["plant_name_ferc1"].nunique()})'
)

if not post_avg.empty:
    ax.plot(
        post_norm['report_year'],
        post_norm['normalized'],
        's--',
        color='#C0392B',
        linewidth=LINE_WIDTH,
        markersize=POINT_SIZE,
        label=f'Post-2001 coal plants (n={n_plants})'
    )

# Reference line at 1.0 (= 2001 level)
ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.5)

ax.set_xlabel('Year',                                          fontsize=FONT_SIZE)
ax.set_ylabel(f'Normalized Capital Balance ({base_year} = 1)', fontsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
ax.legend(loc='upper left', fontsize=FONT_SIZE - 4, frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(output_dir, "pre_vs_post2001_coal_comparison.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"Pre vs. post-2001 comparison plot saved to: {plot2_path}")
plt.close()

# =============================================================================
# 7. ROBUSTNESS CHECK: Main model re-run with post-2001 coal plants included
#    in the treatment group.
#
#    Main model treatment:  is_coal_ge25mw_pre2001  (Coal, >=25 MW, installed <=2000)
#    This robustness check: is_coal_ge25mw_allvintages (Coal, >=25 MW, any vintage)
#    All other specification choices are identical to script 13.
# =============================================================================

print("\n\n" + "="*80)
print(" ROBUSTNESS CHECK: All-Vintage Coal Treatment Group")
print(" (post-2001 coal plants included in treatment)")
print("="*80)

from linearmodels.panel import PanelOLS

# Set up panel data — mirrors script 13 exactly
df_rc = df.copy()
df_rc = df_rc.set_index(['plant_name_ferc1', 'report_year'], inplace=False)
df_rc['report_year']    = df_rc.index.get_level_values('report_year')

# First differences (investment flows)
df_rc['capex_flow']      = df_rc.groupby(level=0)['capex_total'].diff()
df_rc['d_capacity_mw_x'] = df_rc.groupby(level=0)['capacity_mw_x'].diff()

# Drop first year per plant (no lag to diff)
df_flow_rc = df_rc.dropna(subset=['capex_flow']).copy()
print(f"Flow sample: {df_flow_rc.shape[0]} rows, "
      f"{df_flow_rc.index.get_level_values(0).nunique()} plants")

# New treatment indicator: coal, >=25 MW, ALL vintages
df_flow_rc['is_coal_ge25mw_allvintages'] = (
    (df_flow_rc['technology_description'] == 'Coal') &
    (df_flow_rc['capacity_mw_x'] >= 25)
).astype(int)

n_treated_main = int(df_flow_rc['is_coal_ge25mw_pre2001'].max())  # sanity check col exists
n_treated_new  = df_flow_rc[df_flow_rc['is_coal_ge25mw_allvintages'] == 1] \
                     .index.get_level_values(0).nunique()
n_treated_old  = df_flow_rc[df_flow_rc['is_coal_ge25mw_pre2001'] == 1] \
                     .index.get_level_values(0).nunique()

print(f"Treatment plants — main model (pre-2001):    {n_treated_old}")
print(f"Treatment plants — robustness (all vintage): {n_treated_new}")
print(f"Additional plants added to treatment:        {n_treated_new - n_treated_old}")

# Event study variables — identical structure to script 13
df_flow_rc['event_time'] = df_flow_rc['report_year'] - 1995
event_years = range(1, 29)   # 1996–2023; 1995 omitted as baseline
available_years = sorted(df_flow_rc['report_year'].unique())

valid_event_years_rc = []
for t in event_years:
    year = 1995 + t
    if year in available_years:
        var_name = f'event_{t}'
        df_flow_rc[var_name] = np.where(
            (df_flow_rc['is_coal_ge25mw_allvintages'] == 1) &
            (df_flow_rc['event_time'] == t), 1, 0
        )
        valid_event_years_rc.append(t)

event_vars_rc = [f'event_{t}' for t in valid_event_years_rc]

event_formula_rc = f"""
capex_flow ~ {' + '.join(event_vars_rc)}
             + d_capacity_mw_x
"""

print(f"\nRunning robustness model ({len(event_vars_rc)} event indicators, 1995 baseline)...")
model_rc  = PanelOLS.from_formula(event_formula_rc, data=df_flow_rc)
res_rc    = model_rc.fit(cov_type='clustered', cluster_entity=True)
print(res_rc)

# Extract coefficients
rc_coeffs = np.array([res_rc.params[f'event_{t}']     for t in valid_event_years_rc
                       if f'event_{t}' in res_rc.params.index])
rc_ses    = np.array([res_rc.std_errors[f'event_{t}'] for t in valid_event_years_rc
                       if f'event_{t}' in res_rc.params.index])
rc_times  = np.array(valid_event_years_rc)
rc_years  = 1995 + rc_times

rc_ci_lower = rc_coeffs - 1.96 * rc_ses
rc_ci_upper = rc_coeffs + 1.96 * rc_ses

# Cumulative premium 2007-2019 for comparison
rc_premium_2007_2019 = sum(
    rc_coeffs[i] for i, t in enumerate(rc_times) if 12 <= t <= 24
)
print(f"\nCumulative coal investment premium 2007-2019 (all-vintage treatment): "
      f"${rc_premium_2007_2019/1e6:.1f}M per plant")

# ---- Load main-model coefficients from script 13 results for overlay --------
# Re-run main model on the same flow dataset so both series are on equal footing
print("\nRe-running main model on same dataset for direct comparison...")

for t in valid_event_years_rc:
    var_name = f'event_main_{t}'
    df_flow_rc[var_name] = np.where(
        (df_flow_rc['is_coal_ge25mw_pre2001'] == 1) &
        (df_flow_rc['event_time'] == t), 1, 0
    )

event_vars_main = [f'event_main_{t}' for t in valid_event_years_rc]
event_formula_main = f"capex_flow ~ {' + '.join(event_vars_main)} + d_capacity_mw_x"

model_main_rc = PanelOLS.from_formula(event_formula_main, data=df_flow_rc)
res_main_rc   = model_main_rc.fit(cov_type='clustered', cluster_entity=True)

main_coeffs = np.array([res_main_rc.params[f'event_main_{t}'] for t in valid_event_years_rc
                         if f'event_main_{t}' in res_main_rc.params.index])
main_ses    = np.array([res_main_rc.std_errors[f'event_main_{t}'] for t in valid_event_years_rc
                         if f'event_main_{t}' in res_main_rc.params.index])

main_premium_2007_2019 = sum(
    main_coeffs[i] for i, t in enumerate(rc_times) if 12 <= t <= 24
)
print(f"Cumulative coal investment premium 2007-2019 (main model, pre-2001 only): "
      f"${main_premium_2007_2019/1e6:.1f}M per plant")
print(f"Difference: ${(rc_premium_2007_2019 - main_premium_2007_2019)/1e6:.1f}M")

# ---- Comparison plot: main model vs. all-vintage robustness check -----------
FONT_SIZE  = 26
LINE_WIDTH = 5
POINT_SIZE = 12
plt.rcParams['font.family'] = 'Cambria'

fig, ax = plt.subplots(figsize=(14, 8))

# Main model (pre-2001 treatment)
ax.plot(rc_years, main_coeffs / 1e6, 'o-',
        color='darkblue', linewidth=LINE_WIDTH, markersize=POINT_SIZE,
        label='Main model: pre-2001 coal only')
ax.fill_between(rc_years,
                (main_coeffs - 1.96 * main_ses) / 1e6,
                (main_coeffs + 1.96 * main_ses) / 1e6,
                alpha=0.15, color='darkblue')

# Robustness check (all-vintage treatment)
ax.plot(rc_years, rc_coeffs / 1e6, 's--',
        color='#C0392B', linewidth=LINE_WIDTH - 1, markersize=POINT_SIZE - 2,
        label='Robustness: all coal vintages (incl. post-2001)')
ax.fill_between(rc_years,
                rc_ci_lower / 1e6,
                rc_ci_upper / 1e6,
                alpha=0.15, color='#C0392B')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Year',      fontsize=FONT_SIZE, fontweight='bold')
ax.set_ylabel('$ Millions', fontsize=FONT_SIZE, fontweight='bold')
ax.set_xticks(range(1995, 2024, 3))
ax.set_xticklabels(range(1995, 2024, 3), rotation=45, fontsize=FONT_SIZE)
ax.tick_params(axis='y', labelsize=FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', frameon=True, fancybox=True,
          shadow=False, fontsize=FONT_SIZE - 4, framealpha=0.6)

plt.tight_layout()
rc_plot_path = os.path.join(output_dir, "coal_capex_RC_allvintage_vs_main.png")
plt.savefig(rc_plot_path, dpi=300, bbox_inches='tight')
print(f"\nComparison plot saved to: {rc_plot_path}")
plt.close()

# ---- Export robustness check results to tab-separated file ------------------
rc_results_path = os.path.join(output_dir,
    "coal_capex_RC_allvintage_complete_tab_separated.txt")

with open(rc_results_path, 'w') as f:
    f.write("MODEL SUMMARY STATISTICS\n")
    f.write("Statistic\tValue\n")
    f.write(f"Observations\t{res_rc.nobs:,}\n")
    f.write(f"Entities\t{res_rc.entity_info['total']}\n")
    f.write(f"Time Periods\t{res_rc.time_info['total']}\n")
    f.write(f"R-squared\t{res_rc.rsquared:.4f}\n")
    f.write(f"R-squared (Within)\t{res_rc.rsquared_within:.4f}\n")
    f.write(f"F-statistic\t{res_rc.f_statistic.stat:.4f}\n")
    f.write(f"F-statistic p-value\t{res_rc.f_statistic.pval:.6f}\n\n")

    f.write("EVENT STUDY COEFFICIENTS (MILLIONS)\n")
    f.write("Year\tEvent_t\tCoefficient ($M)\tStd Error ($M)\t"
            "t-stat\tP-value\tCI Lower ($M)\tCI Upper ($M)\n")
    for i, t in enumerate(rc_times):
        year   = 1995 + t
        coeff  = res_rc.params[f'event_{t}']
        se     = res_rc.std_errors[f'event_{t}']
        tstat  = res_rc.tstats[f'event_{t}']
        pval   = res_rc.pvalues[f'event_{t}']
        cil    = res_rc.conf_int().loc[f'event_{t}', 'lower']
        ciu    = res_rc.conf_int().loc[f'event_{t}', 'upper']
        f.write(f"{year}\t{t}\t{coeff/1e6:.1f}\t{se/1e6:.1f}\t"
                f"{tstat:.3f}\t{pval:.4f}\t{cil/1e6:.1f}\t{ciu/1e6:.1f}\n")

print(f"Robustness check results exported to: {rc_results_path}")

print("\nScript 18 complete.")
