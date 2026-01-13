import os
import pandas as pd
import plotly.express as px

#.to_csv(os.path.join(root, 'data', 'temp.csv'))

# Define project directory
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(f"Project root: {root}")

# --- 1. Read in the CSV File ---
try:
    path = os.path.join(root, 'data', 'cost_data_v1.csv')
    df = pd.read_csv(path, low_memory=False)
    print("Successfully loaded cost_data_v1.csv")
except FileNotFoundError:
    print("Error: 'cost_data_v1.csv' not found.")
    exit()

# --- 2. Filter and clean the Data ---
df_2023 = df[df['report_year'] == 2023].copy()
print(f"Filtered data to the year 2023, resulting in {len(df_2023)} records.")

#  Filter out rows with missing capacity data
initial_rows = len(df_2023)
df_2023.dropna(subset=['capacity_mw_x'], inplace=True)
rows_removed = initial_rows - len(df_2023)
if rows_removed > 0:
    print(f"Removed {rows_removed} rows with missing 'capacity_mw_x' data.")

# --- 3. Consolidate and Rename Technology Types ---
print("Consolidating technology descriptions...")
technology_rename_map = {
    "Natural Gas Steam Turbine": "Other Natural Gas",
    "Natural Gas Internal Combustion Engine": "Other Natural Gas",
    "Petroleum Liquids": "Other",
    "All Other": "Other",
    "Petroleum Coke": "Other",
    "Wood/Wood Waste Biomass": "Other",
    "Offshore Wind Turbine": "Other",
    "Municipal Solid Waste": "Other",
    "Coal Integrated Gasification Combined Cycle": "Other"
}
df_2023['technology_description'] = df_2023['technology_description'].replace(technology_rename_map)
print("Technology descriptions have been consolidated.")

# --- 4. Aggregate duplicate plant entries ---
print("Aggregating plant data...")
initial_rows = len(df_2023)
aggregation_functions = {
    'capacity_mw_x': 'sum',
    'plant_name_ferc1': 'first'
}
df_aggregated = df_2023.groupby(
    ['latitude', 'longitude', 'technology_description'],
    as_index=False
).agg(aggregation_functions)
rows_after_agg = len(df_aggregated)
print(f"Aggregated {initial_rows} rows down to {rows_after_agg} unique plants.")


# Define the output path and export the DataFrame to a CSV file
output_path = os.path.join(root, 'data', 'cost_data_2023.csv')
df_2023.to_csv(output_path, index=False)
print(f"DataFrame successfully exported to {output_path}")

# --- 5. Define the Final Color Scheme ---
technology_color_map = {
    # Fossil Fuels
    'Conventional Steam Coal': '#3c3c3c',
    'Natural Gas Fired Combined Cycle': '#CB4335',  # Red-Orange
    'Natural Gas Fired Combustion Turbine': '#D35400', # Burnt Orange
    'Other Natural Gas': '#E67E22',          # Lighter Orange

    # Baseload / Other
    'Nuclear': '#6a0dad',                      # Purple
    'Other': '#a9a9a9',                      # Light Gray

    # Renewables
    'Conventional Hydroelectric': '#457B9D',      # Blue
    'Onshore Wind Turbine': '#5D9732',          # Green
    'Solar Photovoltaic': '#FFC423',          # Bright Yellow
    'Geothermal': '#8C564B'                       # Earthy Brown
}


# --- 6. Create the Interactive Map ---
print("Generating map...")

fig = px.scatter_geo(
    data_frame=df_aggregated,
    lat='latitude',
    lon='longitude',
    color='technology_description',
    color_discrete_map=technology_color_map,  # This applies the custom colors
    size='capacity_mw_x',
    hover_name='plant_name_ferc1',
    scope='usa',
    # title='', # Option to include a map title
    projection='albers usa'
)

# --- 7. Update layout for a cleaner look and new font ---
fig.update_layout(
    title_x=0.5,
    legend_title_text='Technology Type',
    # --- NEW: Set the font for the legend ---
    legend_font=dict(
        family="Cambria",  # Use the recommended font
        size=18,                  # Set font size
        color="black"               # Set font color
    )
)

# --- 8. Display the Map ---
fig.show()
output_map_path = os.path.join(root, 'output', 'figure 1 map.html')
fig.write_html(output_map_path)

print("Map has been generated and should open in your web browser.")



# ------------------------------------------------------------------
# --- PART 2: Map EIA Plants Not in the FERC Dataset ---
# ------------------------------------------------------------------
print("\n--- Starting Part 2: Mapping additional EIA plants ---")

# --- 1. Read in the National EIA Dataset ---
try:
    eia_path = os.path.join(root, 'data', 'eia_gen_with_ferc_mapping.csv')
    df_eia = pd.read_csv(eia_path, low_memory=False)
    print("Successfully loaded eia_gen_with_ferc_mapping.csv")
except FileNotFoundError:
    print("Error: 'eia_gen_with_ferc_mapping.csv' not found.")
    exit()

# --- 2. Aggregate the EIA Data (Initial Aggregation) ---
print("Aggregating EIA plant data...")
eia_aggregation_functions = {
    'capacity_mw': 'sum',
    'plant_name_eia': 'first'
}

# Consolidate technology types using the same map from the first part
df_eia['technology_description'] = df_eia['technology_description'].replace(technology_rename_map)

df_eia_aggregated = df_eia.groupby(
    ['latitude', 'longitude', 'technology_description'],
    as_index=False
).agg(eia_aggregation_functions)
print(f"Aggregated EIA data down to {len(df_eia_aggregated)} unique plants.")


# --- 3. Remove Plants that are in the FERC Dataset ---
print("Filtering out plants that already exist in the FERC dataset...")
ferc_locations = set(zip(df_2023['latitude'], df_2023['longitude'], df_2023['technology_description']))

initial_eia_plants = len(df_eia_aggregated)
df_eia_only = df_eia_aggregated[
    ~df_eia_aggregated.apply(lambda row: (row['latitude'], row['longitude'], row['technology_description']) in ferc_locations, axis=1)
].copy() # Use .copy() to avoid SettingWithCopyWarning
plants_removed = initial_eia_plants - len(df_eia_only)
print(f"Removed {plants_removed} EIA plants that were already mapped in the FERC data.")
print(f"Remaining plants to process and map: {len(df_eia_only)}")


# --- 4. Consolidate and Rename Additional EIA Technology Types ---
print("Consolidating additional technology descriptions for the second map...")
eia_rename_map = {
    "Other Waste Biomass": "Other",
    "Landfill Gas": "Other",
    "Other Gases": "Other",
    "Flywheels": "Other",
    "Natural Gas with Compressed Air Storage": "Other Natural Gas",
    "Solar Thermal with Energy Storage": "Solar Thermal",
    "Solar Thermal without Energy Storage": "Solar Thermal"
}
df_eia_only['technology_description'] = df_eia_only['technology_description'].replace(eia_rename_map)
print("Additional EIA technology descriptions have been consolidated.")

# --- Re-aggregate the 'df_eia_only' data after the second renaming ---
df_eia_only = df_eia_only.groupby(
    ['latitude', 'longitude', 'technology_description'],
    as_index=False
).agg(eia_aggregation_functions)
print(f"Final count of unique EIA-only plants after re-aggregation: {len(df_eia_only)}")

# ---  Merge Utility Information ---
print("Merging utility information back into the EIA-only dataset...")

# Create a mapping DataFrame from the original df_eia.
# This isolates the columns we need and removes duplicates based on the plant name,
# keeping the first utility listed for each plant.
utility_mapping = df_eia[['plant_name_eia', 'utility_id_eia',  'utility_name_eia']].drop_duplicates(
    subset='plant_name_eia',
    keep='first'
)

# Perform a left merge. This adds the 'utility_id' and 'utility_name_eia' columns
# from our mapping to the 'df_eia_only' DataFrame, matching on 'plant_name_eia'.
df_eia_only = pd.merge(
    df_eia_only,
    utility_mapping,
    on='plant_name_eia',
    how='left'
)

print(f"Successfully merged utility data. 'df_eia_only' now has {df_eia_only.shape[1]} columns.")

# --- Add Utility Entity Type ---
print("Loading and merging utility entity types...")

try:
    # Define the path to the utilities data file
    utilities_path = os.path.join(root, 'data', 'out_eia__yearly_utilities.csv')

    # Load the data, making sure to include the 'report_date' column
    df_utilities = pd.read_csv(
        utilities_path,
        usecols=['utility_id_eia', 'entity_type', 'report_date'], # Explicitly load required columns
        low_memory=False
    )

    # --- filter for the year 2024 before merging ---
    # Ensure the date column is a string, then filter rows where the date starts with '2024'
    df_utilities['report_date'] = df_utilities['report_date'].astype(str)
    df_utilities_2024 = df_utilities[df_utilities['report_date'].str.startswith('2024')].copy()

    # Drop duplicates to ensure each utility has only one entity_type for a clean merge
    df_utilities_final = df_utilities_2024[['utility_id_eia', 'entity_type']].drop_duplicates(subset='utility_id_eia')

    # Perform a left merge to add the 'entity_type' to df_eia_only
    df_eia_only = pd.merge(
        df_eia_only,
        df_utilities_final,
        on='utility_id_eia',
        how='left'
    )

    print("Successfully added 'entity_type' from 2024 data to the EIA-only dataset.")

except FileNotFoundError:
    print(f"Warning: '{os.path.basename(utilities_path)}' not found. Skipping entity type merge.")
except KeyError:
    print("Warning: Required columns ('utility_id_eia', 'entity_type', 'report_date') not found. Skipping merge.")

df_eia_only.to_csv(os.path.join(root, 'intermediate data', 'eia_gen_only_no_ferc.csv'), index=False)


# --- 5. Define the Expanded Color Scheme for the Second Map ---
technology_color_map_2 = {
    # Fossil Fuels (Consistent with Map 1)
    'Conventional Steam Coal': '#3c3c3c',
    'Natural Gas Fired Combined Cycle': '#CB4335',
    'Natural Gas Fired Combustion Turbine': '#D35400',
    'Other Natural Gas': '#E67E22',

    # Baseload / Other (Consistent with Map 1)
    'Nuclear': '#6a0dad',
    'Other': '#a9a9a9',

    # Renewables (Consistent with Map 1)
    'Conventional Hydroelectric': '#457B9D',
    'Onshore Wind Turbine': '#5D9732',
    'Solar Photovoltaic': '#FFC423',
    'Geothermal': '#8C564B',

    # --- New Categories for EIA Map ---
    'Batteries': '#20c997',                   # Teal
    'Hydroelectric Pumped Storage': '#87CEEB', # Sky Blue
    'Solar Thermal': '#FFB300'                # Deep Gold
}


# --- 6. Create the Second Map for EIA-only Plants ---
print("Generating the second map for EIA-only plants...")

fig2 = px.scatter_geo(
    data_frame=df_eia_only,
    lat='latitude',
    lon='longitude',
    color='technology_description',
    color_discrete_map=technology_color_map_2,
    size='capacity_mw',
    hover_name='plant_name_eia',
    scope='usa',
    projection='albers usa'
)

# Apply the same layout settings for a consistent look
fig2.update_layout(
    title_text='Power Plants in EIA Dataset (Not in FERC Data)',
    title_x=0.5,
    legend_title_text='Technology Type',
    legend_font=dict(
        family="Cambria",
        size=18,
        color="black"
    )
)

# --- 7. Save and Display the Second Map ---
output_map_path_2 = os.path.join(root, 'output', 'figure 3 map_eia_only.html')
fig2.write_html(output_map_path_2)
fig2.show()

print("Second map has been generated and saved as 'figure 2 map_eia_only.html'.")