import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import numpy as np


# ---Helper Function ---
def sanitize_filename(name):
    """
    Takes a string and returns a version safe for use as a filename.
    Replaces spaces with underscores and removes invalid characters.
    """
    # Replace spaces with underscores
    s = name.replace(' ', '_')
    # Remove characters that are not alphanumeric, underscores, or hyphens
    s = re.sub(r'(?u)[^-\w.]', '', s)
    # Ensure it's not empty
    return s if s else "invalid_name"


# 1. Define the project root directory
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Construct paths for data input and plot output
data_path = os.path.join(root, 'intermediate data', 'cleaned_cost_dataV2.csv')
output_dir = os.path.join(root, 'intermediate data', 'raw capital cost plots')
# --- NEW: Define output directory for capex_per_mw plots ---
output_dir_per_mw = os.path.join(root, 'intermediate data', 'raw capex per mw plots')

# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Total capital cost plots will be saved to: {output_dir}")
# --- NEW: Create the new output directory ---
os.makedirs(output_dir_per_mw, exist_ok=True)
print(f"Capex per MW plots will be saved to: {output_dir_per_mw}")

# 3. Read the CSV file and handle potential errors
try:
    #
    # Added low_memory=False argument
    df = pd.read_csv(data_path, low_memory=False)
    print("\nSuccessfully loaded the CSV file.")

    # 4. Get unique plant names
    # Changed 'plant_name' to 'plant_name_ferc1' to match the CSV file
    unique_plants = df['plant_name_ferc1'].unique()

    # ==============================================================================
    # FIRST PLOTTING ROUTINE: TOTAL CAPITAL COST
    # ==============================================================================
    print(f"\nFound {len(unique_plants)} unique plants. Generating and saving total capital cost plots...")

    # Initialize a counter for saved plots to ensure sequential numbering
    plot_counter = 0

    # 5. Loop through each unique plant name to create and save a plot
    #
    for i, plant in enumerate(unique_plants):
        # Filter the DataFrame for the current plant
        # Changed 'plant_name' to 'plant_name_ferc1'
        plant_df = df[df['plant_name_ferc1'] == plant].copy()

        # Omit years where the operational status is 'retired' ***
        plant_df = plant_df[plant_df['operational_status'] != 'retired']

        # Skip this plant if 'technology_description' has missing values
        plant_df.dropna(subset=['technology_description'], inplace=True)
        if plant_df.empty:
            continue  # Silently skip to avoid cluttering the output

        # --- AGGREGATION STEP ---
        # Group by year and sum the values for capacity and capital cost.
        # This handles plants with multiple units reported in the same year.
        plant_df_agg = plant_df.groupby('report_year').agg(
            capex_total=('capex_total', 'sum'),
            capacity_mw_x=('capacity_mw_x', 'sum')
        ).reset_index()

        # Sort original and aggregated dataframes by year
        plant_df.sort_values('report_year', inplace=True)
        plant_df_agg.sort_values('report_year', inplace=True)

        # --- Get Metadata for Title and Filename ---
        # Find the last valid capacity value from the AGGREGATED data
        capacity = np.nan
        for idx in range(len(plant_df_agg) - 1, -1, -1):
            row = plant_df_agg.iloc[idx]
            if pd.notna(row['capacity_mw_x']):
                capacity = row['capacity_mw_x']
                break  # Found a valid capacity, so we can stop searching

        # Find the last valid technology description from the ORIGINAL data
        tech_description = ""
        # Iterate backwards from the last row to the first
        for idx in range(len(plant_df) - 1, -1, -1):
            row = plant_df.iloc[idx]
            # Check if the description is a valid non-empty string
            if pd.notna(row['technology_description']) and row['technology_description'].strip():
                tech_description = row['technology_description']
                break  # Found a valid description, so we can stop searching

        # If after checking all years, we don't have capacity or tech info, skip this plant
        if pd.isna(capacity) or not tech_description:
            # This check is now silent to reduce console clutter
            # print(f"Skipping plant '{plant}' (index {i+1}) because no valid capacity or technology data was found.")
            continue

        # --- Plotting ---
        # Use plt.figure() to create a figure object
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            plant_df_agg['report_year'],
            plant_df_agg['capex_total'] / 1000000,
            marker='o',
            linestyle='-'
        )

        # Set the title and labels
        ax.set_title(
            f"{tech_description}: {plant} ({capacity:.0f} MW)"
        )
        ax.set_xlabel('Report Year')
        ax.set_ylabel('Total Capital Cost (Millions of $)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Ensure x-axis ticks are displayed as integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig.tight_layout()

        # --- Sanitize names and save the plot ---
        # Sanitize components for the filename
        safe_tech = sanitize_filename(tech_description)
        safe_plant_name = sanitize_filename(plant)

        # Create a subdirectory for the technology type if it doesn't exist
        tech_output_dir = os.path.join(output_dir, safe_tech)
        os.makedirs(tech_output_dir, exist_ok=True)

        # Increment the counter for each plot that is actually created
        plot_counter += 1

        # Create the filename using the sequential plot_counter
        # The int(capacity) conversion is now safe because we've ensured it's not NaN
        filename = f"{plot_counter:03d}_{safe_tech}_{int(capacity)}MW_{safe_plant_name}.png"
        output_path = os.path.join(tech_output_dir, filename)

        # Save the figure
        fig.savefig(output_path)
        # print(f"Saved: {filename} to {safe_tech}") # Verbose printing

        # Close the plot to free up memory
        plt.close(fig)

    print("\nProcessing for total capital cost plots complete.")

    # ==============================================================================
    # SECOND PLOTTING ROUTINE: WEIGHTED AVERAGE CAPEX PER MW
    # ==============================================================================
    print(f"\nGenerating and saving weighted average capex per MW plots...")

    # Reset plot counter for the new set of plots
    plot_counter = 0

    for i, plant in enumerate(unique_plants):
        # Filter the DataFrame for the current plant
        plant_df = df[df['plant_name_ferc1'] == plant].copy()

        # Omit years where the operational status is 'retired'
        plant_df = plant_df[plant_df['operational_status'] != 'retired']

        # Skip plant if essential data is missing
        plant_df.dropna(subset=['technology_description', 'capex_per_mw', 'capacity_mw_x'], inplace=True)
        if plant_df.empty:
            continue

        # --- AGGREGATION STEP for CAPEX PER MW ---
        # This calculates a weighted average of capex_per_mw, weighted by capacity_mw_x.
        plant_df['capex_times_capacity'] = plant_df['capex_per_mw'] * plant_df['capacity_mw_x']

        plant_df_agg = plant_df.groupby('report_year').agg(
            capex_times_capacity=('capex_times_capacity', 'sum'),
            capacity_mw_x=('capacity_mw_x', 'sum')
        ).reset_index()

        # Calculate the final weighted average
        plant_df_agg['capex_per_mw_weighted'] = plant_df_agg['capex_times_capacity'] / plant_df_agg['capacity_mw_x']

        # Sort original and aggregated dataframes by year
        plant_df.sort_values('report_year', inplace=True)
        plant_df_agg.sort_values('report_year', inplace=True)

        # --- Get Metadata for Title and Filename (re-using logic from above) ---
        capacity = np.nan
        for idx in range(len(plant_df_agg) - 1, -1, -1):
            row = plant_df_agg.iloc[idx]
            if pd.notna(row['capacity_mw_x']):
                capacity = row['capacity_mw_x']
                break

        tech_description = ""
        for idx in range(len(plant_df) - 1, -1, -1):
            row = plant_df.iloc[idx]
            if pd.notna(row['technology_description']) and row['technology_description'].strip():
                tech_description = row['technology_description']
                break

        if pd.isna(capacity) or not tech_description:
            continue

        # --- NEW Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            plant_df_agg['report_year'],
            plant_df_agg['capex_per_mw_weighted'],
            marker='o',
            linestyle='-'
        )

        ax.set_title(
            f"{tech_description}: {plant} ({capacity:.0f} MW)"
        )
        ax.set_xlabel('Report Year')
        ax.set_ylabel('Weighted Avg. Capex per MW ($/MW)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Format y-axis labels with commas to avoid scientific notation
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig.tight_layout()

        # --- Sanitize names and save the plot in the new directory ---
        safe_tech = sanitize_filename(tech_description)
        safe_plant_name = sanitize_filename(plant)

        # Use the new output directory
        tech_output_dir = os.path.join(output_dir_per_mw, safe_tech)
        os.makedirs(tech_output_dir, exist_ok=True)

        plot_counter += 1

        filename = f"{plot_counter:03d}_{safe_tech}_{int(capacity)}MW_{safe_plant_name}.png"
        output_path = os.path.join(tech_output_dir, filename)

        fig.savefig(output_path)
        # print(f"Saved: {filename} to {safe_tech}") # Verbose printing

        plt.close(fig)

    print("\nProcessing for capex per MW plots complete.")


except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}")
except KeyError as e:
    print(f"Error: A required column is missing from the CSV file: {e}")

