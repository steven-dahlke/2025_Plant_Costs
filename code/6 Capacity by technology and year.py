import os
import pandas as pd
import matplotlib

# --- Set the backend to 'Agg' ---
# This tells matplotlib to use a non-interactive backend, which is perfect for
# scripts that only save plots to a file. It avoids issues with GUI toolkits like Tkinter.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 1. Define the project root directory
# This assumes your script is in a subdirectory (e.g., 'scripts') of the project root.
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    # It assumes the notebook is in a subfolder of the project root
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")

# 2. Construct the full path to the CSV file
file_path = os.path.join(root, 'data', 'cost_data_v1.csv')
print(f"Reading file from: {file_path}")

# 3. Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path, low_memory=False)
    # Display the first 5 rows to confirm it was read correctly
    print("\nSuccessfully loaded data. Here are the first 5 rows:")
    print(df.head())

    # --- 4. Clean Specific Data Errors ---
    print("\nCleaning data...")
    initial_rows = len(df)

    # Task 1: Remove the 'greenwood' plant error
    condition_greenwood = (
            (df['plant_name_ferc1'] == 'greenwood') &
            (df['utility_id_ferc1'] == 255) &
            (df['capacity_mw_x'] == 287566)
    )
    df = df[~condition_greenwood]

    # Task 2: Correct the 'burnips' plant capacity for report_year 1999
    burnips_condition = (df['plant_name_ferc1'] == 'burnips') & (df['report_year'] == 1999)
    df.loc[burnips_condition, 'capacity_mw_x'] = 27.5
    print("Corrected 'burnips' capacity for 1999.")

    # Task 3: Delete 'hersey' and 'vestaburg' rows for report_year 1999
    hersey_condition = (df['plant_name_ferc1'] == 'hersey') & (df['report_year'] == 1999)
    vestaburg_condition = (df['plant_name_ferc1'] == 'vestaburg') & (df['report_year'] == 1999)
    df = df[~(hersey_condition | vestaburg_condition)]

    # Task 4: Correct the 'barry' plant capacity for report_year 1995
    barry_condition = (df['report_year'] == 1995) & (df['plant_name_ferc1'] == 'barry')
    df.loc[barry_condition, 'capacity_mw_x'] = 1771
    print("Corrected 'barry' capacity for 1995.")

    # Task 5: Correct the 'campbell #3' plant capacity for report_year 1999
    campbell_condition = (df['report_year'] == 1999) & (df['plant_name_ferc1'] == 'campbell #3')
    df.loc[campbell_condition, 'capacity_mw_x'] = 15.2
    print("Corrected 'campbell #3' capacity for 1999.")

    final_rows = len(df)
    print(f"Removed a total of {initial_rows - final_rows} row(s) during cleaning.")

    # --- 4.5 Remap Technology Descriptions and Reorder Columns ---
    print("\nRemapping technology descriptions and reordering columns...")

    # Define the mapping for consolidating technology descriptions
    technology_mapping = {
        'Conventional Steam Coal': 'Coal',
        'Natural Gas Fired Combined Cycle': 'Natural Gas Fired CC',
        'Natural Gas Fired Combustion Turbine': 'Natural Gas Fired CT',
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
        'Batteries': 'Batteries',
        'Conventional Hydroelectric': 'Hydroelectric',
        'Hydroelectric Pumped Storage': 'Hydroelectric',
        'Solar Thermal without Energy Storage': 'Solar Thermal',
        'Solar Thermal with Energy Storage': 'Solar Thermal'
    }

    # Apply the mapping to the 'technology_description' column
    df['technology_description'] = df['technology_description'].map(technology_mapping)
    print("Applied new technology names.")

    # Reorder columns to make 'technology_description' the third column
    # Get the current list of all column names
    cols = df.columns.tolist()
    # Remove the column from its current position
    cols.remove('technology_description')
    # Insert the column at the 3rd position (index 2)
    cols.insert(2, 'technology_description')
    # Apply the new column order to the DataFrame
    df = df[cols]
    print("Reordered columns so 'technology_description' is third.")

    # --- 5. Save the Cleaned Data ---
    # Ensure the intermediate data directory exists
    intermediate_dir = os.path.join(root, 'intermediate data')
    os.makedirs(intermediate_dir, exist_ok=True)

    # Save the cleaned DataFrame to a new CSV file
    cleaned_csv_path = os.path.join(intermediate_dir, 'cleaned_cost_data.csv')
    df.to_csv(cleaned_csv_path, index=False)
    print(f"\nCleaned data successfully saved to: {cleaned_csv_path}")

    # --- 6. Prepare Data for Plotting ---
    print("\nAggregating data for the chart...")
    # Define the columns of interest
    columns_of_interest = ['report_year', 'technology_description', 'capacity_mw_x']

    # Drop rows where any of the key columns have NaN values and create an explicit copy
    df_cleaned = df.dropna(subset=columns_of_interest).copy()

    # Convert 'report_year' to integer for cleaner axis labels
    df_cleaned['report_year'] = df_cleaned['report_year'].astype(int)

    # Group by year and technology, then sum the capacity
    capacity_data = df_cleaned.groupby(['report_year', 'technology_description'])[
        'capacity_mw_x'].sum().unstack().fillna(0)

    # Define the desired stacking order (bottom to top)
    stack_order = [
        'Coal',
        'Nuclear',
        'Natural Gas Fired CC',
        'Natural Gas Fired CT',
        'Other Natural Gas',
        'Other',
        'Hydroelectric',
        'Geothermal',
        'Onshore Wind Turbine',
        'Solar Photovoltaic',
        'Solar Thermal',
        'Batteries'
    ]

    # Filter the stack order to only include columns that are actually in the data
    existing_cols_in_order = [col for col in stack_order if col in capacity_data.columns]

    # Reorder the DataFrame columns according to the specified stack order
    capacity_data = capacity_data[existing_cols_in_order]

    # Convert y-axis units from Megawatts to Gigawatts for better readability
    capacity_data = capacity_data / 1000

    print("Data aggregation and column reordering complete.")
    print(capacity_data.head())

    # --- 7. Create and Save the Stacked Bar Chart ---
    print("\nGenerating stacked bar chart...")

    # Set the global font and increase the base font size for all text elements
    # Note: 'Cambria' must be installed on your system for this to work.
    plt.rcParams['font.family'] = 'Cambria'
    plt.rcParams.update({'font.size': 26})  # Set base font size to 26

    # Define the custom color scheme for the plot
    color_map = {
        # Fossil Fuels
        'Coal': '#3c3c3c',
        'Natural Gas Fired CC': '#A0522D',
        'Natural Gas Fired CT': '#FF8C00',
        'Other Natural Gas': '#D2B48C',

        # Baseload / Other
        'Nuclear': '#C8A2C8',
        'Other': '#a9a9a9',

        # Renewables
        'Onshore Wind Turbine': '#5D9732',
        'Solar Photovoltaic': '#FFC423',
        'Solar Thermal': '#E55B00',
        'Geothermal': '#8C5B4B',
        'Hydroelectric': '#457B9D',
        'Batteries': '#800080',
    }

    # Create the plot, passing the custom color map
    ax = capacity_data.plot(
        kind='bar',
        stacked=True,
        figsize=(16, 9),
        color=color_map,
        width=0.85
    )

    # Set the title and labels for clarity (fontsize will be inherited from rcParams)
    plt.title('')
    plt.xlabel('Year')
    plt.ylabel('Capacity (GW)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='center')

    # Selectively hide x-axis labels to prevent overcrowding
    # This keeps all the bars but only shows the label for every 3rd year.
    for i, label in enumerate(ax.get_xticklabels()):
        year = int(label.get_text())
        if year % 2 != 0:
            label.set_visible(False)

    # Get handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()

    # Reverse the order of handles and labels to invert the legend
    ax.legend(
        handles[::-1],
        labels[::-1],
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False  # Removes the grey box outline around the legend
    )

    # Adjust layout to make sure everything fits without overlapping
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.join(root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure to a file in the output directory
    chart_filename = 'figure_capacity_by_technology_and_year_backup.png'
    plt.savefig(os.path.join(output_dir, chart_filename))

    print(f"\nChart successfully saved to the 'output' folder as '{chart_filename}'")

    # --- 8. Read and Process EIA Generator Data for ALL Plots ---
    print("\n--------------------------------------------------")
    # Construct path for the Parquet file
    parquet_path = os.path.join(root, 'data', 'out_eia__yearly_generators.parquet')
    print(f"Reading Parquet file from: {parquet_path}")

    # Read the Parquet file into a new DataFrame
    df_generators = pd.read_parquet(parquet_path)

    # Extract year from the 'report_date' column
    print("Extracting year from 'report_date' column...")
    df_generators['report_year'] = pd.to_datetime(df_generators['report_date']).dt.year

    # Apply the technology mapping
    df_generators['technology_description'] = df_generators['technology_description'].map(technology_mapping)
    print("Applied technology mapping to EIA data.")

    # Prepare the EIA data for plotting
    eia_cols_interest = ['report_year', 'technology_description', 'capacity_mw']
    df_gen_cleaned = df_generators.dropna(subset=eia_cols_interest).copy()

    # Group by year and technology, then sum the capacity
    eia_capacity_data_unfiltered = df_gen_cleaned.groupby(['report_year', 'technology_description'])[
        'capacity_mw'].sum().unstack().fillna(0)

    # --- 9. Generate and Save UNFILTERED EIA Chart (All Years) ---
    print("\nGenerating UNFILTERED EIA chart with all years...")

    # Create a copy for plotting to avoid modifying the base data
    eia_plot_data = eia_capacity_data_unfiltered.copy()

    # Ensure all technologies from the stack order are present as columns
    for col in stack_order:
        if col not in eia_plot_data.columns:
            eia_plot_data[col] = 0

    # Reorder columns, convert to GW
    eia_plot_data = eia_plot_data[stack_order] / 1000

    # Generate the plot
    ax_unfiltered = eia_plot_data.plot(
        kind='bar', stacked=True, figsize=(16, 9), color=color_map, width=0.85
    )

    # Set title and labels
    ax_unfiltered.set_title('')
    ax_unfiltered.set_xlabel('Year')
    ax_unfiltered.set_ylabel('Capacity (GW)')

    # Format x-axis labels
    plt.xticks(rotation=45, ha='center')
    for i, label in enumerate(ax_unfiltered.get_xticklabels()):
        if label.get_text():
            year = int(label.get_text())
            if year % 2 != 0:
                label.set_visible(False)

    # Format legend
    handles, labels = ax_unfiltered.get_legend_handles_labels()
    ax_unfiltered.legend(
        handles[::-1], labels[::-1], bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False
    )

    # Save the figure
    plt.tight_layout()
    unfiltered_filename = 'figure_eia_capacity_by_tech_and_year_backup.png'
    plt.savefig(os.path.join(output_dir, unfiltered_filename))
    print(f"Unfiltered EIA chart successfully saved as '{unfiltered_filename}'")

    # --- 10. Generate and Save FILTERED EIA Chart (2010-2024) ---
    print("\nGenerating FILTERED EIA chart for years 2010-2024...")

    # Apply the filter to the data
    eia_capacity_data_filtered = eia_capacity_data_unfiltered[
        (eia_capacity_data_unfiltered.index >= 2010) & (eia_capacity_data_unfiltered.index <= 2024)
        ].copy()

    # Ensure all technologies are present and reorder columns
    for col in stack_order:
        if col not in eia_capacity_data_filtered.columns:
            eia_capacity_data_filtered[col] = 0

    # Reorder columns, convert to GW
    eia_capacity_data_filtered = eia_capacity_data_filtered[stack_order] / 1000

    # Generate the plot
    ax_filtered = eia_capacity_data_filtered.plot(
        kind='bar', stacked=True, figsize=(16, 9), color=color_map, width=0.85
    )

    # Set title and labels
    ax_filtered.set_title('')
    ax_filtered.set_xlabel('Year')
    ax_filtered.set_ylabel('Capacity (GW)')

    # Format x-axis labels
    plt.xticks(rotation=45, ha='center')
    for i, label in enumerate(ax_filtered.get_xticklabels()):
        if label.get_text():
            year = int(label.get_text())
            if year % 2 != 0:
                label.set_visible(False)

    # Format legend
    handles, labels = ax_filtered.get_legend_handles_labels()
    ax_filtered.legend(
        handles[::-1], labels[::-1], bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False
    )

    # Save the figure
    plt.tight_layout()
    filtered_filename = 'figure_eia_capacity_by_tech_and_year_2010_2024_backup.png'
    plt.savefig(os.path.join(output_dir, filtered_filename))
    print(f"Filtered EIA chart successfully saved as '{filtered_filename}'")


except FileNotFoundError:
    print(f"\nError: The file was not found at the specified path.")
    print("Please ensure the required data files exist inside the 'data' folder in your project root.")
except Exception as e:
    print(f"\nAn error occurred: {e}")