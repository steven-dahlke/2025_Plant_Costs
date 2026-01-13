import os
import pandas as pd
import matplotlib # Import matplotlib
matplotlib.use('Agg') # Set the backend to Agg BEFORE importing pyplot
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Define the project root directory
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(f"Project root: {root}")

# Define the path to the data file
path = os.path.join(root, 'data', 'cost_data_v1.csv')
print(f"Looking for data at: {path}")

# Load the dataframe
try:
    df = pd.read_csv(path, low_memory=False)
    print("Successfully loaded cost_data_v1.csv")
    print(f"Columns in the DataFrame: {df.columns.tolist()}")
    print(f"Data types of columns: \n{df.dtypes}")

    # Filter data for the year 2023 and calculate plant entries for that year
    df_2023 = None # Initialize df_2023 to None
    if 'report_year' in df.columns:
        df_2023 = df[df['report_year'] == 2023].copy()

        # Keep only rows where capacity is greater than zero
        df_2023 = df_2023[df_2023['capacity_mw_x'] > 0].copy()
        # --- 4. Aggregate duplicate plant entries ---
        print("Aggregating plant data...")
        initial_rows = len(df_2023)
        aggregation_functions = {
            'capacity_mw_x': 'sum',
            'plant_name_ferc1': 'first'
        }
        df_2023 = df_2023.groupby(
            ['latitude', 'longitude', 'technology_description'],
            as_index=False
        ).agg(aggregation_functions)

        # Calculate the number of unique power plant entries for 2023
        # Assuming each row for a given year represents a unique plant entry for that year as per FERC Form 1 structure
        num_plants_2023 = len(df_2023)
        print(f"Number of unique power plant entries in FERC Form 1 for 2023: {num_plants_2023}")
    else:
        print("Error: 'report_year' column not found in the DataFrame for 2023 filtering.")

    # Calculate the total number of plant-year observations (total rows)
    total_plant_year_observations = len(df)
    print(f"Total number of plant-year observations in the dataset: {total_plant_year_observations}")

    # Calculate the earliest year in the dataset
    if 'report_year' in df.columns:
        earliest_year = df['report_year'].min()
        print(f"Earliest year in the dataset: {earliest_year}")
    else:
        print("Error: 'report_year' column not found in the DataFrame for earliest year calculation.")

    # Calculate the sum of 'capacity_mw_x' for the year 2023
    if df_2023 is not None: # Check if df_2023 was created
        if 'capacity_mw_x' in df_2023.columns:
            total_capacity_2023 = df_2023['capacity_mw_x'].sum()
            print(f"Total capacity_mw_x for 2023: {total_capacity_2023}")
        else:
            print("Error: 'capacity_mw_x' column not found in the 2023 data.")
    else:
        # This case would typically be hit if 'report_year' was not found earlier,
        # so df_2023 would not have been assigned.
        print("Skipping sum of 'capacity_mw_x' for 2023 as 2023 data could not be filtered.")

    # Calculate and display the number of observations for each year
    if 'report_year' in df.columns:
        print("\n--- Count of Observations per Year ---")
        # Use value_counts() to count occurrences of each year and sort by year (the index)
        yearly_counts = df['report_year'].value_counts().sort_index()

        # Print the counts for each year
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(yearly_counts)
        print("------------------------------------")

        # --- Code to generate and save the line chart ---
        print("\nGenerating revised line chart for publication...")
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a nice style for the plot
        plt.rcParams['font.family'] = 'Cambria'
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes, adjusted size slightly
        axis_size = 28


        # Plot the data with thicker line and larger markers
        ax.plot(yearly_counts.index, yearly_counts.values,
                marker='o',
                linestyle='-',
                linewidth=2.5,  # Increased line thickness
                markersize=8)  # Increased marker size

        # Set the title and labels for clarity with larger fonts
        #ax.set_title('Number of Plant Observations per Year (1994-2023)', fontsize=18, fontweight='bold')
        ax.set_xlabel('Year', fontsize=axis_size)
        ax.set_ylabel('Number of Observations', fontsize=axis_size)

        # --- Revised X-axis tick handling ---
        # Get the list of years from the data index
        years = yearly_counts.index
        # Set x-axis ticks to appear every 5 years for better readability
        ax.set_xticks(years[::5])
        # The rotation is likely not needed with fewer ticks, but can be kept if preferred
        plt.xticks(rotation=0)

        # Increase the font size of the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=24)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Define the output path for the saved figure
        # Saving with a new name to avoid overwriting the original
        output_path = os.path.join(root, 'output', 'observations_per_year.png')
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure to a file
        plt.savefig(output_path, dpi=300)  # Save with high resolution for publications
        print(f"Publication-ready chart saved to: {output_path}")
        # --- End of plotting code ---

    else:
        print("\nCould not calculate yearly observation counts because 'report_year' column was not found.")

except FileNotFoundError:
    print(f"Error: The file {path} was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

