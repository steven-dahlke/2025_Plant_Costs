import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
print(f"Loading data from: {data_path}")

df = pd.read_csv(data_path, low_memory=False)
print(f"Data loaded successfully. Shape: {df.shape}")

# 3. Create faceted plots for selected plants
# -----------------------------------------------------------------------------

# Plot formatting parameters
FONT_SIZE = 24      # Controls all text sizes
LINE_SIZE = 6       # Controls line thickness  
BULLET_SIZE = 10     # Controls marker/point size

# Set font to Cambria
plt.rcParams['font.family'] = 'Cambria'

# Define the plant names to plot (exact case-sensitive matching)
plant_names = ['boswell', 'cayuga', 'dave johnston', 'pawnee']

# First pass: find the maximum capex value across all plants to set consistent y-axis scale
max_capex_billions = 0
for plant_name in plant_names:
    plant_data = df[df['plant_name_ferc1'] == plant_name]
    plant_data = plant_data[(plant_data['report_year'] >= 2001) & (plant_data['report_year'] <= 2023)]
    if len(plant_data) > 0:
        plant_max = (plant_data['capex_total'] / 1e9).max()
        max_capex_billions = max(max_capex_billions, plant_max)

# Add small buffer to max value for better visualization
y_max = max_capex_billions * 1.05

print(f"Setting common y-axis scale: 0 to ${y_max:.2f}B")

# Set up the plot with 2x2 grid (4 plants)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()  # Flatten to make indexing easier

# Create a plot for each plant
for i, plant_name in enumerate(plant_names):
    ax = axes[i]
    
    # Filter data for this plant (exact case-sensitive match)
    plant_data = df[df['plant_name_ferc1'] == plant_name]
    # Further filter for years 2001-2023
    plant_data = plant_data[(plant_data['report_year'] >= 2001) & (plant_data['report_year'] <= 2023)]
    
    if len(plant_data) > 0:
        # Sort by year for cleaner line plot
        plant_data = plant_data.sort_values('report_year')
        
        # Convert to billions for y-axis
        capex_billions = plant_data['capex_total'] / 1e9
        
        # Plot capex_total vs report_year
        ax.plot(plant_data['report_year'], capex_billions, 
                'o-', linewidth=LINE_SIZE, markersize=BULLET_SIZE, color='#1f2937')
        
        # Formatting
        ax.set_title(f'{plant_name.title()}', fontsize=FONT_SIZE, fontweight='bold')
        # Remove individual y-axis labels - we'll add one common label below
        ax.grid(True, alpha=0.3)
        
        # Set axis limits - NOW WITH COMMON Y-SCALE
        ax.set_xlim(2000, 2024)  # Extended to 2024 to show full 2023 data point
        ax.set_ylim(0, y_max)  # Use common y-axis scale
        
        # Set font sizes for tick labels
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        
        # Remove the 0.0 from y-axis to avoid overlap with 2000 on x-axis
        y_ticks = ax.get_yticks()
        y_ticks = y_ticks[y_ticks > 0]  # Remove 0 and any negative values
        ax.set_yticks(y_ticks)
        
    else:
        # If no data found for this plant
        ax.text(0.5, 0.5, f'No data found for\n{plant_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=FONT_SIZE)
        ax.set_title(f'{plant_name.title()} (No Data)', fontsize=FONT_SIZE, fontweight='bold')
        ax.set_xlim(2000, 2024)  # Also extended to 2024 for consistency
        ax.set_ylim(0, y_max)  # Use common y-axis scale even for no-data plots
        
        # Remove the 0.0 from y-axis for no-data plots too
        y_ticks = ax.get_yticks()
        y_ticks = y_ticks[y_ticks > 0]
        ax.set_yticks(y_ticks)

# Add a single y-axis label for the entire figure, centered on the left
fig.text(0.04, 0.5, 'Capital Balance ($ Billions)', va='center', rotation='vertical', 
         fontsize=FONT_SIZE, fontweight='bold')

# Adjust layout to prevent overlap and leave space for the common y-label
plt.tight_layout()
plt.subplots_adjust(left=0.12)  # Add space on left for the common y-axis label

# Save the plot
output_path = os.path.join(root, "output", "capex plots for paper.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plots saved to: {output_path}")

# Show the plot
# plt.show()

