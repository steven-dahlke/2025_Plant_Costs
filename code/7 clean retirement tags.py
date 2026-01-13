import os
import pandas as pd
import io

# 1. Define the project root directory
try:
    # This works when running the script from a subfolder (e.g., project_root/scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This is a fallback for interactive environments like Jupyter notebooks
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

print(f"Project root: {root}")


data_path = os.path.join(root, 'intermediate data', 'cleaned_cost_data.csv')
df = pd.read_csv(data_path, low_memory=False)
#%%
# Create function to accomplish data cleaning task
def update_operational_status(df, plant_name, retirement_year):
    mask = (df['plant_name_ferc1'] == plant_name) & (df['report_year'] < retirement_year)
    df.loc[mask, 'operational_status'] = 'existing'
    return df

# Clean plants with retirement tags
df = update_operational_status(df,'*dolet hills (3)', 2022)
df = update_operational_status(df,'*pirkey (2)', 2024)
df = update_operational_status(df,'a.b. brown station', 2024)
df = update_operational_status(df,'baxter wilson', 2023)
df = update_operational_status(df,'boardman', 2021)
df = update_operational_status(df,'buck', 2021)
df = update_operational_status(df,'buzzard roost', 2013)
df = update_operational_status(df,'canadys', 2014)
df = update_operational_status(df,'carlsbad', 2019)
df = update_operational_status(df,'colstrip 3 & 4', 2024)
df = update_operational_status(df,'comanche PSCo', 2024)
df = update_operational_status(df,'connersville', 2019)
df = update_operational_status(df,'dan river', 2021)
df = update_operational_status(df,'eaton', 2015)
df = update_operational_status(df,'edwardsport', 2012)
df = update_operational_status(df,'four corners (1)', 2024)
df = update_operational_status(df,'gadsden', 2023)
df = update_operational_status(df,'gallagher', 2021)
df = update_operational_status(df,'great bend', 2010)
df = update_operational_status(df,'hardeeville peaking', 2016)
df = update_operational_status(df,'hudson ave gt 3,4 & 5', 2024)
df = update_operational_status(df,'karn 1 & 2', 2024)
df = update_operational_status(df,'lansing #4', 2024)
df = update_operational_status(df,'lee steam', 2023)
df = update_operational_status(df,'little gypsy 2 & 3', 2024)
df = update_operational_status(df,'meramec', 2024)
df = update_operational_status(df,'meramec ct', 2020)
df = update_operational_status(df,'miami fort 6', 2016)
df = update_operational_status(df,'miami wabash', 2019)
df = update_operational_status(df,'montrose', 2019)
df = update_operational_status(df,'moore county', 2014)
df = update_operational_status(df,'morehead', 2013)
df = update_operational_status(df,'parr #1 & #2', 2024)
df = update_operational_status(df,'parr #3 & #4', 2024)
df = update_operational_status(df,'pueblo airport generating station- unit 6', 2024)
df = update_operational_status(df,'riverbend', 2014)
df = update_operational_status(df,'sibley', 2019)
df = update_operational_status(df,'slocum peaker', 2024)
df = update_operational_status(df,'taconite harbor', 2017)
df = update_operational_status(df,'valmont 5', 2018)
df = update_operational_status(df,'wabash river', 2017)
df = update_operational_status(df,'weston w31, w32', 2024)
df = update_operational_status(df,'williams #1 peaking', 2020)
df = update_operational_status(df,'williams #2 peaking', 2023)





#df = update_operational_status(df,'', )

df.to_csv(os.path.join(root, 'intermediate data', 'cleaned_cost_dataV2.csv'),index=False)