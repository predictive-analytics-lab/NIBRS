# %%

import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path(__file__).parents[2] / "data" / "output"
df = pd.read_csv(output_dir / "selection_ratio_county_2010-2019_bootstraps_1000_all.csv", dtype={"FIPS":str})
sr_df = pd.read_csv(output_dir.parent / "misc" / "subregion_counties.csv", dtype={"FIPS": str}, usecols=["FIPS", "State", "County"])
df = df.merge(sr_df, on="FIPS", how="left")
coverage_df = pd.read_csv(output_dir.parent / "misc" / "county_coverage.csv", dtype={"FIPS":str})
df = df.merge(coverage_df, on=["FIPS", "year"], how="left")

df_temp = pd.DataFrame()
for state in df.State.unique():
    if len(df[df.State == state].year.unique()) >= 3 and len(df[df.State == state].FIPS.unique()) > 1:
        df_temp = df_temp.append(df[df.State == state])
df = df_temp

state_df = df.groupby(["State", "year"]).agg({"incidents": "sum",
                                            "population": "sum"})

state_df["incidents_100K"] = (state_df.incidents / state_df.population) * 100_000
state_df["incidents_100K"] = np.nan_to_num(state_df.incidents_100K, posinf=np.nan)
state_df = state_df.reset_index()
state_df["year"] = state_df.year.astype(str)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


misc_dir = Path(__file__).parents[2] / "data" / "misc"

tile_grid = pd.read_csv(misc_dir / "us_tile_grid.csv")
tile_grid = tile_grid.rename(columns={"name": "State"})
tile_grid_2 = tile_grid.merge(state_df, on="State", how="right")

row_max = tile_grid.row.max()
col_max = tile_grid.column.max()

# tile_grid = tile_grid.set_index(['row', 'column'])

fig, axs = plt.subplots(row_max + 1, tile_grid.column.max() + 1)

for i in range(row_max + 1):
    for j in range(col_max + 1):
        df = tile_grid[(tile_grid.row == i) & (tile_grid.column == j)]
        df2 = tile_grid_2[(tile_grid_2.row == i) & (tile_grid_2.column == j)]
        if len(df) > 0:
            ax = axs[row_max - i, j]
            # ax.text(0.5, 0.5, df.iloc[0].code)
            if len(df2) > 0:
                ax.plot(df2.year, df2.incidents_100K)
            else:
                ax.set_facecolor("black")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.set_ylim([0, 600])
        else:
            axs[row_max - i, j].axis('off')

plt.title("Incidents per 100k USA")
plt.show()
# %%
