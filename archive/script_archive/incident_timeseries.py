# %%
import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path(__file__).parents[3] / "data" / "output"
df = pd.read_csv(output_dir / "selection_ratio_county_2012-2019_wilson.csv", dtype={"FIPS":str})
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
g = sns.relplot(data=state_df, x="year", y="incidents_100K", markers=True, facet_kws={'sharey': False, 'sharex': True}, col_order=sorted(list(df.State.unique())), col="State", kind="line", legend=False, col_wrap=7)
for ax in g.fig.axes:
    ax.set_ylabel("Incidents Per 100K")
    ax.set_ylim([0, 600])

g.fig.suptitle("Incidents Per 100K 2012-2019", fontsize=40)
g.fig.subplots_adjust(top=0.92);

plt.show()
# %%
