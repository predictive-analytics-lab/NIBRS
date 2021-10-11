# %%
import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path(__file__).parents[3] / "data" / "output"
df = pd.read_csv(output_dir / "selection_ratio_county_2012-2019_wilson_urban.csv", dtype={"FIPS":str})
sr_df = pd.read_csv(output_dir.parent / "misc" / "subregion_counties.csv", dtype={"FIPS": str}, usecols=["FIPS", "State", "County"])
df = df.merge(sr_df, on="FIPS", how="left")


def weighted_mean(x, **kws):
    val, weight = map(np.asarray, zip(*x))
    weight = np.nan_to_num(weight)
    return (val * weight).sum() / weight.sum()

df_temp = pd.DataFrame()

for state in df.State.unique():
    if len(df[df.State == state].year.unique()) >= 3 and len(df[df.State == state].FIPS.unique()) > 1:
        df_temp = df_temp.append(df[df.State == state])
df = df_temp
        

df["sr_and_ci"] = list(zip(df["selection_ratio"], 1 / df["ci"]))

# election_df = pd.read_csv(output_dir.parent / "misc" / "election_results_x_county.csv", dtype={"FIPS": str})
# election_df = election_df.rename(columns={"state": "ABBRV"})

# fips_abbrv = pd.read_csv(output_dir.parent / "misc" / "FIPS_ABBRV.csv", dtype={"FIPS": str})
# fips_abbrv = fips_abbrv.rename(columns={"STATE": "State"})

# election_df = election_df.merge(fips_abbrv, on="ABBRV", how="left")

# wm = lambda x: np.average(x, weights=election_df.loc[x.index, "total_votes"])

# new_df = election_df.groupby(["State", "year"]).perc_republican_votes.agg(wm).reset_index()

# most_republican_states = new_df[new_df.year == 2020].sort_values(by="perc_republican_votes", ascending=False)[-10:].State.to_list()


# df = df[df.State.isin(most_republican_states)]
# %%

g = sns.relplot(data=df, x="year", y="sr_and_ci", estimator=weighted_mean,markers=True, facet_kws={'sharey': False, 'sharex': True}, col="State", kind="line", legend=False, col_wrap=7)
for ax in g.fig.axes:
    ax.invert_yaxis()
    ax.set_ylabel("Selection Ratio")
plt.show()
# %%
sns.set(rc={'figure.figsize':(16.7,8.27), "lines.linewidth":3})
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

ax = sns.lineplot(data=df, x="year", y="sr_and_ci", estimator=weighted_mean,err_kws={"alpha": 0.1}, markers=True, hue="State")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=6)
plt.ylabel("Selection Ratio")
plt.ylim([0, 10])
plt.xlabel("Year")
plt.show()

# %%

dfm = df.groupby(["State","year"]).incidents.sum().to_frame("incidents").reset_index()

ax = sns.relplot(data=dfm, x="year", y="incidents",err_kws={"alpha": 0.1}, facet_kws={'sharey': False, 'sharex': True}, col="State", kind="line", legend=False, col_wrap=7)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=6)
plt.ylabel("Selection Ratio")
plt.xlabel("Year")
plt.show()

# %%
