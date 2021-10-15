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


g = sns.relplot(data=df, x="year", y="sr_and_ci", col_order=sorted(list(df.State.unique())), estimator=weighted_mean,markers=True, facet_kws={'sharey': False, 'sharex': True}, col="State", kind="line", legend=False, col_wrap=7)
for ax in g.fig.axes:
    ax.invert_yaxis()
    ax.set_ylim([0, 12.5])
    ax.set_ylabel("Selection Ratio")

g.fig.suptitle("Selection Ratio (weighted by 1/ci) 2012-2019", fontsize=40)
g.fig.subplots_adjust(top=0.92);
plt.show()
# %%
