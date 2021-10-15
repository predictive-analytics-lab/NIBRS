# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / "data"

sb_df = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
)

cr_df = pd.read_csv(data_path / "correlates" / "density_all.csv")
cr_df["FIPS"] = cr_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
cr_df = cr_df.rename(columns={"Population_Density_in_2010": "density_all"})
cr_df = cr_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(cr_df, on=["FIPS"], how="left")

sb_df["density_county_ratio"] = np.log10(
    sb_df["density_all"] / sb_df["density_all"].mean()
)

# sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%
sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])
# %%
# ax = sns.regplot(x="cr_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
sb_df = sb_df[~sb_df.density_county_ratio.isnull()]


res = stats.linregress(x=sb_df.density_county_ratio, y=sb_df.selection_ratio_log10)


g = sns.jointplot(
    data=sb_df, x="density_county_ratio", y="selection_ratio_log10", kind="reg"
)
grid = np.linspace(sb_df.density_county_ratio.min(), sb_df.density_county_ratio.max())


# g.ax_joint.plot(grid, res.intercept + res.slope * grid, color="g", lw=2)

plt.text(x=0.5, y=3.2, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

g.ax_joint.set_ylim([-1, 3])
g.ax_joint.set_xlabel("log10 (population density / mean(population density))")
g.ax_joint.set_ylabel("log10 (selection ratio)")

plt.show()

# %%
