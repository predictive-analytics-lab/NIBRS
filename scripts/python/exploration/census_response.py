# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_wilson_poverty.csv', dtype={"FIPS": str})

cr_df = pd.read_csv(data_path / "correlates" / "census_response_all.csv")
cr_df["FIPS"] = cr_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
cr_df = cr_df.rename(columns={"Census_Response_Rate_Social_Capital_Proxy": "census_response_all"})
cr_df = cr_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(cr_df, on=["FIPS"], how="left") 

sb_df["cr_county_ratio"] = sb_df["census_response_all"] / sb_df["census_response_all"].mean()

sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
# ax = sns.regplot(x="cr_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
sb_df = sb_df[~sb_df.cr_county_ratio.isnull()]
res = stats.linregress(x=sb_df.cr_county_ratio, y=sb_df.selection_ratio_log10)



g = sns.jointplot(data=sb_df, x="cr_county_ratio", y="selection_ratio_log10", kind="reg")
grid = np.linspace(sb_df.cr_county_ratio.min(), sb_df.cr_county_ratio.max())


# g.ax_joint.plot(grid, res.intercept + res.slope * grid, color="g", lw=2)

plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

plt.ylim([0, 2])
plt.show()

# %%
