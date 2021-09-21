# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_wilson_poverty.csv', dtype={"FIPS": str})
bst_df = pd.read_csv(data_path / "correlates" / "sametract_black.csv")
wst_df = pd.read_csv(data_path / "correlates" / "sametract_white.csv")
ast_df = pd.read_csv(data_path / "correlates" / "sametract_all.csv")

bst_df["FIPS"] = bst_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
wst_df["FIPS"] = wst_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
ast_df["FIPS"] = ast_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))

ast_df = ast_df.rename(columns={"%_Staying_in_Same_Tract_as_Adults_rP_gP_pall": "sametract_black"})
bst_df = bst_df.rename(columns={"%_Staying_in_Same_Tract_as_Adults_rB_gP_pall": "sametract_white"})
wst_df = wst_df.rename(columns={"%_Staying_in_Same_Tract_as_Adults_rW_gP_pall": "sametract_all"})

bst_df = bst_df.drop(columns=["Name", "cty"])
wst_df = wst_df.drop(columns=["Name", "cty"])
ast_df = ast_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(bst_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(wst_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(ast_df, on=["FIPS"], how="left") 

sb_df["sametract_county_ratio"] = sb_df["sametract_all"] / sb_df["sametract_all"].mean()
sb_df["sametract_bw_ratio"] = sb_df["sametract_black"] / sb_df["sametract_white"]
sb_df["sametract_b_county_ratio"] = sb_df["sametract_black"] / sb_df["sametract_black"].mean()

sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
# ax = sns.regplot(x="sametract_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)

sb_df_temp = sb_df[~sb_df.sametract_county_ratio.isnull()]

res = stats.linregress(x=sb_df_temp.sametract_county_ratio, y=sb_df_temp.selection_ratio_log10)

g = sns.jointplot(data=sb_df_temp, x="sametract_county_ratio", y="selection_ratio_log10", kind="reg")
plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

g.ax_joint.set_ylim([0, 2])
g.ax_joint.set_xlim([0, 2])
plt.show()

# %%

sb_df_temp = sb_df[~sb_df.sametract_bw_ratio.isnull()]

res = stats.linregress(x=sb_df_temp.sametract_bw_ratio, y=sb_df_temp.selection_ratio_log10)
g = sns.jointplot(data=sb_df_temp, x="sametract_bw_ratio", y="selection_ratio_log10", kind="reg")

g.ax_joint.set_ylim([0, 2])
g.ax_joint.set_xlim([0, 2])

plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")


plt.show()
# %%

sb_df_temp = sb_df[~sb_df.sametract_b_county_ratio.isnull()]

res = stats.linregress(x=sb_df_temp.sametract_b_county_ratio, y=sb_df_temp.selection_ratio_log10)

g = sns.jointplot(data=sb_df_temp, x="sametract_b_county_ratio", y="selection_ratio_log10", kind="reg")
plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

g.ax_joint.set_ylim([0, 2])
g.ax_joint.set_xlim([0, 2])
plt.show()

# %%
