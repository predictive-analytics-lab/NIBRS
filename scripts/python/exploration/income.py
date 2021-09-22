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
bi_df = pd.read_csv(data_path / "correlates" / "income_black.csv")
wi_df = pd.read_csv(data_path / "correlates" / "income_white.csv")
ai_df = pd.read_csv(data_path / "correlates" / "income_all.csv")

bi_df["FIPS"] = bi_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
wi_df["FIPS"] = wi_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
ai_df["FIPS"] = ai_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))

bi_df = bi_df.rename(columns={"Household_Income_at_Age_35_rB_gP_pall": "income_black"})
wi_df = wi_df.rename(columns={"Household_Income_at_Age_35_rW_gP_pall": "income_white"})
ai_df = ai_df.rename(columns={"Household_Income_at_Age_35_rP_gP_pall": "income_all"})

bi_df = bi_df.drop(columns=["Name", "cty"])
wi_df = wi_df.drop(columns=["Name", "cty"])
ai_df = ai_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(bi_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(wi_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(ai_df, on=["FIPS"], how="left")

sb_df["income_county_ratio"] = sb_df["income_all"] / sb_df["income_all"].mean()
sb_df["income_bw_ratio"] = sb_df["income_black"] / sb_df["income_white"]
sb_df["income_b_county_ratio"] = sb_df["income_black"] / sb_df["income_black"].mean()

# sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
# ax = sns.regplot(x="income_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
sb_df_temp = sb_df[~sb_df.income_county_ratio.isnull()]

res = stats.linregress(
    x=sb_df_temp.income_county_ratio, y=sb_df_temp.selection_ratio_log10
)

g = sns.jointplot(
    data=sb_df_temp, x="income_county_ratio", y="selection_ratio_log10", kind="reg"
)
plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.3f}\n ci: {res.pvalue:.3f}")

g.ax_joint.set_ylabel("log10(selection_ratio)")
g.ax_joint.set_xlabel("county income / mean(county income)")

plt.ylim([0, 2])
plt.show()

# %%

sb_df_temp = sb_df[~sb_df.income_bw_ratio.isnull()]

res = stats.linregress(x=sb_df_temp.income_bw_ratio, y=sb_df_temp.selection_ratio_log10)
g = sns.jointplot(
    data=sb_df_temp, x="income_bw_ratio", y="selection_ratio_log10", kind="reg"
)

plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.3f}\n p-value: {res.pvalue:.3f}")

plt.ylim([0, 2])
g.ax_joint.set_ylabel("log10(selection_ratio)")
plt.show()
# %%

sb_df_temp = sb_df[~sb_df.income_b_county_ratio.isnull()]

res = stats.linregress(
    x=sb_df_temp.income_b_county_ratio, y=sb_df_temp.selection_ratio_log10
)

g = sns.jointplot(
    data=sb_df_temp, x="income_b_county_ratio", y="selection_ratio_log10", kind="reg"
)
plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.3f}\n ci: {res.pvalue:.3f}")
g.ax_joint.set_ylabel("log10(selection_ratio)")

plt.ylim([0, 2])
plt.show()

# %%
