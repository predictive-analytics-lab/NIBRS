# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df2 = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_wilson_poverty.csv', dtype={"FIPS": str})


sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_wilson_poverty.csv', dtype={"FIPS": str})
bb_df = pd.read_csv(data_path / "correlates" / "birthrate_black.csv")
bw_df = pd.read_csv(data_path / "correlates" / "birthrate_white.csv")
ba_df = pd.read_csv(data_path / "correlates" / "birthrate_all.csv")

bb_df["FIPS"] = bb_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
bw_df["FIPS"] = bw_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
ba_df["FIPS"] = ba_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))

bb_df = bb_df.rename(columns={"Teenage_Birth_Rate_women_only_rB_gF_pall": "birthrate_black"})
bw_df = bw_df.rename(columns={"Teenage_Birth_Rate_women_only_rW_gF_pall": "birthrate_white"})
ba_df = ba_df.rename(columns={"Teenage_Birth_Rate_women_only_rP_gF_pall": "birthrate_all"})

bb_df = bb_df.drop(columns=["Name", "cty"])
bw_df = bw_df.drop(columns=["Name", "cty"])
ba_df = ba_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(bb_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(bw_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(ba_df, on=["FIPS"], how="left") 

sb_df["br_county_ratio"] = sb_df["birthrate_all"] / sb_df["birthrate_all"].mean()
sb_df["br_bw_ratio"] = sb_df["birthrate_black"] / sb_df["birthrate_white"]
sb_df["br_b_county_ratio"] = sb_df["birthrate_black"] / sb_df["birthrate_black"].mean()

sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
ax = sns.regplot(x="br_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.show()

# %%
ax = sns.regplot(x="br_bw_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.show()
# %%
ax = sns.regplot(x="br_b_county_ratio", y="selection_ratio_log10",fit_reg=True, data=sb_df)

plt.show()
