# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_wilson_poverty.csv', dtype={"FIPS": str})
ib_df = pd.read_csv(data_path / "correlates" / "incarceration_black.csv")
iw_df = pd.read_csv(data_path / "correlates" / "incarceration_white.csv")
ia_df = pd.read_csv(data_path / "correlates" / "incarceration_all.csv")

ib_df["FIPS"] = ib_df["cty"].apply(lambda x: x[3:])
iw_df["FIPS"] = iw_df["cty"].apply(lambda x: x[3:])
ia_df["FIPS"] = ia_df["cty"].apply(lambda x: x[3:])

ib_df = ib_df.rename(columns={"Incarceration_Rate_rB_gP_pall": "incarceration_rate_black"})
iw_df = iw_df.rename(columns={"Incarceration_Rate_rW_gP_pall": "incarceration_rate_white"})
ia_df = ia_df.rename(columns={"Incarceration_Rate_rP_gP_pall": "incarceration_rate_all"})

ib_df = ib_df.drop(columns=["Name", "cty"])
iw_df = iw_df.drop(columns=["Name", "cty"])
ia_df = ia_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(ib_df, on=["FIPS"])
sb_df = sb_df.merge(iw_df, on=["FIPS"])
sb_df = sb_df.merge(ia_df, on=["FIPS"]) 

sb_df["ir_county_ratio"] = sb_df["incarceration_rate_all"] / sb_df["incarceration_rate_all"].mean()
sb_df["ir_bw_ratio"] = sb_df["incarceration_rate_black"] / sb_df["incarceration_rate_white"]
sb_df["ir_b_county_ratio"] = sb_df["incarceration_rate_black"] / sb_df["incarceration_rate_black"].mean()

sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
ax = sns.regplot(x="ir_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.show()

# %%
ax = sns.regplot(x="ir_bw_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.xlim([0, 10])
plt.show()
# %%
ax = sns.regplot(x="ir_b_county_ratio", y="selection_ratio_log10",fit_reg=True, data=sb_df)

plt.show()
# %%
