# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_wilson_poverty.csv', dtype={"FIPS": str})
bg_df = pd.read_csv(data_path / "correlates" / "hsgrad_black.csv")
wg_df = pd.read_csv(data_path / "correlates" / "hsgrad_white.csv")
ag_df = pd.read_csv(data_path / "correlates" / "hsgrad_all.csv")

bg_df["FIPS"] = bg_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
wg_df["FIPS"] = wg_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))
ag_df["FIPS"] = ag_df["cty"].apply(lambda x: x[3:].rjust(5, "0"))

bg_df = bg_df.rename(columns={"High_School_Graduation_Rate_rB_gP_pall": "hsgrad_black"})
wg_df = wg_df.rename(columns={"High_School_Graduation_Rate_rW_gP_pall": "hsgrad_white"})
ag_df = ag_df.rename(columns={"High_School_Graduation_Rate_rP_gP_pall": "hsgrad_all"})

bg_df = bg_df.drop(columns=["Name", "cty"])
wg_df = wg_df.drop(columns=["Name", "cty"])
ag_df = ag_df.drop(columns=["Name", "cty"])

sb_df = sb_df.merge(bg_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(wg_df, on=["FIPS"], how="left")
sb_df = sb_df.merge(ag_df, on=["FIPS"], how="left") 

sb_df["hsg_county_ratio"] = sb_df["hsgrad_all"] / sb_df["hsgrad_all"].mean()
sb_df["hsg_bw_ratio"] = sb_df["hsgrad_black"] / sb_df["hsgrad_white"]
sb_df["hsg_b_county_ratio"] = sb_df["hsgrad_black"] / sb_df["hsgrad_black"].mean()

sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]
# %%

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# %%
ax = sns.regplot(x="hsg_county_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.show()

# %%
ax = sns.regplot(x="hsg_bw_ratio", y="selection_ratio_log10", fit_reg=True, data=sb_df)
plt.ylim([0, 10])
plt.show()
# %%
ax = sns.regplot(x="hsg_b_county_ratio", y="selection_ratio_log10",fit_reg=True, data=sb_df)

plt.show()
