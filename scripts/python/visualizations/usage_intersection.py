# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")

from pathlib import Path

data_path = Path(__file__).parents[3] / "data"


## DEMS ##

dems = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)
dems["selection_ratio"] = np.log10(dems["selection_ratio"])
dems["model"] = "Dems Only"


## POVERTY ##

poverty = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)
poverty["selection_ratio"] = np.log10(poverty["selection_ratio"])
poverty["model"] = "Dems + Poverty"

## URBAN ##

urban = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_urban.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)
urban["selection_ratio"] = np.log10(urban["selection_ratio"])
urban["model"] = "Dems + Urban"

## BUYING ##

buying = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)
buying["selection_ratio"] = np.log10(buying["selection_ratio"])
buying["model"] = "Buying"


## BUYING OUTSIDE ##

buying_outside = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)
buying_outside["model"] = "Buying Outside"
buying_outside["selection_ratio"] = np.log10(buying_outside["selection_ratio"])


arrests = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

arrests["model"] = "arrests"
arrests["selection_ratio"] = np.log10(arrests["selection_ratio"])

arrests = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

arrests["model"] = "arrests"
arrests["selection_ratio"] = np.log10(arrests["selection_ratio"])


poverty = poverty.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "var_log": "var_log_base",
        "model": "model_base",
    }
)


dems = poverty.merge(dems, on="FIPS")
urban = poverty.merge(urban, on="FIPS")
buying = poverty.merge(buying, on="FIPS")
buying_outside = poverty.merge(buying_outside, on="FIPS")
arrests = poverty.merge(arrests, on="FIPS")

data = pd.concat([dems, urban, buying, buying_outside, arrests], ignore_index=True)

# %%
data["diff"] = data["selection_ratio"] - data["selection_ratio_base"]

ax = sns.kdeplot(
    data=data,
    x="diff",
    hue="model",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-1, 1])
plt.show()
# %%
