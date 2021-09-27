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
    usecols=["FIPS", "selection_ratio"],
)
dems["selection_ratio"] = np.log(dems["selection_ratio"])
dems["Model"] = "Dems Only"


## POVERTY ##

poverty = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
poverty["selection_ratio"] = np.log(poverty["selection_ratio"])
poverty["Model"] = "Dems + Poverty"

## metro ##

metro = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_metro.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
metro["selection_ratio"] = np.log(metro["selection_ratio"])
metro["Model"] = "Dems + metro"

## BUYING ##

buying = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
buying["selection_ratio"] = np.log(buying["selection_ratio"])
buying["Model"] = "Buying"


## BUYING OUTSIDE ##

buying_outside = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
buying_outside["Model"] = "Buying Outside"
buying_outside["selection_ratio"] = np.log(buying_outside["selection_ratio"])

## ARRESTS ##

arrests = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)

arrests["Model"] = "Arrests (Poverty)"
arrests["selection_ratio"] = np.log(arrests["selection_ratio"])


poverty = poverty.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "Model": "Model_base",
    }
)


dems = poverty.merge(dems, on="FIPS")
metro = poverty.merge(metro, on="FIPS")
buying = poverty.merge(buying, on="FIPS")
buying_outside = poverty.merge(buying_outside, on="FIPS")
arrests = poverty.merge(arrests, on="FIPS")

data = pd.concat([dems, metro, buying, buying_outside, arrests], ignore_index=True)

# %%
data["diff"] = data["selection_ratio"] - data["selection_ratio_base"]
sns.set_style("whitegrid")

ax = sns.histplot(
    data=data,
    x="diff",
    hue="Model",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-2, 2])

ax.set_xlabel("Selection Ratio X Model - Selection Ratio Poverty Model")

plt.show()
# %%


poverty = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
poverty["selection_ratio"] = np.log(poverty["selection_ratio"])
poverty["Model"] = "Dems + Poverty"

poverty = poverty.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "Model": "Model_base",
    }
)

day = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_day.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "white_incidents", "black_incidents"],
)
day["selection_ratio"] = np.log(day["selection_ratio"])
day["Model"] = "Dems + Poverty (Day)"

night = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_night.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "white_incidents", "black_incidents"],
)
night["selection_ratio"] = np.log(night["selection_ratio"])
night["Model"] = "Dems + Poverty (Night)"

day = poverty.merge(day, on="FIPS")
night = poverty.merge(night, on="FIPS")

day = day[day.FIPS.isin(night.FIPS)]
night = night[night.FIPS.isin(day.FIPS)]

data = pd.concat([day, night], ignore_index=True)

# %%
data["diff"] = data["selection_ratio"] - data["selection_ratio_base"]
sns.set_style("whitegrid")

ax = sns.histplot(
    data=data,
    x="diff",
    hue="Model",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-2, 2])

ax.set_xlabel("Selection Ratio X Model - Selection Ratio Poverty Model")

plt.show()
# %%


plt.scatter(x=np.log(day.selection_ratio), y=np.log(night.selection_ratio))
plt.plot(np.linspace(-3, 10), np.linspace(-3, 10))
plt.xlabel("Day Selection Ratio")
plt.ylabel("Night Selection Ratio")
plt.show()


# %%
import statsmodels.api as sm
from patsy import dmatrices

daynight = day.merge(night, on="FIPS")


def check_coef(df, col1, col2):
    df = df[~df[col1].isnull()]
    df = df[~df[col2].isnull()]
    y, X = dmatrices(f"{col1} ~ {col2}", data=df, return_type="dataframe")
    model = sm.OLS(
        y,
        X,
    )
    model_res = model.fit()
    model_res = model_res.get_robustcov_results(cov_type="HC1")
    print(model_res.summary())


# %%

daynight["diff"] = daynight["selection_ratio_x"] - daynight["selection_ratio_y"]

ax = sns.histplot(
    data=daynight,
    x="diff",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)

ax.set_xlabel("Selection Ratio X Model - Selection Ratio Poverty Model")

plt.show()
# %%

poverty = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)
poverty["selection_ratio"] = np.log(poverty["selection_ratio"])
poverty["Model"] = "Dems + Poverty"

poverty = poverty.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "Model": "Model_base",
    }
)

day_all = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_all_incidents_day.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "white_incidents", "black_incidents"],
)
day_all["selection_ratio"] = np.log(day["selection_ratio"])
day_all["Model"] = "Dems + Poverty (Day)"

night_all = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_all_incidents_night.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "white_incidents", "black_incidents"],
)
night_all["selection_ratio"] = np.log(night["selection_ratio"])
night_all["Model"] = "Dems + Poverty (Night)"

day_all = poverty.merge(day, on="FIPS")
night_all = poverty.merge(night, on="FIPS")

day_all = day[day.FIPS.isin(night.FIPS)]
night_all = night[night.FIPS.isin(day.FIPS)]

data = pd.concat([day_all, night_all], ignore_index=True)

data["diff"] = data["selection_ratio"] - data["selection_ratio_base"]
sns.set_style("whitegrid")

ax = sns.histplot(
    data=data,
    x="diff",
    hue="Model",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-2, 2])

ax.set_xlabel("Selection Ratio X Model - Selection Ratio Poverty Model")

plt.show()
# %%

night = pd.read_csv(
    data_path / "NIBRS" / "incidents_2010-2019_night.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "incidents", "race", "age", "sex"],
)

night = night.rename(columns={"incidents": "incidents_night"})

day = pd.read_csv(
    data_path / "NIBRS" / "incidents_2010-2019_day.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "incidents", "race", "age", "sex"],
)

day = day.rename(columns={"incidents": "incidents_day"})


data = night.merge(day, on=["FIPS", "race", "age", "sex"], how="outer")

data["incidents"] = data["incidents_night"] + data["incidents_day"]
data = data.groupby(["FIPS", "race"]).sum().reset_index()
# %%
