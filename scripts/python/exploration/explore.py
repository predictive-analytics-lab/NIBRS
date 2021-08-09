# %% 
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
# %%
data_path = Path(__file__).parent.parent.parent / "data"
df = pd.read_csv(data_path / "NIBRS" / "cannabis_agency_2019_20210608.csv")
# %%

a = df.groupby(["agency", "population_group_code"]).size().reset_index().groupby("population_group_code").agg(["mean", "std"])
b = df.groupby("population_group_code")["agency"].nunique()
comb = pd.concat([a, b], axis=1).reset_index

# %%

crimes_per_county = df.groupby("county_name").size() 
agencies_per_county = df.groupby("county_name")["agency"].nunique()

# %%
crimes_per_county.hist(bins=np.linspace(0, 1000, 100))
plt.show()
# %%
agencies_per_county.hist(bins=np.linspace(0, 100, 100))
plt.show()

# %%

df.groupby("agency").size()
# %%

df.groupby("agency").agency_id.unique()
# %%

officers_by_agency = df.groupby("agency_id").officers.first()
crimes_per_agency = df.groupby("agency_id").size()
officers_vs_crimes = pd.concat([officers_by_agency, crimes_per_agency], axis=1)
officers_vs_crimes = officers_vs_crimes.rename({0: "crimes"})
officers_vs_crimes.dropna()
# %%
from scipy.stats import pearsonr

pearsonr(officers_vs_crimes.dropna()[0], officers_vs_crimes.dropna()["officers"])
# %%

drug_use_df = pd.read_csv(data_path / "NSDUH" / "NSDUH_2019_Tab.txt", sep="\t")

# %%

race_dict = {1 : "white",
2 : "black",
3 : "other/mixed",
4 : "other/mixed",
5 : "other/mixed",
6 : "other/mixed",
7 : "other/mixed"}

sex_dict = {
    0: "total",
    1: "male",
    2: "female",
}

age_dict = {
    1: "12-17",
    2: "18-25",
    3: "26-34",
    4: "35-49",
    5: "50+",
}


personal_income_dict = {
    1: "0 - 20K",
    2: "0 - 20K",
    3: "20K-50K",
    4: "20K-50K",
    5: "20K-50K",
    6: "50-75K",
    7: "75K+"
}


hh_income_dict = {
    1: "0 - 20K",
    2: "20K-50K",
    3: "50K-75K",
    4: "75K+",
}


def usage(n):
    if n <= 30:
        return n
    else:
        return 0

# %%
drug_use_df.rename(columns={
    "NEWRACE2": "RACE",
    "CATAG3": "AGE",
    "IRSEX": "SEX"
}, inplace=True)
drug_use_df.RACE = drug_use_df.RACE.map(race_dict)
drug_use_df = drug_use_df[drug_use_df.RACE != "other/mixed"]
drug_use_df.AGE = drug_use_df.AGE.map(age_dict)
drug_use_df.SEX = drug_use_df.SEX.map(sex_dict)
drug_use_df.MJDAY30A = drug_use_df.MJDAY30A.map(usage)
drug_use_df["PINCOME"] = drug_use_df.IRPINC3.map(personal_income_dict)
drug_use_df["INCOME"] = drug_use_df.INCOME.map(hh_income_dict)


cross = pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df.RACE, drug_use_df.AGE, drug_use_df.INCOME, drug_use_df.SEX], normalize="columns")
cross_mult = cross.multiply(cross.index, axis="rows")
cross_sum = cross_mult.sum(axis="rows") / 30
# %%
data = cross_sum.reset_index()

plt.style.use('ggplot')

p = ["#a7ffeb", "#00695c"]

sns.set_context("talk")

sns.set_context(rc = {'patch.linewidth': 1.0})
sns.set_context(rc = {'bar.edgecolor': "black"})

row_order = sorted(drug_use_df.AGE.unique())
hue_order = sorted(drug_use_df.INCOME.unique())
order = ["white", "black"]
sns.catplot(data=data, kind="bar", x="RACE", order = order,  y=0, row="AGE", col="SEX", hue="INCOME", hue_order=hue_order, row_order=row_order, sharex=False)

# %%
sns.catplot(data=drug_use_df, kind="count", x="RACE", order = order, row="AGE", row_order=row_order, col="SEX", hue="INCOME", hue_order=hue_order, sharex=False)

# %%
sns.catplot(data=drug_use_df[drug_use_df.MJDAY30A > 0], order=order, kind="count", x="RACE", row="AGE", row_order=row_order, col="SEX", hue="INCOME", hue_order=hue_order, sharex=False)