# %%
import numpy as np
import pandas as pd
import baynet
from baynet import DAG

drug_use_df = pd.read_csv("NSDUH_2019_Tab.txt", sep="\t")
race_dict = {1 : "white",
2 : "black",
3 : "other/mixed",
4 : "other/mixed",
5 : "other/mixed",
6 : "other/mixed",
7 : "hispanic/latino"}

sex_dict = {
    0: "total",
    1: "male",
    2: "female",
}

age_dict = {
    1: "12-17 Years Old",
    2: "18-25 Years Old",
    3: "26-34 Years Old",
    4: "35-49 Years Old",
    5: "50 or Older",
}

def usage(n):
    if n <= 30:
        return n
    else:
        return 0

# %%
drug_use_df.NEWRACE2 = drug_use_df.NEWRACE2.map(race_dict)
drug_use_df.CATAG3 = drug_use_df.CATAG3.map(age_dict)
drug_use_df.IRSEX = drug_use_df.IRSEX.map(sex_dict)
drug_use_df.MJDAY30A = drug_use_df.MJDAY30A.map(usage)

cross = pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df.NEWRACE2, drug_use_df.CATAG3, drug_use_df.IRSEX], normalize="columns")
cross_mult = cross.multiply(cross.index, axis="rows")
cross_sum = cross_mult.sum(axis="rows") / 30

x = cross_sum.to_xarray()
neg_x = 1 - x.values.copy()
cpt = np.stack([x.copy(), neg_x], axis=-1)


# %%
dag = DAG.from_modelstring("[uses_cannabis|pop_race:pop_age:pop_sex][pop_race|pop_age:pop_sex][pop_age|pop_sex][pop_sex]")
dag.get_node("pop_sex")["levels"] = list(x.IRSEX.values)
dag.get_node("pop_age")["levels"] = list(x.CATAG3.values)
dag.get_node("pop_race")["levels"] = list(x.NEWRACE2.values)
dag.get_node("uses_cannabis")["levels"] = ["y", "n"]

# %%
dag.get_node("uses_cannabis")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("uses_cannabis"))
dag.get_node("uses_cannabis")["CPD"].parents = ['pop_race', 'pop_age', 'pop_sex']
dag.get_node("uses_cannabis")["CPD"].array = cpt
dag.get_node("uses_cannabis")["CPD"].rescale_probabilities()
# %%

def RACE(x):
    if x["ORIGIN"] == 2:
        return "hispanic/latino"
    if x["RACE"] == 1:
        return "white"
    if x["RACE"] == 2:
        return "black"
    return "other/mixed"

def AGE(x):
    if x < 18:
        return "12-17 Years Old"
    if x < 26:
        return "18-25 Years Old"
    if x < 35:
        return "26-34 Years Old"
    if x < 50:
        return "35-49 Years Old"
    return "50 or Older"

df = pd.read_csv("sc-est2019-alldata5.csv")

df = df[df.ORIGIN != 0]
df.SEX = df.SEX.map(sex_dict)
df.RACE = df.apply(RACE, axis=1)
df = df[df.SEX != "total"]
df = df[df.AGE >= 12]
df.AGE = df.AGE.map(AGE)

# In[67]:


def get_cpd(data, child, parents):
    if parents:
        return (data.groupby([*parents, child])["POPESTIMATE2019"].sum() / data.groupby([*parents])["POPESTIMATE2019"].sum()).to_xarray().values
    else:
        return (data.groupby([*parents, child])["POPESTIMATE2019"].sum() / data["POPESTIMATE2019"].sum()).values

dag.get_node("pop_race")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_race"))
dag.get_node("pop_race")["CPD"].array = get_cpd(df, "RACE", ["AGE", "SEX"])
dag.get_node("pop_race")["CPD"].rescale_probabilities()


dag.get_node("pop_sex")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_sex"))
dag.get_node("pop_sex")["CPD"].array = get_cpd(df, "SEX", [])
dag.get_node("pop_sex")["CPD"].rescale_probabilities()

dag.get_node("pop_age")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_age"))
dag.get_node("pop_age")["CPD"].array = get_cpd(df, "AGE", ["SEX"])
dag.get_node("pop_age")["CPD"].rescale_probabilities()


# %%
dag.sample(100)


# %%
