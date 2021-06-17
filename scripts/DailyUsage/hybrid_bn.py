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
    1: "12-17",
    2: "18-25",
    3: "26-34",
    4: "35-49",
    5: "50+",
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

cross = pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df.RACE, drug_use_df.AGE, drug_use_df.SEX], normalize="columns")
cross_mult = cross.multiply(cross.index, axis="rows")
cross_sum = cross_mult.sum(axis="rows") / 30

x = cross_sum.to_xarray()
neg_x = 1 - x.values.copy()
cpt = np.stack([x.copy(), neg_x], axis=-1)


# %%
dag = DAG.from_modelstring("[incident|uses_cannabis:pop_race:pop_age:pop_sex][uses_cannabis|pop_race:pop_age:pop_sex][pop_race|pop_age:pop_sex][pop_age|pop_sex][pop_sex]")
dag.get_node("pop_sex")["levels"] = list(x.SEX.values)
dag.get_node("pop_age")["levels"] = list(x.AGE.values)
dag.get_node("pop_race")["levels"] = list(x.RACE.values)
dag.get_node("uses_cannabis")["levels"] = ["y", "n"]
dag.get_node("incident")["levels"] = ["y", "n"]

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
        return "12-17"
    if x < 26:
        return "18-25"
    if x < 35:
        return "26-34"
    if x < 50:
        return "35-49"
    return "50+"

df = pd.read_csv("sc-est2019-alldata5.csv")

df = df[df.ORIGIN != 0]
df.SEX = df.SEX.map(sex_dict)
df.RACE = df.apply(RACE, axis=1)
df = df[df.RACE != "other/mixed"]
df = df[df.SEX != "total"]
df = df[df.AGE >= 12]
df.AGE = df.AGE.map(AGE)

# In[67]:


def get_cpd(data, child, parents, norm=True):
    grouped = data.groupby([*parents, child])["POPESTIMATE2019"].sum()
    if not norm:
        return grouped
    if parents:
        return (grouped / data.groupby([*parents])["POPESTIMATE2019"].sum()).to_xarray().values
    else:
        return (grouped / data["POPESTIMATE2019"].sum()).values

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
nibrs_df = pd.read_csv("drugs_2019_20210603.csv", usecols=["dm_offender_race_ethnicity", "dm_offender_age", "dm_offender_sex", "arrest_type","crack_mass","cocaine_mass","heroin_mass","cannabis_mass","meth_amphetamines_mass","other_drugs"])
nibrs_df.rename(columns={
    "dm_offender_race_ethnicity": "RACE",
    "dm_offender_age": "AGE",
    "dm_offender_sex": "SEX"
}, inplace=True)
nibrs_df = nibrs_df[nibrs_df.arrest_type != "No Arrest"]
nibrs_df = nibrs_df[(nibrs_df.cannabis_mass > 0) & (nibrs_df.crack_mass + nibrs_df.cocaine_mass + nibrs_df.heroin_mass + nibrs_df.meth_amphetamines_mass + nibrs_df.other_drugs == 0)]


# %%
def get_incident_cpt(pop_df, usage_df, inc_df, dem_order):
    cross = pd.crosstab(usage_df.MJDAY30A, [usage_df[col] for col in dem_order], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / 30
    inc_cpt = inc_df.groupby(cols).size() / (pop_df.groupby(cols)["POPESTIMATE2019"].sum() * cross_sum)
    return inc_cpt
cols = ["AGE", "SEX", "RACE"]
inc_cpt = get_incident_cpt(df, drug_use_df, nibrs_df, cols)
inc_cpt.plot(kind="bar")
inc_array = inc_cpt.to_xarray().values
inc_array = np.stack([inc_array, np.zeros(inc_array.shape)], axis=-1)
inc_array = np.stack([inc_array, 1-inc_array], axis=-1)

# %%
dag.get_node("incident")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("incident"))
dag.get_node("incident")["CPD"].parents = ["pop_" + col.lower() for col in cols] + ["uses_cannabis"]

# %%

dag.get_node("incident")["CPD"].array = inc_array
dag.get_node("incident")["CPD"].rescale_probabilities()



# %%
