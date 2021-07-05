# %%
import numpy as np
import pandas as pd
import baynet
from baynet import DAG
import seaborn as sns

drug_use_df = pd.read_csv("../../data/NSDUH_2019_Tab.txt", sep="\t")
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
dag.get_node("pop_sex")["levels"] = drug_use_df.SEX.unique().tolist()
dag.get_node("pop_age")["levels"] = drug_use_df.AGE.unique().tolist()
dag.get_node("pop_race")["levels"] = drug_use_df.RACE.unique().tolist()
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

df = pd.read_csv("../../data/sc-est2019-alldata5.csv")

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
nibrs_df = pd.read_csv("../../data/cannabis_2019_20210608.csv", usecols=["dm_offender_race_ethnicity", "dm_offender_age", "dm_offender_sex", "arrest_type","cannabis_mass"])
nibrs_df.rename(columns={
    "dm_offender_race_ethnicity": "RACE",
    "dm_offender_age": "AGE",
    "dm_offender_sex": "SEX"
}, inplace=True)
nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]

# %%
def get_incident_cpt(pop_df, usage_df, inc_df, dem_order):
    cross = pd.crosstab(usage_df.MJDAY30A, [usage_df[col] for col in dem_order], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / 30
    inc_cpt = inc_df.groupby(cols).size() / (pop_df.groupby(cols)["POPESTIMATE2019"].sum() * cross_sum)
    return inc_cpt
cols = ["AGE", "SEX", "RACE"]
inc_cpt = get_incident_cpt(df, drug_use_df, nibrs_df, cols)
arrests_cpt =  get_incident_cpt(df, drug_use_df, nibrs_arrests, cols)

# %%

inc_df = inc_cpt.to_frame().reset_index()
inc_df["TYPE"] = "Incident"

arrest_df = arrests_cpt.to_frame().reset_index()
arrest_df["TYPE"] = "Arrest"


plotting_data = pd.concat([inc_df, arrest_df])

# %%
p = ["#a7ffeb", "#00695c"]

sns.set_context("talk")
sns.set_style("whitegrid")

sns.set_context(rc = {'patch.linewidth': 1.0})
sns.set_context(rc = {'bar.edgecolor': "black"})


g = sns.FacetGrid(plotting_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p, height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("P(Incident|D, Use = T)")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Probability of an Incident given demographics")

#inc_cpt.plot(kind="bar")
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
p = ["#00685c", "#00695c"]

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)


incidents["TYPE"] = "Incident"
arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([incidents, arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Frequency")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Frequency of Incident / Arrests given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")
        # if patch._facecolor[0] == 0.0514705882352941:
        #     pass
        # else:
        #     patch.set_hatch("\\")



# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
census_data = df.groupby(["AGE", "SEX", "RACE"])["POPESTIMATE2019"].sum().reset_index()

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
incidents[0] = (incidents[0] / census_data["POPESTIMATE2019"]) * 100_000

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
arrests[0] = (arrests[0] / census_data["POPESTIMATE2019"]) * 100_000

incidents["TYPE"] = "Incident"
arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([incidents, arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Rate per 100K")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Number of Incidents/Arrests per 100,000 population by demographic")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")


# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)

arrests[0] /= incidents[0]

g = sns.FacetGrid(arrests, row = 'SEX',  col = 'AGE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Ratio")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Ratio of arrests to incidents given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%

census_data = df.groupby(["AGE", "SEX", "RACE"])["POPESTIMATE2019"].sum().reset_index()

census_data["POPESTIMATE2019"] /= sum(census_data["POPESTIMATE2019"])

g = sns.FacetGrid(census_data, row = 'SEX',  col = 'AGE', sharex=False, palette=p, height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', "POPESTIMATE2019", ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Proportion")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Demographic distribution as a proportion of total population")
for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%
nibrs_data.groupby(["AGE", "SEX", "RACE"]).count() / len(nibrs_data)
# %%
nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()

# %%

drug_use_df
# %%


cross = pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df[col] for col in ["AGE", "SEX", "RACE"]], normalize="columns")
cross_mult = cross.multiply(cross.index, axis="rows")
cross_sum = (cross_mult.sum(axis="rows") / 30).reset_index()

g = sns.FacetGrid(cross_sum, row = 'SEX',  col = 'AGE', sharex=False, palette=p, height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("P(U|D = d)")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Probability of drug use (in a given month) by demographics according to the NSDUH survey")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")
# %%
pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df[col] for col in ["AGE", "SEX", "RACE"]], normalize="columns")
# %%

# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)


g = sns.FacetGrid(arrests, row = 'SEX',  col = 'AGE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Ratio")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Ratio of arrests to incidents given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

nibrs_df["TYPE"] = "Incident"
nibrs_arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([nibrs_df, nibrs_arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', "cannabis_mass", ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Frequency")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Frequency of Incident / Arrests given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")


# %%

nibrs_data.groupby(["TYPE", "RACE", "SEX", "AGE"]).mean().reset_index()

# %%
