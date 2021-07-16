# %%
import numpy as np
import pandas as pd
import baynet
from baynet import DAG
from baynet.interventions import collapse_posterior
import seaborn as sns
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "data"
df = pd.read_csv(data_path / "demographics" / "counties_agecat.csv", dtype={'FIPS': object})

drug_use_df = pd.read_csv(data_path / "NSDUH" / "NSDUH_2019_Tab.txt", sep="\t", usecols=["NEWRACE2", "CATAG3", "IRSEX", "MJDAY30A"])
race_dict = {
    1 : "white",
    2 : "black",
    3 : "other/mixed",
    4 : "other/mixed",
    5 : "other/mixed",
    6 : "other/mixed",
    7 : "other/mixed"
}

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

def binary_usage(n):
    if n <= 30:
        return 1
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
drug_use_df["MJBINARY"] = drug_use_df.MJDAY30A.map(binary_usage)
drug_use_df.MJDAY30A = drug_use_df.MJDAY30A.map(usage)

def get_usage_cpt(usage_name: str):
    cross = pd.crosstab(drug_use_df[usage_name], [drug_use_df.AGE, drug_use_df.RACE, drug_use_df.SEX], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / (len(cross.index) - 1)
    x = cross_sum.to_xarray()
    neg_x = 1 - x.values.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

cpt = get_usage_cpt("MJDAY30A")
binary_cpt = get_usage_cpt("MJBINARY")
# %%
dag = DAG.from_modelstring("[incident|county:pop_age:pop_race:pop_sex:uses_cannabis][uses_cannabis|pop_age:pop_race:pop_sex][pop_race|county:pop_age:pop_sex][pop_age|county:pop_sex][pop_sex|county][county]")
dag.get_node("pop_sex")["levels"] = drug_use_df.SEX.unique().tolist()
dag.get_node("pop_age")["levels"] = drug_use_df.AGE.unique().tolist()
dag.get_node("pop_race")["levels"] = drug_use_df.RACE.unique().tolist()
dag.get_node("uses_cannabis")["levels"] = ["n", "y"]
dag.get_node("incident")["levels"] = ["n", "y"]
dag.get_node("county")["levels"] = df["FIPS"].unique().tolist()
# %%
dag.get_node("uses_cannabis")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("uses_cannabis"))
dag.get_node("uses_cannabis")["CPD"].parents = ['pop_age', 'pop_race', 'pop_sex']
dag.get_node("uses_cannabis")["CPD"].array = cpt
dag.get_node("uses_cannabis")["CPD"].rescale_probabilities()


# In[67]:


def get_cpd(data, child, parents, norm=True):
    grouped = data.groupby([*parents, child])["frequency"].sum()
    if not norm:
        return grouped
    if parents:
        return (grouped / data.groupby([*parents])["frequency"].sum()).to_xarray().values
    else:
        return (grouped / data["frequency"].sum()).values

dag.get_node("pop_race")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_race"))
dag.get_node("pop_race")["CPD"].array = get_cpd(df, "RACE", ["FIPS", "AGE", "SEX"])
dag.get_node("pop_race")["CPD"].rescale_probabilities()

dag.get_node("pop_sex")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_sex"))
dag.get_node("pop_sex")["CPD"].array = get_cpd(df, "SEX", ["FIPS"])
dag.get_node("pop_sex")["CPD"].rescale_probabilities()

dag.get_node("pop_age")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_age"))
dag.get_node("pop_age")["CPD"].array = get_cpd(df, "AGE", ["FIPS", "SEX"])
dag.get_node("pop_age")["CPD"].rescale_probabilities()

dag.get_node("county")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("county"))
dag.get_node("county")["CPD"].array = get_cpd(df, "FIPS", [])
dag.get_node("county")["CPD"].rescale_probabilities()

def get_usage_by_county(bn: DAG, county_name: str):
    local_bn = bn.copy()
    county_idx = local_bn.get_node("county")["levels"].index(county_name)
    county_array = np.zeros(local_bn.get_node("county")["CPD"].array.shape)
    county_array[county_idx] = 1
    local_bn.get_node("county")["CPD"].array = county_array
    return collapse_posterior(local_bn, "uses_cannabis")[0]

# df["cannabis_usage"] = df.FIPS.map(lambda x: get_usage_by_county(dag, x))
# %%
nibrs_df = pd.read_csv(data_path / "NIBRS" / "cannabis_agency_2019_20210608.csv", usecols=["dm_offender_race_ethnicity", "dm_offender_age", "dm_offender_sex", "arrest_type","cannabis_mass", "ori"])
nibrs_df.rename(columns={
    "dm_offender_race_ethnicity": "RACE",
    "dm_offender_age": "AGE",
    "dm_offender_sex": "SEX"
}, inplace=True)
nibrs_df = nibrs_df[nibrs_df["RACE"] != "hispanic/latino"]
fips_ori = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t", usecols=["ORI9", "FIPS"], dtype={'FIPS': object})
fips_ori = fips_ori.rename(columns={"ORI9": "ori"})

nibrs_df = pd.merge(nibrs_df, fips_ori, on="ori")

nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]

# %%
# def get_county_incident(
#     pop_df: pd.DataFrame,
#     usage_df: pd.DataFrame,
#     inc_df: pd.DataFrame,
#     dem_order: pd.DataFrame,
#     fips: str
# ) -> np.ndarray:
#     county_pop = pop_df[pop_df["FIPS"] == fips]
#     county_incidents = inc_df[]
#     cross = pd.crosstab(usage_df.MJDAY30A, [usage_df[col] for col in dem_order], normalize="columns")
#     cross_mult = cross.multiply(cross.index, axis="rows")
#     cross_sum = cross_mult.sum(axis="rows") / 30
#     inc_cpt = inc_df.groupby(cols).size() / (county_pop.groupby(cols)["frequency"].sum() * cross_sum)


def get_incident_cpt(pop_df, usage_df, inc_df, dem_order):
    cross = pd.crosstab(usage_df.MJDAY30A, [usage_df[col] for col in dem_order], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / 30
    # pop_df = pop_df[pop_df["FIPS"].isin(inc_df.FIPS.unique().tolist())]
    grouped_incidents = inc_df.groupby(cols + ["FIPS"]).size()
    expected_incidents = (pop_df.groupby(cols + ["FIPS"])["frequency"].sum() * cross_sum)
    inc_cpt = grouped_incidents / expected_incidents
    x = inc_cpt.to_xarray()
    x = np.stack([np.zeros(x.shape), x.values.copy()])
    x = np.nan_to_num(x)
    neg_x = 1 - x.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

cols = ["AGE", "RACE", "SEX"]

df["SEX"] = df.SEX.apply(lambda x: x.lower())
df["RACE"] = df.RACE.apply(lambda x: x.lower())

inc_cpt = get_incident_cpt(df, drug_use_df, nibrs_df, cols)

dag.get_node("incident")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("incident"))
dag.get_node("incident")["CPD"].array = inc_cpt
dag.get_node("incident")["CPD"].parents = ['uses_cannabis', 'pop_age', 'pop_race', 'pop_sex', 'county']
dag.get_node("incident")["CPD"].rescale_probabilities()

# %%

s = dag.sample(10)
breakpoint()
def get_selection_by_vars(bn: DAG, county_name: str, race_level: str, age_level: str, sex_level: str):
    local_bn = bn.copy()
    def _set_cpt(node: str, level: str) -> np.ndarray:
        idx = local_bn.get_node(node)["levels"].index(level)
        array = np.zeros(local_bn.get_node(node)["CPD"].array.shape)
        array[..., idx] = 1
        return array
    local_bn.get_node("county")["CPD"].array = _set_cpt("county", county_name)
    local_bn.get_node("pop_race")["CPD"].array = _set_cpt("pop_race", race_level)
    # local_bn.get_node("pop_age")["CPD"].array = _set_cpt("pop_age", age_level)
    # local_bn.get_node("pop_sex")["CPD"].array = _set_cpt("pop_sex", sex_level)
    return collapse_posterior(local_bn, "incident")

x = get_selection_by_vars(dag, "41051", "white", "18-25", "male")



# %%
arrests_cpt =  get_incident_cpt(df, drug_use_df, nibrs_arrests, cols)


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
