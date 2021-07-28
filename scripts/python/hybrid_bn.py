# %%
import numpy as np
import pandas as pd
import baynet
from baynet import DAG
from baynet.interventions import collapse_posterior
import seaborn as sns
from pathlib import Path
from itertools import product

data_path = Path(__file__).parent.parent.parent / "data"
df = pd.read_csv(data_path / "demographics" / "counties_agecat.csv", dtype={'FIPS': object})
subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv", dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

drug_use_df = pd.read_csv(data_path / "NSDUH" / "NSDUH_2019_Tab.txt", sep="\t", usecols=["NEWRACE2", "CATAG3", "IRSEX", "MJDAY30A"])
df = df.dropna(subset=["STATEREGION"], how="all")

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

dag = DAG.from_modelstring("[incident|pop_age:pop_race:pop_sex:region:uses_cannabis][uses_cannabis|pop_age:pop_race:pop_sex][pop_race|pop_age:pop_sex:region][pop_age|pop_sex:region][pop_sex|region][region]")
dag.get_node("pop_sex")["levels"] = drug_use_df.SEX.unique().tolist()
dag.get_node("pop_age")["levels"] = drug_use_df.AGE.unique().tolist()
dag.get_node("pop_race")["levels"] = drug_use_df.RACE.unique().tolist()
dag.get_node("uses_cannabis")["levels"] = ["n", "y"]
dag.get_node("incident")["levels"] = ["n", "y"]
dag.get_node("region")["levels"] = df["STATEREGION"].unique().tolist()

def get_usage_cpt(usage_name: str):
    cross = pd.crosstab(drug_use_df[usage_name], [drug_use_df.AGE, drug_use_df.RACE, drug_use_df.SEX], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / (len(cross.index) - 1)
    columns = [f"pop_{dem.lower()}" for dem in ["AGE", "RACE", "SEX"]]
    tuples = list(product(*[dag.get_node(col)["levels"] for col in columns]))
    new_index = pd.MultiIndex.from_tuples(tuples, names=columns)
    cross_sum = cross_sum[new_index]
    x = cross_sum.to_xarray()
    neg_x = 1 - x.values.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

cpt = get_usage_cpt("MJDAY30A")
binary_cpt = get_usage_cpt("MJBINARY")

# %%
dag.get_node("uses_cannabis")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("uses_cannabis"))
dag.get_node("uses_cannabis")["CPD"].parents = ['pop_age', 'pop_race', 'pop_sex']
dag.get_node("uses_cannabis")["CPD"].array = cpt
dag.get_node("uses_cannabis")["CPD"].rescale_probabilities()


# In[67]:

data_to_dag_names = {
    "FIPS": "county",
    "STATEREGION": "region",
    "AGE": "pop_age",
    "RACE": "pop_race",
    "SEX": "pop_sex",
}

df["SEX"] = df.SEX.apply(lambda x: x.lower())
df["RACE"] = df.RACE.apply(lambda x: x.lower())

def get_cpd(data, child, parents, norm=True):
    grouped = data.groupby([*parents, child])["frequency"].sum()
    if not norm:
        return grouped
    if parents:
        denom = data.groupby([*parents])["frequency"].sum()
        tuples = list(product(*[dag.get_node(data_to_dag_names[col])["levels"] for col in [*parents, child]]))
        new_index = pd.MultiIndex.from_tuples(tuples, names=[*parents, child])
        grouped = grouped[new_index]
        if len(parents) > 1:
            tuples_denom = list(product(*[dag.get_node(data_to_dag_names[col])["levels"] for col in [*parents]]))
            new_index_denom = pd.MultiIndex.from_tuples(tuples_denom, names=[*parents])
            denom = denom[new_index_denom]
        else:
            denom = denom[dag.get_node(data_to_dag_names[parents[0]])["levels"]]
        dims = (grouped / denom).to_xarray().values.shape
        return (grouped / denom).values.reshape(dims)
    else:
        grouped = grouped[dag.get_node(data_to_dag_names[child])["levels"]]
        return (grouped / data["frequency"].sum()).values

dag.get_node("pop_race")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_race"))
dag.get_node("pop_race")["CPD"].array = get_cpd(df, "RACE", ["AGE", "SEX", "STATEREGION"])
dag.get_node("pop_race")["CPD"].rescale_probabilities()

dag.get_node("pop_sex")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_sex"))
dag.get_node("pop_sex")["CPD"].array = get_cpd(df, "SEX", ["STATEREGION"])
dag.get_node("pop_sex")["CPD"].rescale_probabilities()

dag.get_node("pop_age")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("pop_age"))
dag.get_node("pop_age")["CPD"].array = get_cpd(df, "AGE", ["SEX", "STATEREGION"])
dag.get_node("pop_age")["CPD"].rescale_probabilities()

dag.get_node("region")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("region"))
dag.get_node("region")["CPD"].array = get_cpd(df, "STATEREGION", [])
dag.get_node("region")["CPD"].rescale_probabilities()

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
nibrs_df = pd.merge(nibrs_df, subregion_df, on="FIPS")
nibrs_df["STATEREGION"] = nibrs_df["State"] + "-" + nibrs_df["Region"]

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
    grouped_incidents = inc_df.groupby(cols + ["STATEREGION"]).size()
    expected_incidents = (pop_df.groupby(cols + ["STATEREGION"])["frequency"].sum() * cross_sum * 365)
    inc_cpt = grouped_incidents / expected_incidents
    columns = [f"pop_{dem.lower()}" for dem in dem_order] + ["region"]
    tuples = list(product(*[dag.get_node(col)["levels"] for col in columns]))
    new_index = pd.MultiIndex.from_tuples(tuples, names=columns)
    inc_cpt = inc_cpt[new_index]
    dims = inc_cpt.to_xarray().values.shape
    x = inc_cpt.values.reshape(dims)
    x = np.stack([np.zeros(x.shape), x.copy()])
    x = np.nan_to_num(x)
    neg_x = 1 - x.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

cols = ["AGE", "RACE", "SEX"]

inc_cpt = get_incident_cpt(df, drug_use_df, nibrs_df, cols)

dag.get_node("incident")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("incident"))
dag.get_node("incident")["CPD"].array = inc_cpt
dag.get_node("incident")["CPD"].parents = ['uses_cannabis', 'pop_age', 'pop_race', 'pop_sex', 'region']
dag.get_node("incident")["CPD"].rescale_probabilities()

# %%
def get_selection_by_vars(bn: DAG, county_name: str, race_level: str):
    local_bn = bn.copy()
    def _set_cpt(node: str, level: str) -> np.ndarray:
        idx = local_bn.get_node(node)["levels"].index(level)
        array = np.zeros(local_bn.get_node(node)["CPD"].array.shape)
        array[..., idx] = 1
        return array
    local_bn.get_node("region")["CPD"].array = _set_cpt("region", county_name)
    local_bn.get_node("pop_race")["CPD"].array = _set_cpt("pop_race", race_level)
    return collapse_posterior(local_bn, "incident")[1]

def get_ratio_by_vars(bn: DAG, region_name: str):
    return get_selection_by_vars(bn, region_name, "black") / get_selection_by_vars(bn, region_name, "white")


df["RATIOS"] = df.STATEREGION.apply(lambda x: get_ratio_by_vars(dag, x))

#TRACE EXTRAS

inc_per_sr = nibrs_df.groupby("STATEREGION").size().reset_index()
inc_per_sr = inc_per_sr.rename(columns={0: "incidents"})

race_ratio = df.groupby(["STATEREGION", "RACE"]).frequency.sum().reset_index()
race_ratio = race_ratio.pivot("STATEREGION", columns="RACE").reset_index()
race_ratio.columns = ["STATEREGION", "black", "white"]
race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]

df = pd.merge(df, inc_per_sr, on="STATEREGION", how="left")
df = pd.merge(df, race_ratio, on="STATEREGION", how="left")

# %%
df = df.round({'bwratio': 4, "RATIOS": 4})

import plotly.express as px

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(df, geojson=counties, locations='FIPS', color="RATIOS",
                           color_continuous_scale="Viridis",
                           scope="usa", hover_data = ["incidents", "bwratio"],
                            range_color=(0, 15),
                           labels={"RATIOS":'Incident Ratios by Race (Black/White)',
                           'incidents': "Number of incidents in the subregion",
                           "bwratio": "ratio of black / white population"}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
fig.update_traces(marker_line_width=0)
fig.write_html("../../choropleths/ratios_subregions_rm.html")

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
