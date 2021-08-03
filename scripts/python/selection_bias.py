"""
This python script investigates the DEMOGRAPHIC SELECTION-BIAS for
CANNABIS-RELATED incidents at a given GEOGRAPHIC RESOLUTION.
"""

########## IMPORTS ############

from typing import List
from typing_extensions import Literal
import numpy as np
import pandas as pd
import baynet
from baynet import DAG
from baynet.interventions import collapse_posterior
import seaborn as sns
from pathlib import Path
from itertools import product


########## RESOLUTION ##########

# Options: ["state", "region", "county", "agency"]
geographic_resolution = "agency"


##### LOAD DATASETS ######

data_path = Path(__file__).parent.parent.parent / "data"

# This is the county level census data. See "process_census_data.py".
# FIPS code is loaded in as an 'object' to avoid integer conversion.
census_df = pd.read_csv(data_path / "demographics" /
                        "county_census.csv", dtype={'FIPS': object})


# This is the procsssed NSDUH dataset. See "process_nsduh_data.py".
nsduh_df = pd.read_csv(data_path / "NSDUH" / "processed_cannabis_usage.csv")

# This is the processed NIBRS dataset. See "process_nibrs_data.py".
nibrs_df = pd.read_csv(data_path / "NIBRS" / "cannabis_processed.csv")


# Function to create dag and initialize levels dependent on desired geographic resolution.

# Dictionaries that converts between names of nodes and their data column names:

resolution_dict = {"state": "State",
                   "region": "STATEREGION", "county": "FIPS", "agency":  "ori"}

data_to_dag_names = {
    "FIPS": "county",
    "STATEREGION": "region",
    "ori": "agency",
    "State": "state",
    "AGE": "age_pop",
    "RACE": "race_pop",
    "SEX": "sex_pop",
}


############ HELPER FUNCTIONS #########

def create_dag(drug_use_df: pd.DataFrame, census_df: pd.DataFrame, resolution: Literal['state', 'region', 'county']):
    dag = DAG.from_modelstring(
        f"[incident|age_pop:race_pop:sex_pop:{resolution}:uses_cannabis][uses_cannabis|age_pop:race_pop:sex_pop][race_pop|age_pop:sex_pop:{resolution}][age_pop|sex_pop:{resolution}][sex_pop|{resolution}][{resolution}]")
    dag.get_node("sex_pop")["levels"] = drug_use_df.SEX.unique().tolist()
    dag.get_node("age_pop")["levels"] = drug_use_df.AGE.unique().tolist()
    dag.get_node("race_pop")["levels"] = drug_use_df.RACE.unique().tolist()
    dag.get_node("uses_cannabis")["levels"] = ["n", "y"]
    dag.get_node("incident")["levels"] = ["n", "y"]
    dag.get_node(resolution)[
        "levels"] = census_df[resolution_dict[resolution]].unique().tolist()
    return dag


# Function to create cannabis usage cpt.


def get_usage_cpt(drug_use_df: pd.DataFrame, usage_name: str):
    cross = pd.crosstab(drug_use_df[usage_name], [
                        drug_use_df.AGE, drug_use_df.RACE, drug_use_df.SEX], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / (len(cross.index) - 1)
    columns = [f"{dem.lower()}_pop" for dem in ["AGE", "RACE", "SEX"]]
    tuples = list(product(*[dag.get_node(col)["levels"] for col in columns]))
    new_index = pd.MultiIndex.from_tuples(tuples, names=columns)
    cross_sum = cross_sum[new_index]
    x = cross_sum.to_xarray()
    neg_x = 1 - x.values.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

# Function to selection "incident" cpt.


def get_incident_cpt(census_df: pd.DataFrame, nsduh_df: pd.DataFrame, nibrs_df: pd.DataFrame, dag: DAG, resolution: str, dem_order: List[str] = ["AGE", "RACE", "SEX"]):
    cross = pd.crosstab(nsduh_df.MJDAY30A, [
                        nsduh_df[col] for col in dem_order], normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / 30
    grouped_incidents = nibrs_df.groupby(
        sorted(dem_order + [resolution_dict[resolution]], key=str.casefold)).size()
    expected_incidents = (census_df.groupby(sorted(
        dem_order + [resolution_dict[resolution]], key=str.casefold))["frequency"].sum() * cross_sum * 365)
    inc_cpt = grouped_incidents / expected_incidents
    columns = sorted(
        [f"{dem.lower()}_pop" for dem in dem_order] + [resolution])
    tuples = list(product(*[dag.get_node(col)["levels"] for col in columns]))
    new_index = pd.MultiIndex.from_tuples(tuples, names=columns)
    inc_cpt = inc_cpt[new_index]
    dims = inc_cpt.to_xarray().values.shape
    x = inc_cpt.values.reshape(dims)
    x = np.stack([np.zeros(x.shape), x.copy()], axis=-1)
    x = np.nan_to_num(x)
    neg_x = 1 - x.copy()
    return np.stack([neg_x, x.copy()], axis=-1)


# Function to create the CPT of NODE given PARENTS

def get_cpd(census_df: pd.DataFrame, dag: DAG, child, parents, norm=True):
    parents = sorted(parents, key=str.casefold)
    grouped = census_df.groupby([*parents, child])["frequency"].sum()
    if not norm:
        return grouped
    if parents:
        denom = census_df.groupby([*parents])["frequency"].sum()
        tuples = list(product(
            *[dag.get_node(data_to_dag_names[col])["levels"] for col in [*parents, child]]))
        new_index = pd.MultiIndex.from_tuples(tuples, names=[*parents, child])
        grouped = grouped[new_index]
        if len(parents) > 1:
            tuples_denom = list(product(
                *[dag.get_node(data_to_dag_names[col])["levels"] for col in [*parents]]))
            new_index_denom = pd.MultiIndex.from_tuples(
                tuples_denom, names=[*parents])
            denom = denom[new_index_denom]
        else:
            denom = denom[dag.get_node(
                data_to_dag_names[parents[0]])["levels"]]
        dims = (grouped / denom).to_xarray().values.shape
        return (grouped / denom).values.reshape(dims)
    else:
        grouped = grouped[dag.get_node(data_to_dag_names[child])["levels"]]
        return (grouped / census_df["frequency"].sum()).values


def populate_cpd(dag: DAG, node: str, cpt: np.ndarray):
    dag.get_node(node)["CPD"] = baynet.parameters.ConditionalProbabilityTable(
        dag.get_node(node))
    dag.get_node(node)["CPD"].parents = sorted([p["name"] for p in dag.get_ancestors(
        node, only_parents=True)])
    dag.get_node(node)["CPD"].array = cpt
    dag.get_node(node)["CPD"].rescale_probabilities()

############ DAG CREATION #############

# Create DAG structure and Initialize levels.


dag = create_dag(nsduh_df, census_df, resolution=geographic_resolution)

# Populate Cannabis Usage CPT

populate_cpd(dag, "uses_cannabis", get_usage_cpt(nsduh_df, "MJDAY30A"))

# Populate demographic CPTs.

populate_cpd(dag, "race_pop", get_cpd(census_df, dag,
             "RACE", ["AGE", "SEX", resolution_dict[geographic_resolution]]))


populate_cpd(dag, "sex_pop", get_cpd(census_df, dag,
             "SEX", [resolution_dict[geographic_resolution]]))

populate_cpd(dag, "age_pop", get_cpd(census_df, dag,
             "AGE", ["SEX", resolution_dict[geographic_resolution]]))

populate_cpd(dag, geographic_resolution, get_cpd(census_df, dag,
             resolution_dict[geographic_resolution], []))

# Populate Incident CPT

populate_cpd(dag, "incident", get_incident_cpt(census_df, nsduh_df,
             nibrs_df, dag, geographic_resolution))


###### Conditioning #######

def get_selection_by_vars(bn: DAG, race_level: str, resolution: str, resolution_name: str):
    local_bn = bn.copy()

    def _set_cpt(node: str, level: str) -> np.ndarray:
        idx = local_bn.get_node(node)["levels"].index(level)
        array = np.zeros(local_bn.get_node(node)["CPD"].array.shape)
        array[..., idx] = 1
        return array
    local_bn.get_node(resolution)["CPD"].array = _set_cpt(
        resolution, resolution_name)
    local_bn.get_node("race_pop")["CPD"].array = _set_cpt(
        "race_pop", race_level)
    return collapse_posterior(local_bn, "incident")[1]


def get_ratio_by_vars(bn: DAG, resolution: str, resolution_name: str):
    return get_selection_by_vars(bn, "black", resolution, resolution_name) / get_selection_by_vars(bn, "white", resolution, resolution_name)




incident_counts = nibrs_df.groupby(resolution_dict[geographic_resolution]).size().to_frame("incidents").reset_index()

# incident_counts = incident_counts[incident_counts.incidents >= 10]

incident_counts["selection_ratio"] = incident_counts[resolution_dict[geographic_resolution]].apply(lambda x: get_ratio_by_vars(dag, geographic_resolution, x))

incident_counts.to_csv(data_path / "output"/ "agency_output_p.csv")

race_ratio = census_df.groupby([resolution_dict[geographic_resolution], "RACE"]).frequency.sum().reset_index()
race_ratio = race_ratio.pivot(resolution_dict[geographic_resolution], columns="RACE").reset_index()
race_ratio.columns = [resolution_dict[geographic_resolution], "black", "white"]
race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]

incident_counts = pd.merge(incident_counts, race_ratio, on=resolution_dict[geographic_resolution], how="left")


incident_counts.to_csv(data_path / "output"/ "agency_output.csv")