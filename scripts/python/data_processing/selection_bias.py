"""
This python script investigates the DEMOGRAPHIC SELECTION-BIAS for
CANNABIS-RELATED incidents at a given GEOGRAPHIC RESOLUTION.
"""

########## IMPORTS ############

import warnings
import argparse
from typing import List, Tuple
from typing_extensions import Literal
import numpy as np
import pandas as pd
import baynet
from baynet import DAG
from baynet.interventions import collapse_posterior
import seaborn as sns
from pathlib import Path
from itertools import product
import subprocess

##### LOAD DATASETS ######

base_path = Path(__file__).parent.parent.parent.parent

data_processing_path = base_path / "scripts" / "python" / "data_processing"

data_path = base_path / "data"

# Function to create dag and initialize levels dependent on desired geographic resolution.

# Dictionaries that converts between names of nodes and their data column names:

resolution_dict = {"state": "state",
                   "state_region": "state_region", "county": "FIPS", "agency":  "ori"}

data_to_dag_names = {
    "FIPS": "county",
    "state_region": "state_region",
    "ori": "agency",
    "state": "state",
    "age": "age_pop",
    "race": "race_pop",
    "sex": "sex_pop",
}


############ HELPER FUNCTIONS #########

def create_dag(drug_use_df: pd.DataFrame, census_df: pd.DataFrame, resolution: Literal['state', 'state_region', 'county']):
    dag = DAG.from_modelstring(
        f"[incident|age_pop:race_pop:sex_pop:{resolution}:uses_cannabis][uses_cannabis|age_pop:race_pop:sex_pop][race_pop|age_pop:sex_pop:{resolution}][age_pop|sex_pop:{resolution}][sex_pop|{resolution}][{resolution}]")
    dag.get_node("sex_pop")["levels"] = drug_use_df.sex.unique().tolist()
    dag.get_node("age_pop")["levels"] = drug_use_df.age.unique().tolist()
    dag.get_node("race_pop")["levels"] = drug_use_df.race.unique().tolist()
    dag.get_node("uses_cannabis")["levels"] = ["n", "y"]
    dag.get_node("incident")["levels"] = ["n", "y"]
    dag.get_node(resolution)[
        "levels"] = census_df[resolution_dict[resolution]].unique().tolist()
    return dag


# Function to create cannabis usage cpt.

def weighted_average_aggregate(group: pd.DataFrame):
    return (group["MJDAY30A"] * group["ANALWT_C"]).sum() / 30

def get_usage_cpt(dag: DAG, drug_use_df: pd.DataFrame, usage_name: str):
    group_weights = drug_use_df.groupby(["age", "race", "sex", usage_name])["ANALWT_C"].sum() / drug_use_df.groupby(["age", "race", "sex"])["ANALWT_C"].sum()
    group_weights = group_weights.reset_index()
    group_weights = group_weights.groupby(["age", "race", "sex"]).apply(weighted_average_aggregate).to_frame("MJDAY").reset_index()
    columns = [f"{dem.lower()}_pop" for dem in ["age", "race", "sex"]]
    tuples = list(product(*[dag.get_node(col)["levels"] for col in columns]))
    new_index = pd.MultiIndex.from_tuples(tuples, names=["age", "race", "sex"])
    cross_sum = group_weights.groupby(["age", "race", "sex"]).mean().reindex(new_index)
    x = cross_sum.squeeze().to_xarray()
    neg_x = 1 - x.values.copy()
    return np.stack([neg_x, x.copy()], axis=-1)

# Function to selection "incident" cpt.


def get_incident_cpt(census_df: pd.DataFrame, nsduh_df: pd.DataFrame, nibrs_df: pd.DataFrame, dag: DAG, resolution: str, dem_order: List[str] = ["age", "race", "sex"]):
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

def create_bn(nsduh_df: pd.DataFrame, nibrs_df: pd.DataFrame, census_df: pd.DataFrame, geographic_resolution: str) -> DAG:

    dag = create_dag(nsduh_df, census_df, resolution=geographic_resolution)

    # Populate Cannabis Usage CPT

    populate_cpd(dag, "uses_cannabis", get_usage_cpt(dag, nsduh_df, "MJDAY30A"))

    # Populate demographic CPTs.

    populate_cpd(dag, "race_pop", get_cpd(census_df, dag,
                "race", ["age", "sex", resolution_dict[geographic_resolution]]))


    populate_cpd(dag, "sex_pop", get_cpd(census_df, dag,
                "sex", [resolution_dict[geographic_resolution]]))

    populate_cpd(dag, "age_pop", get_cpd(census_df, dag,
                "age", ["sex", resolution_dict[geographic_resolution]]))

    populate_cpd(dag, geographic_resolution, get_cpd(census_df, dag,
                resolution_dict[geographic_resolution], []))

    # Populate Incident CPT

    populate_cpd(dag, "incident", get_incident_cpt(census_df, nsduh_df,
                nibrs_df, dag, geographic_resolution))
    
    return dag


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


def get_selection_ratio(dag: DAG, nibrs_df: pd.DataFrame, geographic_resolution: str, min_incidents: int = 0) -> pd.DataFrame:

    incident_counts = nibrs_df.groupby(resolution_dict[geographic_resolution]).size().to_frame("incidents").reset_index()
        
    incident_counts = incident_counts[incident_counts.incidents >= min_incidents]

    incident_counts["selection_ratio"] = incident_counts[resolution_dict[geographic_resolution]].apply(lambda x: get_ratio_by_vars(dag, geographic_resolution, x))

    return incident_counts

def add_race_ratio(census_df: pd.DataFrame, incident_df: pd.DataFrame, geographic_resolution: str):

    race_ratio = census_df.groupby([resolution_dict[geographic_resolution], "race"]).frequency.sum().reset_index()
    race_ratio = race_ratio.pivot(resolution_dict[geographic_resolution], columns="race").reset_index()
    race_ratio.columns = [resolution_dict[geographic_resolution], "black", "white"]
    race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]
    
    incident_df = pd.merge(incident_df, race_ratio, on=resolution_dict[geographic_resolution], how="left")

    return incident_df

############ DATASET LOADING #############

def load_datasets(years: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # This is the county level census data. See "process_census_data.py".
    # FIPS code is loaded in as an 'object' to avoid integer conversion.
    if (data_path / "census" / f"census_processed_{years}.csv").exists():
        census_df = pd.read_csv(data_path / "census" / f"census_processed_{years}.csv", dtype={'FIPS': object})
    else:
        subprocess.run(["python", str((data_processing_path / "process_census_data.py").resolve()), "--year", years])
        census_df = pd.read_csv(data_path / "census" / f"census_processed_{years}.csv", dtype={'FIPS': object})

    if (data_path / "NSDUH" / f"nsduh_processed_{years}.csv").exists():
        nsduh_df = pd.read_csv(data_path / "NSDUH" / f"nsduh_processed_{years}.csv")
    else:
        subprocess.run(["python", str((data_processing_path / "process_nsduh_data.py").resolve()), "--year", years])
        nsduh_df = pd.read_csv(data_path / "NSDUH" / f"nsduh_processed_{years}.csv")

    if (data_path / "NIBRS" / f"incidents_processed_{years}.csv").exists():
        nibrs_df = pd.read_csv(data_path / "NIBRS" / f"incidents_processed_{years}.csv", dtype={'FIPS': object})
    else:
        subprocess.run(["python", str((data_processing_path / "process_nibrs_data.py").resolve()), "--year", years])
        nibrs_df = pd.read_csv(data_path / "NIBRS" / f"incidents_processed_{years}.csv")
    return census_df, nsduh_df, nibrs_df


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument("--resolution", help="The geographic resolution", default="state")
    parser.add_argument("--min_incidents", help="Minimum number of incidents to be included in the selection bias df.", default=0)

    args=parser.parse_args()

    if "-" in args.year:
        years = args.year.split("-")
        years = range(int(years[0]), int(years[1]) + 1)

    else:
        years = [int(args.year)]

    selection_bias_df = None

    for year in years:
        try:
            census_df, nsduh_df, nibrs_df = load_datasets(str(year))
        except FileNotFoundError:
            warnings.warn(f"Data missing for {year}. Skipping.")
            continue
        bn = create_bn(nsduh_df, nibrs_df, census_df, args.resolution)
        temp_df = get_selection_ratio(bn, nibrs_df, args.resolution, args.min_incidents)
        temp_df = add_race_ratio(census_df, temp_df, args.resolution)
        temp_df["year"] = year
        if selection_bias_df is not None:
            selection_bias_df = selection_bias_df.append(temp_df.copy())
        else:
            selection_bias_df = temp_df.copy()
    selection_bias_df.to_csv(data_path / "output" / f"selection_ratio_{args.resolution}_{args.year}.csv")