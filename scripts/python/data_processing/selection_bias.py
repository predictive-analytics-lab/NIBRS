"""
This python script investigates the DEMOGRAPHIC SELECTION-BIAS for
CANNABIS-RELATED incidents at a given GEOGRAPHIC RESOLUTION.
"""

########## IMPORTS ############

from functools import partial
import warnings
import argparse
from typing import Callable, List, Tuple
from typing_extensions import Literal
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
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
    expected_incidents = (census_df.groupby(sorted(
        dem_order + [resolution_dict[resolution]], key=str.casefold))["frequency"].sum() * cross_sum * 365)
    incidents = nibrs_df.groupby(sorted(dem_order + [resolution_dict[resolution]], key=str.casefold)).incidents.sum()
    inc_cpt = incidents / expected_incidents
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


def get_selection_ratio(dag: DAG, nibrs_df: pd.DataFrame, incident_df: pd.DataFrame, census_df: pd.DataFrame, geographic_resolution: str, min_incidents: int = 0) -> pd.DataFrame:
    
    non_smoothed_counts = incident_df.groupby(resolution_dict[geographic_resolution]).incidents.sum().reset_index()

    incident_counts = nibrs_df.groupby(resolution_dict[geographic_resolution]).incidents.sum().reset_index()

    incident_counts = incident_counts[incident_counts.incidents >= min_incidents]

    incident_counts["selection_ratio"] = incident_counts[resolution_dict[geographic_resolution]].apply(lambda x: get_ratio_by_vars(dag, geographic_resolution, x))
    
    incident_counts.drop(columns=["incidents"], inplace=True)
    
    incident_counts = pd.merge(incident_counts, non_smoothed_counts, on=[resolution_dict[geographic_resolution]], how="left")
    
    census_df.drop(columns=["ori"], inplace=True)
    
    census_df.drop_duplicates(inplace=True)
        
    popdf = census_df.groupby([resolution_dict[geographic_resolution]]).frequency.sum().reset_index()
    
    incident_counts = pd.merge(incident_counts, popdf, on=[resolution_dict[geographic_resolution]], how="left")
    
    urban_codes = pd.read_csv(data_path / "misc" / "NCHSURCodes2013.csv", usecols=["FIPS code", "2013 code"])
    urban_codes.rename(columns={"FIPS code":"FIPS", "2013 code": "urban_code"}, inplace=True)
    urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))
    incident_counts = pd.merge(incident_counts, urban_codes, on="FIPS", how="left")
    
    return incident_counts

def add_race_ratio(census_df: pd.DataFrame, incident_df: pd.DataFrame, geographic_resolution: str):
    race_ratio = census_df.groupby([resolution_dict[geographic_resolution], "race"]).frequency.sum().reset_index()
    race_ratio = race_ratio.pivot(resolution_dict[geographic_resolution], columns="race").reset_index()
    race_ratio.columns = [resolution_dict[geographic_resolution], "black", "white"]
    race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]

    incident_df = pd.merge(incident_df, race_ratio, on=resolution_dict[geographic_resolution], how="left")

    return incident_df

############ DATASET LOADING #############
def join_with_counties(df: pd.DataFrame, county_shp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mainland_states = list(set(range(57)) - {3, 7, 14, 43, 52})
    mainland_states = [str(i).zfill(2) for i in mainland_states]
    county_shp = county_shp[county_shp["statefp"].isin(mainland_states)]
    county_shp.rename(columns={"geoid":"FIPS"}, inplace=True)
    county_shp = county_shp.merge(df, on="FIPS", how="inner")
    return county_shp.reset_index()

def join_state_with_counties(df: pd.DataFrame, county_shp: gpd.GeoDataFrame, state: str) -> gpd.GeoDataFrame:
    county_shp = county_shp[county_shp["state_name"] == state]
    county_shp.rename(columns={"geoid":"FIPS"}, inplace=True)
    county_shp = county_shp.merge(df, on="FIPS", how="left")
    return county_shp.reset_index()


def load_datasets(years: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # This is the county level census data. See "process_census_data.py".
    # FIPS code is loaded in as an 'object' to avoid integer conversion.
    subprocess.run(["python", str((data_processing_path / "process_census_data.py").resolve()), "--year", years])
    subprocess.run(["python", str((data_processing_path / "process_nsduh_data.py").resolve()), "--year", years])
    subprocess.run(["python", str((data_processing_path / "process_nibrs_data.py").resolve()), "--year", years, "--resolution", "county"])

    census_df = pd.read_csv(data_path / "census" / f"census_processed_{years}.csv", dtype={'FIPS': object}, index_col=0)
    nsduh_df = pd.read_csv(data_path / "NSDUH" / f"nsduh_processed_{years}.csv")
    nibrs_df = pd.read_csv(data_path / "NIBRS" / f"incidents_processed_{years}.csv", dtype={'FIPS': object}, index_col=0)

    return census_df, nsduh_df, nibrs_df

def smooth_nibrs(nibrs_df: pd.DataFrame) -> pd.DataFrame:
    county_shp = gpd.read_file(data_path / "misc" / "us-county-boundaries.geojson")
    smoothed_df = None
    for state in nibrs_df.state.unique():
        if smoothed_df is not None:
            smoothed_df = smoothed_df.append(smooth_nibrs_state(nibrs_df[nibrs_df.state == state], county_shp))
        else:
            smoothed_df = smooth_nibrs_state(nibrs_df[nibrs_df.state == state], county_shp)
    return smoothed_df


def reporting(state_df: gpd.GeoDataFrame) -> pd.DataFrame:
    agency_df = pd.read_csv(data_path / "misc" / "agency_participation.csv", usecols=["ori", "nibrs_participated", "data_year"])
    agency_df = agency_df[agency_df.data_year == 2019]
    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t", usecols=["ORI9", "FIPS"], dtype={'FIPS': object})
    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})
    agency_df = pd.merge(agency_df, fips_ori_df, on="ori")
    reporting = agency_df.groupby("FIPS").nibrs_participated.apply(lambda x: "Y" if any(x == "Y") else "N").to_frame("reporting").reset_index()
    return state_df.merge(reporting, how="left", on="FIPS")


def smooth_nibrs_state(state_df: pd.DataFrame, county_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if len(state_df.FIPS.unique()) == 1:
        return state_df
    state = state_df.state.unique()[0]
    state_df, urban_df = filter_urban(state_df, 2)
    state_df.drop(["urban_code"], axis=1, inplace=True)
    urban_df.drop(["urban_code"], axis=1, inplace=True)
    if len(state_df) <= 0:
        return urban_df
    locations = state_df[["state", "state_region", "FIPS"]].drop_duplicates()
    state_df_p = state_df.pivot_table(index=["FIPS"], columns=["age", "race", "sex"], values="incidents")
    state_gdf_p = join_state_with_counties(state_df_p, county_gdf, state).sort_values(by=["FIPS"])
    qW = Queen.from_dataframe(state_gdf_p)
    amat, _ = qW.full()
    county_weights = get_county_weights(amat)
    
    state_gdf_p = reporting(state_gdf_p)
    state_gdf_p.loc[state_gdf_p.FIPS.isin(urban_df.FIPS.unique()), "reporting"] = "N"
    indicies = np.nonzero((state_gdf_p.reporting == "Y").values)[0]
    state_gdf_p = state_gdf_p.iloc[indicies, :]
    state_gdf_p = state_gdf_p.fillna(0)
    state_gdf_p.drop(["reporting"], axis=1, inplace=True)    
    county_weights = county_weights[np.ix_(indicies, indicies)]
    state_gdf_p.iloc[:, 22:] = county_weights @ state_gdf_p.iloc[:, 22:].values
                
    state_gdf_p = state_gdf_p[["FIPS",  *state_gdf_p.columns[22:].values]].melt(id_vars=["FIPS"], value_name="incidents")
    state_gdf_p['age'], state_gdf_p['race'], state_gdf_p['sex'] = state_gdf_p['variable'].str
    state_gdf_p.drop(["variable"], axis=1, inplace=True)
    state_gdf_p = state_gdf_p.merge(locations, on="FIPS", how="inner")
    return state_gdf_p.append(urban_df).reset_index()

def get_county_weights(state_amat: np.ndarray, max_path_length: int = 5, distance_weighting: Callable[[int], float] = lambda x,y: 0.0 if x==0 else 1/(y+1)) -> np.ndarray:
    vfunc = np.vectorize(distance_weighting)
    new_bool_amat = state_amat.copy()
    new_weighted_amat = vfunc(new_bool_amat, 1).astype(float)
    for path_length in range(2, max_path_length+1):
        paths = (np.linalg.matrix_power(state_amat, path_length) > 0).astype(int)
        added_paths = ((paths - new_bool_amat) > 0).astype(int)
        new_bool_amat += added_paths
        new_weighted_amat += vfunc(added_paths, path_length)
    np.fill_diagonal(new_weighted_amat, 1)
    return new_weighted_amat

def smooth_census(census_df: pd.DataFrame) -> gpd.GeoDataFrame:
    county_shp = gpd.read_file(data_path / "misc" / "us-county-boundaries.geojson")
    smoothed_df = None
    for state in census_df.state.unique():
        if smoothed_df is not None:
            smoothed_df = smoothed_df.append(smooth_census_state(census_df[census_df.state == state], county_shp))
        else:
            smoothed_df = smooth_census_state(census_df[census_df.state == state], county_shp)
    return smoothed_df

def filter_urban(df: pd.DataFrame, urban_level: int, coverage_required: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    urban_codes = pd.read_csv(data_path / "misc" / "NCHSURCodes2013.csv", usecols=["FIPS code", "2013 code"])
    urban_codes.rename(columns={"FIPS code":"FIPS", "2013 code": "urban_code"}, inplace=True)
    urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))
    df = pd.merge(df, urban_codes, on="FIPS", how="left")
    coverage = pd.read_csv(data_path / "misc" / "county_coverage.csv", dtype=str)
    df = pd.merge(df, coverage, on="FIPS", how="left")
    condition = (df.urban_code <= urban_level) & (df.coverage.astype(float) > coverage_required)
    return df[~condition], df[condition]

def smooth_census_state(state_df: pd.DataFrame, county_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    state_df, urban_df = filter_urban(state_df, 2)
    if len(state_df) <= 0:
        return urban_df
    joiner = state_df.groupby(["FIPS", "age", "race", "sex"]).first().sort_values(by=["age", "race", "sex"])
    joiner = joiner.drop(["ori"], axis=1)
    state_df_p = state_df.pivot_table(index=["FIPS"], columns=["age", "race", "sex"], values="frequency")
    state_df = state_df[["FIPS", "ori"]].drop_duplicates()
    state_gdf_p = join_with_counties(state_df_p.reset_index(), county_gdf).sort_values(by=["FIPS"])
    qW = Queen.from_dataframe(state_gdf_p)
    amat, _ = qW.full()
    county_weights = get_county_weights(amat)
    state_gdf_p.iloc[:, 22:] = county_weights @ state_gdf_p.iloc[:, 22:].values
    joiner["frequency"] = state_gdf_p.iloc[:, 22:].values.flatten("F")
    state_df = pd.merge(state_df, joiner.reset_index(), how="left", on="FIPS")
    return state_df.append(urban_df).reset_index()


def main(args):
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
        incident_df = nibrs_df.copy()
        population_df = census_df.copy()
        if args.smooth == "True":
            census_df = smooth_census(census_df)
            nibrs_df = smooth_nibrs(nibrs_df)
        bn = create_bn(nsduh_df, nibrs_df, census_df, args.resolution)
        temp_df = get_selection_ratio(bn, nibrs_df, incident_df, population_df, args.resolution, args.min_incidents)
        temp_df = add_race_ratio(census_df, temp_df, args.resolution)
        temp_df["year"] = year
        if selection_bias_df is not None:
            selection_bias_df = selection_bias_df.append(temp_df.copy())
        else:
            selection_bias_df = temp_df.copy()
    if args.smooth == "True":
            selection_bias_df.to_csv(data_path / "output" / f"selection_ratio_{args.resolution}_{args.year}_smoothed.csv")
    else:
        selection_bias_df.to_csv(data_path / "output" / f"selection_ratio_{args.resolution}_{args.year}.csv")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument("--resolution", help="The geographic resolution", default="state")
    parser.add_argument("--min_incidents", help="Minimum number of incidents to be included in the selection bias df.", default=0)
    parser.add_argument("--smooth", help="Minimum number of incidents to be included in the selection bias df.", default=False)
    parser.add_argument("--month-interpolation", help="Whether to interpolate over months", default=False)

    args=parser.parse_args()
    
    main(args)