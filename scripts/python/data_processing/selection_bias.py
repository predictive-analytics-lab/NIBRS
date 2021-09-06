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


###### Conditioning #######


def incident_users(nibrs_df: pd.DataFrame, census_df: pd.DataFrame, nsduh_df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    census_df = census_df[census_df.FIPS.isin(nibrs_df.FIPS)]
    census_df = census_df[["FIPS", "race", "age", "sex", "frequency"]]
    census_df = census_df.drop_duplicates()
    cross = pd.crosstab(nsduh_df.MJDAY30A, nsduh_df.race, normalize="columns")
    cross_mult = cross.multiply(cross.index, axis="rows")
    cross_sum = cross_mult.sum(axis="rows") / 30
    users = (census_df.groupby(sorted(["race"] + [resolution_dict[resolution]], key=str.casefold))["frequency"].sum() * cross_sum).to_frame("user_count").reset_index()
    incidents = nibrs_df.groupby(sorted(["race"] + [resolution_dict[resolution]], key=str.casefold)).incidents.sum().to_frame("incident_count").reset_index()
    users = users.pivot(index="FIPS", columns="race", values="user_count")
    incidents = incidents.pivot(index="FIPS", columns="race", values="incident_count")
    return incidents.merge(users, on="FIPS") 


def selection_ratio(incident_user_df: pd.DataFrame) -> pd.DataFrame:
    incident_user_df = incident_user_df.fillna(0).reset_index()
    incident_user_df = incident_user_df[incident_user_df.black_y > 0]
    incident_user_df = incident_user_df[incident_user_df.white_y > 0]
    incident_user_df["result"] = incident_user_df.apply(lambda x: wilson_selection(x["black_x"], x["black_y"], x["white_x"], x["white_y"]), axis=1)
    incident_user_df["selection_ratio"], incident_user_df["ci"] = incident_user_df.result.str
    incident_user_df = incident_user_df.drop(columns=["result", "black_x", "black_y","white_x","white_y"])
    return incident_user_df

def add_extra_information(selection_bias_df: pd.DataFrame, nibrs_df: pd.DataFrame, census_df: pd.DataFrame, geographic_resolution: str) -> pd.DataFrame:
        
    incidents = nibrs_df.groupby(resolution_dict[geographic_resolution]).incidents.sum().reset_index()
    
    # Remove agency duplicates
    census_df.drop(columns=["ori"], inplace=True)
    census_df.drop_duplicates(inplace=True)
        
    popdf = census_df.groupby([resolution_dict[geographic_resolution]]).frequency.sum().reset_index()
    
    if geographic_resolution == "county":
        urban_codes = pd.read_csv(data_path / "misc" / "NCHSURCodes2013.csv", usecols=["FIPS code", "2013 code"])
        urban_codes.rename(columns={"FIPS code":"FIPS", "2013 code": "urban_code"}, inplace=True)
        urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))
        selection_bias_df = selection_bias_df.merge(urban_codes, how="left", on="FIPS")
    
    selection_bias_df = selection_bias_df.merge(incidents, how="left", on=resolution_dict[geographic_resolution])
    selection_bias_df = selection_bias_df.merge(popdf, how="left", on=resolution_dict[geographic_resolution])

    return selection_bias_df

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

def wilson_error(n_s: int, n: int, z = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval
    
    param n_us: number of successes
    param n: total number of events
    param z: The z-value
    
    return: The lower and upper bound of the Wilson score interval
    """
    n_f = n - n_s
    denom = (n + z**2)
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2  / 4))
    return adjusted_p, ci

def wilson_selection(n_s_1, n_1, n_s_2, n_2) -> Tuple[float, float]:
    """
    Get the adjusted selection bias and wilson cis.
    """
    p1, e1 = wilson_error(n_s_1, n_1)
    p2, e2 = wilson_error(n_s_2, n_2)
    sr = p1 / p2
    ci = np.sqrt((e1 / p1)**2 + (e2 / p2)**2) * sr
    return sr, ci


def main(resolution: str, year: str, smooth: bool, bootstraps: int = -1):
    if "-" in year:
        years = year.split("-")
        years = range(int(years[0]), int(years[1]) + 1)

    else:
        years = [int(year)]

    selection_bias_df = pd.DataFrame()

    for year in years:
        
        ##### DATA LOADING ######
        try:
            census_df, nsduh_df, nibrs_df = load_datasets(str(year))
        except FileNotFoundError:
            warnings.warn(f"Data missing for {year}. Skipping.")
            continue
        
        
        # Copy to retain unsmoothed information
        incident_df = nibrs_df.copy()
        population_df = census_df.copy()
        
        if smooth:
            census_df = smooth_census(census_df)
            nibrs_df = smooth_nibrs(nibrs_df)
            
        ##### END DATA LOADING ######
        
        
        
        #### START ####
        incident_users_df = incident_users(nibrs_df, census_df, nsduh_df, resolution)
        incident_users_df = incident_users_df.fillna(0)
        
        def _sample_incidents(prob, trials, bootstraps: int, seed: int) -> int:
            np.random.seed(seed)
            if not np.isnan(prob):
                successes = np.random.binomial(n = trials, p = prob, size=bootstraps)
            else:
                successes = np.nan
            return successes / trials
                    
        if bootstraps < 0:
            temp_df = selection_ratio(incident_users_df)
        else:
            black_count = []
            white_count = []
            incident_users_df["black_prob"] = incident_users_df.black_x / (incident_users_df.black_y * 365.0)
            incident_users_df["white_prob"] = incident_users_df.white_x / (incident_users_df.white_y * 365.0)
            for i, row in incident_users_df.iterrows():
                white_vector = _sample_incidents(row["white_prob"], row["white_y"], bootstraps, seed=1)
                black_vector = _sample_incidents(row["black_prob"], row["black_y"], bootstraps, seed=1)
                selection_ratios = black_vector / white_vector
                selection_ratios = np.nan_to_num(selection_ratios, nan=np.inf, posinf=np.inf)
                incident_users_df.loc[i, "selection_ratio"] = row["black_prob"] / row["white_prob"]
                incident_users_df.loc[i, "lb"] = np.nan_to_num(np.quantile(selection_ratios, 0.025), nan=np.inf)
                incident_users_df.loc[i, "ub"] = np.nan_to_num(np.quantile(selection_ratios, 0.975), nan=np.inf)
            temp_df = incident_users_df.reset_index()[["FIPS", "selection_ratio", "lb", "ub"]]
        
        
        temp_df = add_extra_information(temp_df, incident_df, population_df, resolution)
        temp_df = add_race_ratio(census_df, temp_df, resolution)
        #### END ###
        
        temp_df["year"] = year
        selection_bias_df = selection_bias_df.append(temp_df.copy())
    if smooth:
        filename = f"selection_ratio_{resolution}_{year}_smoothed.csv"
    else:
        filename = f"selection_ratio_{resolution}_{year}.csv"
    selection_bias_df.to_csv(data_path / "output" / filename)

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument("--resolution", help="The geographic resolution", default="state")
    parser.add_argument("--smooth", help="Minimum number of incidents to be included in the selection bias df.", default=False)
    parser.add_argument("--bootstraps", help="The number of bootstraps to perform in order to get a confidence interval", default=-1)

    args = parser.parse_args()
    
    main(resolution = args.resolution, year = args.year, smooth = args.smooth == "True", bootstraps = int(args.bootstraps))