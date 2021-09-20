"""
This python script investigates the DEMOGRAPHIC SELECTION-BIAS for
CANNABIS-RELATED incidents at a given GEOGRAPHIC RESOLUTION.
"""

########## IMPORTS ############

from functools import partial
import warnings
import argparse
from typing import Callable, List, Optional, Tuple
from typing_extensions import Literal
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from itertools import product
import subprocess

from process_census_data import get_census_data
from process_nsduh_data import get_nsduh_data
from process_nibrs_data import load_and_process_nibrs
from smooth import smooth_data

##### LOAD DATASETS ######

base_path = Path(__file__).parent.parent.parent.parent

data_processing_path = base_path / "scripts" / "python" / "data_processing"

data_path = base_path / "data"

# Function to create dag and initialize levels dependent on desired geographic resolution.

# Dictionaries that converts between names of nodes and their data column names:

resolution_dict = {"state": "state",
                   "state_region": "state_region", "county": "FIPS", "agency":  "ori"}


###### Conditioning #######

def weighted_average_aggregate(group: pd.DataFrame):
    return (group["MJDAY30A"] * group["ANALWT_C"]).sum() / 30

def incident_users(nibrs_df: pd.DataFrame, census_df: pd.DataFrame, nsduh_df: pd.DataFrame, resolution: str, poverty: bool, urban: bool, usage_name: str = "MJDAY30A") -> pd.DataFrame:
    vars = ["race", "age", "sex"]
    if poverty:
        vars += ["poverty"]
    if urban:
        vars += ["urbancounty"]
    if poverty and urban:
        raise ValueError("Only EITHER poverty OR urban may be used.")
    
    census_df = census_df[census_df[resolution_dict[resolution]].isin(nibrs_df[resolution_dict[resolution]])]
    census_df = census_df[vars + ["frequency", resolution_dict[resolution]]]
    census_df = census_df.drop_duplicates()
    
    census_df = census_df.merge(nsduh_df, on=vars, how="left")
        
    #prob_dem = (census_df.groupby([*vars, resolution_dict[resolution]]).frequency.sum() / census_df.groupby(["race", resolution_dict[resolution]]).frequency.sum()).to_frame("prob_dem").reset_index()
    
    #census_df = census_df.merge(prob_dem, on=[resolution_dict[resolution], *vars])
    
    census_df["users"] = census_df["frequency"] * census_df["prob_usage_one_dat"] * 365.0
    
    #census_df["users_var"] = (census_df["prob_dem"] ** 2) * census_df["users"] * (census_df["prob_usage_one_dat_se"] ** 2) * 365.0

    census_df["users_var"] = census_df["users"] * (census_df["prob_usage_one_dat_se"] ** 2) * 365.0

    users = census_df.groupby(["race", resolution_dict[resolution]]).users.sum().to_frame("user_count").reset_index()
    
    users_var = census_df.groupby(["race", resolution_dict[resolution]]).users_var.sum().to_frame("user_var").reset_index()
        
    users = users.pivot(index=resolution_dict[resolution], columns="race", values="user_count")
    users_var = users_var.pivot(index=resolution_dict[resolution], columns="race", values="user_var")
    
    incidents = nibrs_df.groupby(sorted(["race"] + [resolution_dict[resolution]], key=str.casefold)).incidents.sum().to_frame("incident_count").reset_index()
    incidents = incidents.pivot(index=resolution_dict[resolution], columns="race", values="incident_count")
    incidents = incidents.merge(users, on=resolution_dict[resolution])
    df =  incidents.merge(users_var, on=resolution_dict[resolution]) 
    df = df.rename({"black": "black_users_variance", "white": "white_users_variance"}, axis=1)
    return df



def selection_ratio(incident_user_df: pd.DataFrame, wilson: bool) -> pd.DataFrame:
    incident_user_df = incident_user_df.fillna(0).reset_index()
    incident_user_df = incident_user_df[incident_user_df.black_y > 0]
    incident_user_df = incident_user_df[incident_user_df.white_y > 0]
    if wilson:
        incident_user_df["result"] = incident_user_df.apply(lambda x: wilson_selection(x["black_x"], x["black_y"], x["white_x"], x["white_y"]), axis=1)
        incident_user_df["selection_ratio"], incident_user_df["ci"] = incident_user_df.result.str
    else:
        incident_user_df["selection_ratio"] = incident_user_df.apply(lambda x: simple_selection_ratio(x["black_x"], x["black_y"], x["white_x"], x["white_y"]), axis=1)
    incident_user_df = incident_user_df.rename(columns={"black_x": "black_incidents", "black_y": "black_users","white_x": "white_incidents","white_y": "white_users"})
    return incident_user_df

def add_extra_information(selection_bias_df: pd.DataFrame, nibrs_df: pd.DataFrame, census_df: pd.DataFrame, geographic_resolution: str, year: int) -> pd.DataFrame:
        
    incidents = nibrs_df.groupby(resolution_dict[geographic_resolution]).incidents.sum().reset_index()
    
    # Remove agency duplicates
    census_df.drop(columns=["ori"], inplace=True)
    census_df.drop_duplicates(inplace=True)
        
    popdf = census_df.groupby([resolution_dict[geographic_resolution]]).frequency.sum().reset_index()
    
    if geographic_resolution == "county" and isinstance(year, int):
        urban_codes = pd.read_csv(data_path / "misc" / "NCHSURCodes2013.csv", usecols=["FIPS code", "2013 code"])
        urban_codes.rename(columns={"FIPS code":"FIPS", "2013 code": "urban_code"}, inplace=True)
        urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))
        selection_bias_df = selection_bias_df.merge(urban_codes, how="left", on="FIPS")
        coverage = pd.read_csv(data_path / "misc" / "county_coverage.csv", usecols=["FIPS", "coverage", "year"], dtype={"FIPS": str, "year":int})
        selection_bias_df["year"] = year
        selection_bias_df = selection_bias_df.merge(coverage, how="left", on=["FIPS", "year"])

    selection_bias_df = selection_bias_df.merge(incidents, how="left", on=resolution_dict[geographic_resolution])
    selection_bias_df = selection_bias_df.merge(popdf, how="left", on=resolution_dict[geographic_resolution])

    return selection_bias_df

def add_race_ratio(census_df: pd.DataFrame, incident_df: pd.DataFrame, geographic_resolution: str):
    if "ori" in census_df.columns:
        census_df.drop(columns=["ori"], inplace=True)
        census_df.drop_duplicates(inplace=True)
    race_ratio = census_df.groupby([resolution_dict[geographic_resolution], "race"]).frequency.sum().reset_index()
    race_ratio = race_ratio.pivot(resolution_dict[geographic_resolution], columns="race").reset_index()
    race_ratio.columns = [resolution_dict[geographic_resolution], "black", "white"]
    race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]

    incident_df = pd.merge(incident_df, race_ratio, on=resolution_dict[geographic_resolution], how="left")

    return incident_df

############ DATASET LOADING #############

def load_datasets(years: str, resolution: str, poverty: bool, urban: bool, all_incidents: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    census_df = get_census_data(years = years, poverty = poverty, urban = urban)
    nsduh_df = get_nsduh_data(years=years, poverty=poverty, urban=urban)
    nibrs_df = load_and_process_nibrs(years=years, resolution=resolution, all_incidents=all_incidents)
    return census_df, nsduh_df, nibrs_df

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

def simple_selection_ratio(n_s_1, n_1, n_s_2, n_2) -> float:
    # Add 2 to avoid division by zero - bad solution, but temporary.
    n_s_1 += 0.1
    n_s_2 += 0.1
    n_1 += 1
    n_2 += 1
    p1 = n_s_1 / n_1
    p2 = n_s_2 / n_2
    return p1 / p2

def wilson_selection(n_s_1, n_1, n_s_2, n_2) -> Tuple[float, float]:
    """
    Get the adjusted selection bias and wilson cis.
    """
    p1, e1 = wilson_error(n_s_1, n_1)
    p2, e2 = wilson_error(n_s_2, n_2)
    sr = p1 / p2
    ci = np.sqrt((e1 / p1)**2 + (e2 / p2)**2) * sr
    return sr, ci

def bootstrap_selection(incident_users_df: pd.DataFrame, bootstraps: int):
    def _sample_incidents(prob, trials, bootstraps: int, seed: int) -> int:
        prob = np.min([prob, 1])
        np.random.seed(seed)
        if not np.isnan(prob):
            successes = np.random.binomial(n = trials, p = prob, size=bootstraps)
        else:
            successes = np.nan
        return successes
    black_count = []
    white_count = []
    incident_users_df.black_x += 0.5
    incident_users_df.white_x += 0.5
    incident_users_df.black_y += 1
    incident_users_df.white_y += 1
    incident_users_df["black_prob"] = incident_users_df.black_x / (incident_users_df.black_y)
    incident_users_df["white_prob"] = incident_users_df.white_x / (incident_users_df.white_y)
    for i, row in incident_users_df.iterrows():
        white_vector = _sample_incidents(row["white_prob"], row["white_y"], bootstraps, seed=1)
        black_vector = _sample_incidents(row["black_prob"], row["black_y"], bootstraps, seed=1)
        selection_ratios = (black_vector / row["black_y"]) / (white_vector / row["white_y"])
        selection_ratios = np.nan_to_num(selection_ratios, nan=np.inf, posinf=np.inf)
        incident_users_df.loc[i, "selection_ratio"] = row["black_prob"] / row["white_prob"]
        incident_users_df.loc[i, "lb"] = np.nan_to_num(np.quantile(selection_ratios, 0.025), nan=np.inf)
        incident_users_df.loc[i, "ub"] = np.nan_to_num(np.quantile(selection_ratios, 0.975), nan=np.inf)
        incident_users_df.loc[i, "black_incident_variance"] = np.nan_to_num(np.var(black_vector), nan=0)
        incident_users_df.loc[i, "white_incident_variance"] = np.nan_to_num(np.var(white_vector), nan=0)
    incident_users_df = incident_users_df.rename(columns={"black_x": "black_incidents", "black_y": "black_users","white_x": "white_incidents","white_y": "white_users"})
    return incident_users_df.reset_index()[["FIPS", "selection_ratio", "lb", "ub", "black_incidents", "black_users_variance", "white_users_variance", "black_incident_variance", "white_incident_variance", "black_users", "white_incidents", "white_users"]]

def delta_method(df: pd.DataFrame):
    """['FIPS', 'selection_ratio', 'lb', 'ub', 'black_incidents',
       'black_users_variance', 'white_users_variance',
       'black_incident_variance', 'white_incident_variance', 'black_users',
       'white_incidents', 'white_users']"""
    var_log_sr = 1 / (df.black_incidents ** 2) * df.black_incident_variance + 1 / (df.white_incidents ** 2) * df.white_incident_variance + \
    1 / (df.black_users ** 2) * df.black_users_variance +  1 / (df.white_users ** 2) * df.white_users_variance 
    df['ub'] = df.selection_ratio * np.exp(1.96 * np.sqrt(var_log_sr))
    df['lb'] = df.selection_ratio * np.exp(- 1.96 * np.sqrt(var_log_sr))
    return df

def main(resolution: str, year: str, smooth: bool, ci: Optional[Literal['none', 'wilson', 'bootstrap']], bootstraps: int, poverty: bool, urban: bool, all_incidents: bool, urban_filter: int, smoothing_param: int, group_years: bool):
    
    if not group_years:
        if "-" in year:
            years = year.split("-")
            years = range(int(years[0]), int(years[1]) + 1)

        else:
            years = [int(year)]
    else:
        years=[year]

    selection_bias_df = pd.DataFrame()

    for yi in years:
        
        ##### DATA LOADING ######
        
        try:
            census_df, nsduh_df, nibrs_df = load_datasets(str(yi), resolution, poverty, urban, all_incidents)
        except FileNotFoundError:
            warnings.warn(f"Data missing for {yi}. Skipping.")
            continue
        # Copy to retain unsmoothed information
        incident_df = nibrs_df.copy()
        population_df = census_df.copy()
        

        if smooth:
            nibrs_df, census_df = smooth_data(nibrs_df, census_df, urban=urban, poverty=poverty, urban_filter=urban_filter, smoothing_param=smoothing_param)
            
        ##### END DATA LOADING ######
        
        #### START ####
        incident_users_df = incident_users(nibrs_df, census_df, nsduh_df, resolution, poverty, urban)
        incident_users_df = incident_users_df.fillna(0)
                            
        if ci == "none":
            temp_df = selection_ratio(incident_users_df, wilson=False)
        elif ci == "wilson":
            temp_df = selection_ratio(incident_users_df, wilson=True)
        else:
            temp_df = bootstrap_selection(incident_users_df, bootstraps)
            temp_df = delta_method(temp_df)
        
        temp_df = add_extra_information(temp_df, incident_df, population_df, resolution, yi)
        temp_df = add_race_ratio(population_df, temp_df, resolution)
        #### END ###
        
        temp_df["year"] = yi
        selection_bias_df = selection_bias_df.append(temp_df.copy())
        
    filename = f"selection_ratio_{resolution}_{year}"
    
    if group_years:
        filename += "_grouped"
        
    if ci == "bootstrap":
        filename += f"_bootstraps_{bootstraps}"
    elif ci == "wilson":
        filename += "_wilson"
        
    if poverty:
        filename += "_poverty"
        
    if urban:
        filename += "_urban"
        
    if all_incidents:
        filename += "_all_incidents"
        
    if urban_filter != 2:
        filename += f"_urban_filter_{urban_filter}"
    
    if smooth:
        filename += "_smoothed"
        if smoothing_param != 1:
            filename += f"_{str(smoothing_param).replace('.', '-')}"
        
    filename += ".csv"
    selection_bias_df.to_csv(data_path / "output" / filename)

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range. e.g. 2015-2019, 2013, etc.", default="2019")
    parser.add_argument("--resolution", help="The geographic resolution. Options: county, region, state, agency. Default = county.", default="county")
    parser.add_argument("--ci", type=str, help="The type of confidence interval to use. Options: [None, bootstrap, wilson]. Default: None", default="none")
    parser.add_argument("--bootstraps", type=int, help="The number of bootstraps to perform in order to get a confidence interval.", default=-1)
    parser.add_argument("--smooth", help="Flag indicating whether to perform geospatial smoothing. Default = False.", default=False, action='store_true')
    parser.add_argument("--poverty", help="Whether to include poverty in the usage model.", default=False, action='store_true')
    parser.add_argument("--urban", help="Whether to include urban/rural definitions in the usage model.", default=False, action='store_true')
    parser.add_argument("--all_incidents", help="Whether to include all incident types that involve cannabis in some way.", default=False, action='store_true')
    parser.add_argument("--urban_filter", type=int, help="The level of urban areas to filter out when smoothing; UA >= N are removed before smoothing. Default=2.", default=2)
    parser.add_argument("--smoothing_param", type=float, help="Smoothing is currently 1/(d+1)**p. Where d is (graph) distance. This parameter sets p. Default=1", default=1)
    parser.add_argument("--group_years", help="Whether to group years into one poor for SR and CI calculation.", default=False, action='store_true')


    args = parser.parse_args()
    
    if int(args.bootstraps) > 0 and args.ci != "bootstrap":
        raise ValueError("If bootstraps is specified, ci must be bootstrap.")
        
    main(resolution = args.resolution,
         year = args.year,
         smooth = args.smooth,
         ci=args.ci.lower(),
         bootstraps = int(args.bootstraps),
         poverty = args.poverty, 
         urban = args.urban,
         all_incidents = args.all_incidents,
         urban_filter=args.urban_filter,
         smoothing_param=args.smoothing_param,
         group_years=args.group_years)