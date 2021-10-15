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
from pathlib import Path
from itertools import product
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns

from process_census_data import get_census_data
from process_nsduh_data import get_nsduh_data
from process_nibrs_data import load_and_process_nibrs
from process_ucr_data import get_ucr_data
from geographic_smoothing import smooth_data

##### LOAD DATASETS ######

base_path = Path(__file__).parent.parent.parent.parent

data_processing_path = base_path / "scripts" / "python" / "data_processing"

data_path = base_path / "data"

# Function to create dag and initialize levels dependent on desired geographic resolution.

# Dictionaries that converts between names of nodes and their data column names:

resolution_dict = {
    "state": "state",
    "state_region": "state_region",
    "county": "FIPS",
    "agency": "ori",
}


###### Conditioning #######


def weighted_average_aggregate(group: pd.DataFrame):
    return (group["MJDAY30A"] * group["ANALWT_C"]).sum() / 30


def incident_users(
    nibrs_df: pd.DataFrame,
    census_df: pd.DataFrame,
    nsduh_df: pd.DataFrame,
    resolution: str,
    poverty: bool,
    metro: bool,
    years: bool = False,
    ucr: bool = False,
    dont_aggregate: bool = False,
) -> pd.DataFrame:
    vars = ["race", "age", "sex"]
    if poverty:
        vars += ["poverty"]
    if metro:
        vars += ["metrocounty"]
    if poverty and metro:
        raise ValueError("Only EITHER poverty OR metro may be used.")
    if years:
        vars += ["year"]
    # 30053

    census_df = census_df[
        census_df[resolution_dict[resolution]].isin(
            nibrs_df[resolution_dict[resolution]]
        )
    ]
    census_df = census_df[vars + ["frequency", resolution_dict[resolution]]]
    census_df = census_df.drop_duplicates()

    census_df = census_df.merge(nsduh_df, on=vars, how="left")

    # prob_dem = (census_df.groupby([*vars, resolution_dict[resolution]]).frequency.sum() / census_df.groupby(["race", resolution_dict[resolution]]).frequency.sum()).to_frame("prob_dem").reset_index()

    # census_df = census_df.merge(prob_dem, on=[resolution_dict[resolution], *vars])

    census_df["users"] = census_df["frequency"] * census_df["MJ"] * 365.0

    # census_df["users_var"] = (census_df["prob_dem"] ** 2) * census_df["users"] * (census_df["prob_usage_one_dat_se"] ** 2) * 365.0

    census_df["users_var"] = census_df["users"] * (census_df["MJ_SE"] ** 2) * 365.0

    if dont_aggregate:
        vars = list(set(vars) - {"poverty", "metrocounty"})
        census_df = (
            census_df.groupby([*vars, resolution_dict[resolution]])
            .sum()
            .reset_index()[[*vars, resolution_dict[resolution], "users", "users_var"]]
        )
        return census_df.merge(nibrs_df, on=[*vars, resolution_dict[resolution]])

    users = (
        census_df.groupby(["race", resolution_dict[resolution]])
        .users.sum()
        .to_frame("user_count")
        .reset_index()
    )

    users_var = (
        census_df.groupby(["race", resolution_dict[resolution]])
        .users_var.sum()
        .to_frame("user_var")
        .reset_index()
    )

    users = users.pivot(
        index=resolution_dict[resolution], columns="race", values="user_count"
    )
    users_var = users_var.pivot(
        index=resolution_dict[resolution], columns="race", values="user_var"
    )

    if not ucr:
        incidents = (
            nibrs_df.groupby(
                sorted(["race"] + [resolution_dict[resolution]], key=str.casefold)
            )
            .incidents.sum()
            .to_frame("incident_count")
            .reset_index()
        )
        incidents = incidents.pivot(
            index=resolution_dict[resolution], columns="race", values="incident_count"
        )
    else:
        incidents = nibrs_df
    incidents = incidents.merge(users, on=resolution_dict[resolution])
    df = incidents.merge(users_var, on=resolution_dict[resolution])
    df = df.rename(
        {
            "black": "black_users_variance",
            "white": "white_users_variance",
            "black_x": "black_incidents",
            "black_y": "black_users",
            "white_x": "white_incidents",
            "white_y": "white_users",
        },
        axis=1,
    )
    return df


def selection_ratio(incident_user_df: pd.DataFrame, wilson: bool) -> pd.DataFrame:
    incident_user_df = incident_user_df.fillna(0).reset_index()
    incident_user_df = incident_user_df[incident_user_df.black_users > 0]
    incident_user_df = incident_user_df[incident_user_df.white_users > 0]
    if wilson:
        incident_user_df["result"] = incident_user_df.apply(
            lambda x: wilson_selection(
                x["black_incidents"],
                x["black_users"],
                x["white_incidents"],
                x["white_users"],
            ),
            axis=1,
        )
        (
            incident_user_df["selection_ratio"],
            incident_user_df["ci"],
        ) = incident_user_df.result.str
    else:
        incident_user_df["selection_ratio"] = incident_user_df.apply(
            lambda x: simple_selection_ratio(
                x["black_incidents"],
                x["black_users"],
                x["white_incidents"],
                x["white_users"],
            ),
            axis=1,
        )
    incident_user_df = incident_user_df.rename(
        columns={
            "black_incidents": "black_incidents",
            "black_users": "black_users",
            "white_incidents": "white_incidents",
            "white_users": "white_users",
        }
    )
    return incident_user_df


def add_extra_information(
    selection_bias_df: pd.DataFrame,
    nibrs_df: pd.DataFrame,
    census_df: pd.DataFrame,
    geographic_resolution: str,
    year: int,
) -> pd.DataFrame:

    incidents = (
        nibrs_df.groupby(resolution_dict[geographic_resolution])
        .incidents.sum()
        .reset_index()
    )

    # Remove agency duplicates
    census_df.drop(columns=["ori"], inplace=True)
    census_df.drop_duplicates(inplace=True)

    popdf = (
        census_df.groupby([resolution_dict[geographic_resolution]])
        .frequency.sum()
        .reset_index()
    )

    if geographic_resolution == "county":
        if isinstance(year, int):
            coverage = pd.read_csv(
                data_path / "misc" / "county_coverage.csv",
                usecols=["FIPS", "coverage", "year"],
                dtype={"FIPS": str, "year": int},
            )
            selection_bias_df["year"] = year
            selection_bias_df = selection_bias_df.merge(
                coverage, how="left", on=["FIPS", "year"]
            )
        urban_codes = pd.read_csv(
            data_path / "misc" / "NCHSURCodes2013.csv",
            usecols=["FIPS code", "2013 code"],
        )
        urban_codes.rename(
            columns={"FIPS code": "FIPS", "2013 code": "urban_code"}, inplace=True
        )
        urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))

        selection_bias_df = selection_bias_df.merge(urban_codes, how="left", on="FIPS")

    selection_bias_df = selection_bias_df.merge(
        incidents, how="left", on=resolution_dict[geographic_resolution]
    )
    selection_bias_df = selection_bias_df.merge(
        popdf, how="left", on=resolution_dict[geographic_resolution]
    )

    return selection_bias_df


def add_race_ratio(
    census_df: pd.DataFrame, incident_df: pd.DataFrame, geographic_resolution: str
):
    if "ori" in census_df.columns:
        census_df.drop(columns=["ori"], inplace=True)
        census_df.drop_duplicates(inplace=True)
    race_ratio = (
        census_df.groupby([resolution_dict[geographic_resolution], "race"])
        .frequency.sum()
        .reset_index()
    )
    race_ratio = race_ratio.pivot(
        resolution_dict[geographic_resolution], columns="race"
    ).reset_index()
    race_ratio.columns = [resolution_dict[geographic_resolution], "black", "white"]
    race_ratio["bwratio"] = race_ratio["black"] / race_ratio["white"]

    incident_df = pd.merge(
        incident_df, race_ratio, on=resolution_dict[geographic_resolution], how="left"
    )

    return incident_df


############ DATASET LOADING #############


def load_datasets(
    years: str,
    resolution: str,
    poverty: bool,
    metro: bool,
    all_incidents: bool,
    target: str,
    arrests: bool,
    time: str,
    time_type: str,
    hispanics: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    census_df = get_census_data(years=years, poverty=poverty, metro=metro)
    nsduh_df = get_nsduh_data(years=years, poverty=poverty, metro=metro, target=target)
    if target not in ["dui", "drunkeness", "ucr_possesion"]:
        nibrs_df = load_and_process_nibrs(
            years=years,
            resolution=resolution,
            all_incidents=all_incidents,
            arrests=arrests,
            time=time,
            time_type=time_type,
            hispanic=hispanics,
        )
    else:
        nibrs_df = get_ucr_data(years=years, target=target)
    return census_df, nsduh_df, nibrs_df


def wilson_error(n_s: int, n: int, z=1.96) -> Tuple[float, float]:
    """
    Wilson score interval

    param n_us: number of successes
    param n: total number of events
    param z: The z-value

    return: The lower and upper bound of the Wilson score interval
    """
    n_f = n - n_s
    denom = n + z ** 2
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2 / 4))
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
    ci = np.sqrt((e1 / p1) ** 2 + (e2 / p2) ** 2) * sr
    return sr, ci


def bootstrap_selection(incident_users_df: pd.DataFrame, bootstraps: int):
    def _sample_incidents(prob, trials, bootstraps: int, seed: int) -> int:
        prob = np.min([prob, 1])
        np.random.seed(seed)
        if not np.isnan(prob):
            successes = np.random.binomial(n=trials, p=prob, size=bootstraps)
        else:
            successes = np.nan
        return successes

    black_count = []
    white_count = []
    incident_users_df.black_incidents += 0.5
    incident_users_df.white_incidents += 0.5
    incident_users_df.black_users += 1
    incident_users_df.white_users += 1
    incident_users_df["black_prob"] = incident_users_df.black_incidents / (
        incident_users_df.black_users
    )
    incident_users_df["white_prob"] = incident_users_df.white_incidents / (
        incident_users_df.white_users
    )
    for i, row in incident_users_df.iterrows():
        white_vector = _sample_incidents(
            row["white_prob"], row["white_users"], bootstraps, seed=1
        )
        black_vector = _sample_incidents(
            row["black_prob"], row["black_users"], bootstraps, seed=1
        )
        selection_ratios = (black_vector / row["black_users"]) / (
            white_vector / row["white_users"]
        )
        selection_ratios = np.nan_to_num(selection_ratios, nan=np.inf, posinf=np.inf)
        incident_users_df.loc[i, "selection_ratio"] = (
            row["black_prob"] / row["white_prob"]
        )
        incident_users_df.loc[i, "lb"] = np.nan_to_num(
            np.quantile(selection_ratios, 0.025), nan=np.inf
        )
        incident_users_df.loc[i, "ub"] = np.nan_to_num(
            np.quantile(selection_ratios, 0.975), nan=np.inf
        )
        incident_users_df.loc[i, "black_incident_variance"] = np.nan_to_num(
            np.var(black_vector), nan=0
        )
        incident_users_df.loc[i, "white_incident_variance"] = np.nan_to_num(
            np.var(white_vector), nan=0
        )
    return incident_users_df.reset_index()[
        [
            "FIPS",
            "selection_ratio",
            "lb",
            "ub",
            "black_incidents",
            "black_users_variance",
            "white_users_variance",
            "black_incident_variance",
            "white_incident_variance",
            "black_users",
            "white_incidents",
            "white_users",
        ]
    ]


def delta_method(df: pd.DataFrame):
    """['FIPS', 'selection_ratio', 'lb', 'ub', 'black_incidents',
    'black_users_variance', 'white_users_variance',
    'black_incident_variance', 'white_incident_variance', 'black_users',
    'white_incidents', 'white_users']"""
    var_log_sr = (
        1 / (df.black_incidents ** 2) * df.black_incident_variance
        + 1 / (df.white_incidents ** 2) * df.white_incident_variance
        + 1 / (df.black_users ** 2) * df.black_users_variance
        + 1 / (df.white_users ** 2) * df.white_users_variance
    )
    df["ub"] = df.selection_ratio * np.exp(1.96 * np.sqrt(var_log_sr))
    df["lb"] = df.selection_ratio * np.exp(-1.96 * np.sqrt(var_log_sr))
    df["var_log"] = var_log_sr
    return df


def main(
    resolution: str,
    year: str,
    smooth: bool,
    ci: Optional[Literal["none", "wilson", "bootstrap"]],
    bootstraps: int,
    poverty: bool,
    metro: bool,
    all_incidents: bool,
    urban_filter: int,
    smoothing_param: int,
    group_years: bool,
    target: str,
    arrests: bool,
    time: str,
    time_type: str,
    hispanics: bool,
    dont_aggregate: bool,
):

    if not group_years:
        if "-" in year:
            years = year.split("-")
            years = range(int(years[0]), int(years[1]) + 1)

        else:
            years = [int(year)]
    else:
        years = [year]

    selection_bias_df = pd.DataFrame()

    for yi in years:

        ##### DATA LOADING ######

        try:
            census_df, nsduh_df, nibrs_df = load_datasets(
                str(yi),
                resolution,
                poverty,
                metro,
                all_incidents,
                target,
                arrests,
                time,
                time_type,
                hispanics,
            )
        except FileNotFoundError:
            warnings.warn(f"Data missing for {yi}. Skipping.")
            continue
        # Copy to retain unsmoothed information
        incident_df = nibrs_df.copy()
        population_df = census_df.copy()

        if smooth:
            nibrs_df, census_df = smooth_data(
                nibrs_df,
                census_df,
                metro=metro,
                poverty=poverty,
                urban_filter=urban_filter,
                smoothing_param=smoothing_param,
                group_years=group_years,
            )

        ##### END DATA LOADING ######
        ucr = target in ["dui", "drunkeness", "ucr_possesion"]
        #### START ####
        incident_users_df = incident_users(
            nibrs_df,
            census_df,
            nsduh_df,
            resolution,
            poverty,
            metro,
            years=group_years,
            ucr=ucr,
            dont_aggregate=dont_aggregate,
        )
        incident_users_df = incident_users_df.fillna(0)

        if not dont_aggregate:
            if ci == "none":
                temp_df = selection_ratio(incident_users_df, wilson=False)
            elif ci == "wilson":
                temp_df = selection_ratio(incident_users_df, wilson=True)
            else:
                temp_df = bootstrap_selection(incident_users_df, bootstraps)
                temp_df = delta_method(temp_df)

            if not ucr:
                temp_df = add_extra_information(
                    temp_df, incident_df, population_df, resolution, yi
                )
                temp_df = add_race_ratio(population_df, temp_df, resolution)
                #### END ###
        else:
            temp_df = incident_users_df.copy()

        temp_df["year"] = yi
        selection_bias_df = selection_bias_df.append(temp_df.copy())

    filename = f"selection_ratio_{resolution}_{year}"

    if arrests:
        selection_bias_df.columns = [
            col.replace("incident", "arrest") for col in selection_bias_df.columns
        ]

    if group_years:
        filename += "_grouped"

    if ci == "bootstrap":
        filename += f"_bootstraps_{bootstraps}"
    elif ci == "wilson":
        filename += "_wilson"

    if poverty:
        filename += "_poverty"

    if metro:
        filename += "_metro"

    if all_incidents:
        filename += "_all_incidents"

    if hispanics:
        filename += "_hispanics"

    if urban_filter != 2:
        filename += f"_urban_filter_{urban_filter}"

    if smooth:
        filename += "_smoothed"
        if smoothing_param != 1:
            filename += f"_{str(smoothing_param).replace('.', '-')}"

    if target != "using":
        filename += f"_{target}"

    if time != "any":
        filename += f"_{time}_{time_type}"

    if arrests:
        filename += "_arrests"

    if dont_aggregate:
        filename += "_not_aggregated"

    filename += ".csv"
    selection_bias_df.to_csv(data_path / "output" / filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year", help="year, or year range. e.g. 2015-2019, 2013, etc.", default="2019"
    )
    parser.add_argument(
        "--resolution",
        help="The geographic resolution. Options: county, region, state, agency. Default = county.",
        default="county",
    )
    parser.add_argument(
        "--ci",
        type=str,
        help="The type of confidence interval to use. Options: [None, bootstrap, wilson]. Default: None",
        default="none",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        help="The number of bootstraps to perform in order to get a confidence interval.",
        default=-1,
    )
    parser.add_argument(
        "--smooth",
        help="Flag indicating whether to perform geospatial smoothing. Default = False.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--poverty",
        help="Whether to include poverty in the usage model.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--metro",
        help="Whether to include metro/non-metro definitions in the usage model.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--all_incidents",
        help="Whether to include all incident types that involve cannabis in some way.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--urban_filter",
        type=int,
        help="The level of urban areas to filter out when smoothing; UA >= N are removed before smoothing. Default=2.",
        default=2,
    )
    parser.add_argument(
        "--smoothing_param",
        type=float,
        help="Smoothing is currently 1/(d+1)**p. Where d is (graph) distance. This parameter sets p. Default=1",
        default=1,
    )
    parser.add_argument(
        "--group_years",
        help="Whether to group years into one poor for SR and CI calculation.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="""The target to use, options are: 
        using, buying, buying_outside, traded, traded_outside, dui, drunkeness, ucr_possesion""",
        default="using",
    )
    parser.add_argument(
        "--arrests",
        help="""Whether to calculate the selection bias according to arrests,
        rather than incidents.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--time",
        help="The time of the offense. Options: any, day, night. Default: any.",
        type=str,
        default="any",
    )
    parser.add_argument(
        "--time_type",
        help="""The type of time to use. Options are: simple which uses 6am-8pm inclusive, and daylight which assigns day/night based on whether it is light. Default: simple.""",
        default="simple",
    )
    parser.add_argument(
        "--hispanics",
        help="""Whether to include hispanics in the incident counts.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dont-aggregate",
        help="""Flag indicating whether to simply return the raw counts of users and incidents. Used for visualizations etc.""",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if int(args.bootstraps) > 0 and args.ci != "bootstrap":
        raise ValueError("If bootstraps is specified, ci must be bootstrap.")

    main(
        resolution=args.resolution,
        year=args.year,
        smooth=args.smooth,
        ci=args.ci.lower(),
        bootstraps=int(args.bootstraps),
        poverty=args.poverty,
        metro=args.metro,
        all_incidents=args.all_incidents,
        urban_filter=args.urban_filter,
        smoothing_param=args.smoothing_param,
        group_years=args.group_years,
        target=args.target,
        arrests=args.arrests,
        time=args.time,
        time_type=args.time_type,
        hispanics=args.hispanics,
        dont_aggregate=args.dont_aggregate,
    )
