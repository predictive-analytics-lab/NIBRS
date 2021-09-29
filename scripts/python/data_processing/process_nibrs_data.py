"""
This python script processes, and modifies, the NIBRS data output from the SQL query.

"""
import functools
from typing import List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import glob
import warnings
from collections import Counter
from ast import literal_eval as make_tuple
from astral.sun import sunrise, sunset
from astral import Observer
from pytz import timezone
import datetime
from tqdm import tqdm
import swifter

data_path = Path(__file__).parent.parent.parent.parent / "data"
data_name_drug_incidents = data_path / "NIBRS" / "raw" / "cannabis_allyears.csv"
data_name_all_incidents = (
    data_path / "NIBRS" / "raw" / "cannabis_allyears_allincidents.csv"
)
data_name_hispanics = data_path / "NIBRS" / "raw" / "cannabis_allyears_hispanics.csv"
fips_to_latlong = pd.read_csv(
    data_path / "misc" / "us-county-boundaries-latlong.csv", dtype={"FIPS": str}
)

cols_to_use = [
    "race",
    "age_num",
    "sex_code",
    "arrest_type_name",
    "ori",
    "data_year",
    "location",
    "incident_hour",
    "incident_date",
]

resolution_dict = {
    "state": "state",
    "state_region": "state_region",
    "county": "FIPS",
    "agency": "ori",
}


def age_cat(age: int) -> str:
    if age < 18:
        return "12-17"
    if age < 26:
        return "18-25"
    if age < 35:
        return "26-34"
    if age < 50:
        return "35-49"
    if age >= 50:
        return "50+"
    return np.nan


# def load_and_combine_years(years: List[int]) -> pd.DataFrame:
#     """
#     This function takes a wildcard string, loading in matching dataframes, and combining the years into one CSV.
#     """
#     df = None
#     for year in years:
#         try:
#             if df is None:
#                 df = pd.read_csv(data_path / "NIBRS" / "raw" / script_name(year), usecols=cols_to_use)
#             else:
#                 df = df.append(pd.read_csv(data_path / "NIBRS" / "raw" / script_name(year), usecols=cols_to_use))
#         except FileNotFoundError:
#             years.remove(year)
#             print(f"No NIBRS data for {year}")
#     return df, years


def transform_location(location: tuple) -> str:
    try:
        location = make_tuple(location)
    except:
        return "None"
    location_counts = Counter(li for li in location)
    max_key = max(location_counts, key=location_counts.get)
    if max_key == "nan":
        max_key = "None"
    return max_key


@functools.lru_cache(maxsize=None)
def sunrise_sunset_time(
    latitude: float,
    longitude: float,
    timezone: str,
    incident_date: datetime,
    incident_hour: int,
) -> Tuple[int, int]:
    county = Observer(latitude=latitude, longitude=longitude)
    sunrise_time = sunrise(county, date=incident_date, tzinfo=timezone)
    sunset_time = sunset(county, date=incident_date, tzinfo=timezone)
    if incident_hour > sunrise_time.hour and incident_hour <= sunset_time.hour:
        return "day"
    elif incident_hour <= sunrise_time.hour or incident_hour > sunset_time.hour:
        return "night"
    else:
        return "invalid"


def load_and_process_nibrs(
    years: str,
    resolution: str,
    hispanic: bool = False,
    arrests: bool = False,
    all_incidents: bool = False,
    location: bool = False,
    time: str = "any",
    time_type: str = "daylight",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the current nibrs dataset, and processes it.
    Additionally, it adds the FIPS code and state subregion code to the data.

    :param years: The years to load.
    """

    if "-" in years:
        years = years.split("-")
        years = range(int(years[0]), int(years[1]) + 1)
        try:
            years = [int(year) for year in years]
        except:
            print("invalid year format.")
    else:
        try:
            years = [int(years)]
        except:
            print("invalid year format. Run appropriate SQL script.")

    if all_incidents:
        nibrs_df = pd.read_csv(data_name_all_incidents, usecols=cols_to_use)
    elif hispanic:
        nibrs_df = pd.read_csv(data_name_hispanics, usecols=cols_to_use)
    else:
        nibrs_df = pd.read_csv(data_name_drug_incidents, usecols=cols_to_use)

    nibrs_df = nibrs_df[nibrs_df.data_year.isin(years)]

    nibrs_df["age_num"] = nibrs_df.age_num.apply(age_cat)
    nibrs_df["sex_code"] = nibrs_df.sex_code.map({"F": "female", "M": "male"})

    nibrs_df.rename(
        columns={
            "sex_code": "sex",
            "age_num": "age",
            "arrest_type_name": "arrest_type",
        },
        inplace=True,
    )

    fips_ori_df = pd.read_csv(
        data_path / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "FIPS"],
        dtype={"FIPS": object},
    )

    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    subregion_df = pd.read_csv(
        data_path / "misc" / "subregion_counties.csv",
        dtype={"FIPS": object},
        usecols=["State", "Region", "FIPS"],
    )

    nibrs_df = pd.merge(nibrs_df, fips_ori_df, on="ori")

    nibrs_df = nibrs_df.merge(fips_to_latlong, how="left", on="FIPS")

    nibrs_df["incident_date"] = pd.to_datetime(nibrs_df["incident_date"])

    if time != "any":
        if time_type.lower() == "daylight":
            nibrs_df["night_day"] = nibrs_df.apply(
                lambda x: sunrise_sunset_time(
                    x.lat, x.lon, x.timezone, x.incident_date, x.incident_hour
                ),
                axis=1,
            )
            assert time.lower() in ["day", "night"], "Invalid Time."
            nibrs_df = nibrs_df[nibrs_df.night_day == time.lower()]
        elif time_type.lower() == "simple":
            if time.lower() == "day":
                nibrs_df = nibrs_df[
                    (nibrs_df.incident_hour >= 6) & (nibrs_df.incident_hour <= 20)
                ]
            elif time.lower() == "night":
                nibrs_df = nibrs_df[
                    (nibrs_df.incident_hour < 6) | (nibrs_df.incident_hour > 20)
                ]
        else:
            raise ValueError("Invalid Time Type.")

    nibrs_df = pd.merge(nibrs_df, subregion_df, on="FIPS")
    nibrs_df["state_region"] = nibrs_df["State"] + "-" + nibrs_df["Region"]

    # Clean up
    nibrs_df.rename(columns={"State": "state"}, inplace=True)
    nibrs_df.drop(["Region"], axis=1, inplace=True)

    groupers = ["age", "race", "sex"]
    if location:
        nibrs_df["location"] = nibrs_df["location"].apply(transform_location)
        groupers += ["location"]

    nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]

    locations = nibrs_df[["state", "state_region", "FIPS"]].drop_duplicates()

    nibrs_df = (
        nibrs_df.groupby(
            sorted(groupers + [resolution_dict[resolution]], key=str.casefold)
        )
        .size()
        .to_frame("incidents")
        .reset_index()
    )
    nibrs_arrests = (
        nibrs_arrests.groupby(
            sorted(groupers + [resolution_dict[resolution]], key=str.casefold)
        )
        .size()
        .to_frame("incidents")
        .reset_index()
    )

    nibrs_df = nibrs_df.merge(locations, on=resolution_dict[resolution], how="inner")
    nibrs_arrests = nibrs_arrests.merge(
        locations, on=resolution_dict[resolution], how="inner"
    )

    nibrs_df["year"] = "-".join([str(y) for y in years])
    nibrs_arrests["year"] = "-".join([str(y) for y in years])
    if arrests:
        return nibrs_arrests

    return nibrs_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument(
        "--resolution",
        help="Geographic resolution to aggregate incidents over.",
        default="state",
    )
    parser.add_argument(
        "--all_incidents",
        help="Geographic resolution to aggregate incidents over.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--hispanic",
        help="Whether to include hispanic individuals",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--location",
        help="Whether to additionally group by location of offense.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--time",
        help="The time of the offense. Options: any, day, night. Default: any.",
        type=str,
        default="any",
    )
    parser.add_argument(
        "--arrests",
        help="""Whether to calculate the selection bias according to arrests,
        rather than incidents.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--time_type",
        help="""The type of time to use. Options are: simple which uses 6am-8pm inclusive, and daylight which assigns day/night based on whether it is light. Default: simple.""",
        default="simple",
    )

    args = parser.parse_args()

    df = load_and_process_nibrs(
        args.year,
        args.resolution,
        args.hispanic,
        all_incidents=args.all_incidents,
        arrests=args.arrests,
        location=args.location,
        time=args.time,
    )
    year = args.year if args.year else "2019"
    if args.arrests:
        df.to_csv(data_path / "NIBRS" / f"arrests_{year}.csv")
    else:
        df.to_csv(data_path / "NIBRS" / f"incidents_{year}.csv")
