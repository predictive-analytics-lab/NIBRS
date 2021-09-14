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

data_path = Path(__file__).parent.parent.parent.parent / "data"
data_name_drug_incidents = data_path / "NIBRS"/ "raw" / "cannabis_allyears.csv"
data_name_all_incidents = data_path / "NIBRS"/ "raw" / "cannabis_allyears_allincidents.csv"

cols_to_use = [
    "race",
    "age_num",
    "sex_code",
    "arrest_type_name",
    "ori",
    "data_year",
    "location"
]

resolution_dict = {"state": "state",
                   "state_region": "state_region", "county": "FIPS", "agency":  "ori"}


def age_cat(age: int) -> str:
    if age < 18: return '12-17'
    if age < 26: return '18-25'
    if age < 35: return '26-34'
    if age < 50: return '35-49'
    if age >= 50: return '50+'
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

def disjunction(*conditions):
    """
    Function which takes a list of conditions and returns the logical OR of them.
    """
    return functools.reduce(np.logical_or, conditions)

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
    
def load_and_process_nibrs(years: str, resolution: str, hispanic: bool = False, arrests: bool = False, all_incidents: bool = False, location: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    else:
        nibrs_df = pd.read_csv(data_name_drug_incidents, usecols=cols_to_use)

    nibrs_df = nibrs_df[disjunction(*[nibrs_df.data_year == yi for yi in years])] 
    
    nibrs_df["age_num"] = nibrs_df.age_num.apply(age_cat)
    nibrs_df["sex_code"] = nibrs_df.sex_code.map({"F": "female", "M": "male"})

    nibrs_df.rename(columns={"sex_code": "sex", "age_num": "age", "arrest_type_name": "arrest_type"}, inplace=True)

    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t",
                            usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                            dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

    nibrs_df = pd.merge(nibrs_df, fips_ori_df, on="ori")
    nibrs_df = pd.merge(nibrs_df, subregion_df, on="FIPS")
    nibrs_df["state_region"] = nibrs_df["State"] + "-" + nibrs_df["Region"]

    # Clean up
    nibrs_df.rename(columns={"State":"state"}, inplace=True)
    nibrs_df.drop(["Region"], axis=1, inplace=True)
    
    
    groupers = ["age", "race", "sex"]
    if location:
        nibrs_df["location"] = nibrs_df["location"].apply(transform_location)
        groupers += ["location"]
        
    nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]
    
    locations = nibrs_df[["state", "state_region", "FIPS"]].drop_duplicates()
        
    nibrs_df = nibrs_df.groupby(sorted(groupers + [resolution_dict[resolution]], key=str.casefold)).size().to_frame("incidents").reset_index()
    nibrs_arrests = nibrs_arrests.groupby(sorted(groupers + [resolution_dict[resolution]], key=str.casefold)).size().to_frame("incidents").reset_index()
    
    nibrs_df = nibrs_df.merge(locations, on=resolution_dict[resolution], how="inner")
    nibrs_arrests = nibrs_arrests.merge(locations, on=resolution_dict[resolution], how="inner")
    
    if arrests:
        return nibrs_df, nibrs_arrests
    
    return nibrs_df

if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument("--resolution", help="Geographic resolution to aggregate incidents over.", default="state")
    parser.add_argument("--all_incidents", help="Geographic resolution to aggregate incidents over.", default=False, action='store_true')
    parser.add_argument("--hispanic", help="Whether to include hispanic individuals", default=False, action='store_true')
    parser.add_argument("--location", help="Whether to additionally group by location of offense.", default=False, action='store_true')

    args=parser.parse_args()

    df, df_a = load_and_process_nibrs(args.year, args.resolution, args.hispanic, all_incidents=args.all_incidents, arrests=True, location=args.location)
    year = args.year if args.year else "2019"
    if df is not None:
        df.to_csv(data_path / "NIBRS" / f"incidents_processed_{year}.csv")
        df_a.to_csv(data_path / "NIBRS" / f"arrests_processed_{year}.csv")


