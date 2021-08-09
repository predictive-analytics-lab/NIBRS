"""
This python script processes, and modifies, the NIBRS data output from the SQL query.

"""
import functools
from typing import Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

data_path = Path(__file__).parent.parent.parent.parent / "data"
current_script_name = "cannabis_agency_2019_20210608.csv"

cols_to_use = [
    "dm_offender_race_ethnicity",
    "dm_offender_age",
    "dm_offender_sex",
    "arrest_type",
    "cannabis_mass",
    "ori"
]

def disjunction(*conditions):
    """
    Function which takes a list of conditions and returns the logical OR of them.
    """
    return functools.reduce(np.logical_or, conditions)

def load_and_process_nibrs(years: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the current nibrs dataset, and processes it.
    Additionally, it adds the FIPS code and state subregion code to the data.

    :param years: The years to load.
    """

    if "-" in years:
        years = years.split("-")
        years = range(int(years[0]), int(years[1]) + 1)
        try:
            years = [12 - (2019 - int(yi)) for yi in years]
        except:
            print("invalid year format.")
    else:
        try:
            years = [12 - (2019 - int(years))]
        except:
            print("invalid year format.")


    nibrs_df = pd.read_csv(data_path / "NIBRS" / current_script_name, usecols=cols_to_use)

    nibrs_df = nibrs_df[disjunction(*[nibrs_df.data_year == yi for yi in years])]

    nibrs_df.rename(columns={
        "dm_offender_race_ethnicity": "race",
        "dm_offender_age": "age",
        "dm_offender_sex": "sex"
    }, inplace=True)

    nibrs_df = nibrs_df[nibrs_df["race"] != "hispanic/latino"]

    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t",
                            usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                            dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

    nibrs_df = pd.merge(nibrs_df, fips_ori_df, on="ori")
    nibrs_df = pd.merge(nibrs_df, subregion_df, on="FIPS")
    nibrs_df["state_region"] = nibrs_df["State"] + "-" + nibrs_df["Region"]

    nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]

    # Clean up
    nibrs_df.rename(columns={"State":"state"}, inplace=True)
    nibrs_df.drop(["Region"], axis=1, inplace=True)

    return nibrs_df, nibrs_arrests

if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.")

    args=parser.parse_args()

    if not args.year:
        print("No year specified. Defaulting to 2019")
        df, df_a = load_and_process_nibrs("2019")
    else:
        df, df_a = load_and_process_nibrs(args.year)
    year = args.year if args.year else "2019"
    df.to_csv(data_path / "NIBRS" / f"incidents_processed_{year}.csv")
    df_a.to_csv(data_path / "NIBRS" / f"arrests_processed_{year}.csv")


