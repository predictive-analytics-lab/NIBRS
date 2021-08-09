"""This python file loads the NSDUH dataset and formats it for the project needs."""

import pandas as pd
from pathlib import Path
import argparse

data_path = Path(__file__).parent.parent.parent.parent / "data"

############ Level (factor) Conversion Dicts + Functions ###########

race_dict = {
    1: "white",
    2: "black",
    3: "other/mixed",
    4: "other/mixed",
    5: "other/mixed",
    6: "other/mixed",
    7: "other/mixed"
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

######### Data Loading ##########


def join_years(years: str) -> pd.DataFrame:
    """
    Function which converts a string range of years to a list of year ints.
    Following this, each corresponding NSDUH year is loaded and merged with the
    NSDUH data.
    """
    if "-" not in years:
        return load_and_process_nsduh(int(years))
    years = years.split("-")
    years = range(int(years[0]), int(years[1]) + 1)
    
    assert min(years) >= 2015 and max(years) <= 2019, "Invalid year range. Valid range is: 2015-2019."

    # Load NSDUH data
    nsduh_df = load_and_process_nsduh(years[0])

    # Load each year of NSDUH data
    for year in years[1:]:
        nsduh_df = nsduh_df.append(load_and_process_nsduh(year), ignore_index=True)

    return nsduh_df

def load_and_process_nsduh(year: int) -> pd.DataFrame:
    # This is the NSDUH dataset. Only AGE/RACE/SEX/MJDAY30A are selected.
    nsduh_df = pd.read_csv(data_path / "NSDUH" / f"NSDUH_{year}_Tab.tsv",
                        sep="\t", usecols=["NEWRACE2", "CATAG3", "IRSEX", "MJDAY30A"])

    nsduh_df.rename(columns={
        "NEWRACE2": "race",
        "CATAG3": "age",
        "IRSEX": "sex"
    }, inplace=True)

    nsduh_df["race"] = nsduh_df.race.map(race_dict)
    nsduh_df["age"] = nsduh_df.age.map(age_dict)
    nsduh_df["sex"] = nsduh_df.sex.map(sex_dict)
    nsduh_df["MJBINARY"] = nsduh_df.MJDAY30A.map(binary_usage)
    nsduh_df["MJDAY30A"] = nsduh_df.MJDAY30A.map(usage)
    nsduh_df = nsduh_df[nsduh_df.race != "other/mixed"]
    nsduh_df["year"] = year

    return nsduh_df


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.")

    args=parser.parse_args()

    if not args.year:
        print("No year specified. Defaulting to 2019")
        df = join_years("2019")
    else:
        df = join_years(args.year)

    year = args.year if args.year else "2019"
    df.to_csv(data_path / "NSDUH" / f"nsduh_processed_{year}.csv")



