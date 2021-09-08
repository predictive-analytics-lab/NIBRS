"""This python file loads the NSDUH dataset and formats it for the project needs."""

from typing import List, Union
import pandas as pd
from pathlib import Path
import argparse
import csv

data_path = Path(__file__).parent.parent.parent.parent / "data"

############ Level (factor) Conversion Dicts + Functions ###########

columns_to_use = [
    "NEWRACE2",
    "CATAG3",
    "IRSEX",
    "MJDAY30A",
    "ANALWT_C",
]


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

poverty_dict = {
    1: "living in poverty",
    2: "income higher than poverty level",
    3: "income higher than poverty level"
}

urbancounty_dict = {
    1: "Large metro",
    2: "Small metro",
    3: "Nonmetro"
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


def get_nsduh_data(years: Union[str, List[Union[str, int]]]) -> pd.DataFrame:
    """
    Function which converts a string range of years to a list of year ints.
    Following this, each corresponding NSDUH year is loaded and merged with the
    NSDUH data.
    """
    if isinstance(years, str):
        if "-" not in years:
            return load_and_process_nsduh(years)
        years = years.split("-")
        years = range(int(years[0]), int(years[1]) + 1)
    
    assert min(years) >= 2012 and max(years) <= 2019, "Invalid year range. Valid range is: 2012-2019."

    # Load NSDUH data
    nsduh_df = pd.DataFrame()

    # Load each year of NSDUH data
    for year in years:
        nsduh_df = nsduh_df.append(load_and_process_nsduh(year), ignore_index=True)

    return nsduh_df

def load_and_process_nsduh(year: Union[int, str]) -> pd.DataFrame:
    if isinstance(year, str):
        year = int(year)
    
    # This is the NSDUH dataset. Only AGE/RACE/SEX/MJDAY30A are selected.
    with open(data_path / "NSDUH" / f"NSDUH_{year}_Tab.tsv", newline='') as f:
        csv_reader = csv.reader(f)
        csv_headings = next(csv_reader)[0].split("\t")
    poverty_var = [x for x in csv_headings if "poverty" in x.lower()]
    coutyp_var = [x for x in csv_headings if "coutyp" in x.lower()]

    nsduh_df = pd.read_csv(data_path / "NSDUH" / f"NSDUH_{year}_Tab.tsv",
                        sep="\t", usecols=columns_to_use + poverty_var + coutyp_var)

    nsduh_df.rename(columns={
        "NEWRACE2": "race",
        "CATAG3": "age",
        "IRSEX": "sex",
        poverty_var[0]: "poverty",
        coutyp_var[0]: "urbancounty"
    }, inplace=True)

    nsduh_df["race"] = nsduh_df.race.map(race_dict)
    nsduh_df["age"] = nsduh_df.age.map(age_dict)
    nsduh_df["sex"] = nsduh_df.sex.map(sex_dict)
    nsduh_df["poverty"] = nsduh_df.poverty.map(poverty_dict)
    nsduh_df["urbancounty"] = nsduh_df.urbancounty.map(urbancounty_dict)
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
        df = get_nsduh_data("2019")
    else:
        df = get_nsduh_data(args.year)

    year = args.year if args.year else "2019"
    df.to_csv(data_path / "NSDUH" / f"nsduh_processed_{year}.csv")



