"""This python file loads the NSDUH dataset and formats it for the project needs."""

from typing import Literal
import pandas as pd
from pathlib import Path
import argparse

data_path = Path(__file__).parent.parent.parent.parent / "data"

############ Level (factor) Conversion Dicts + Functions ###########

columns_to_use = [
    "NEWRACE2",
    "CATAG3",
    "IRSEX",
    "ANALWT_C",
    "POVERTY3",
    "COUTYP4"
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
    2: "income up to 2x poverty threshold",
    3: "income more than 2x poverty threshold"
}

urbancounty_dict = {
    1: "Large metro",
    2: "Small metro",
    3: "Nonmetro"
}


usage_column_dict = {
    "cannabis": "MJDAY30A",
    "cocaine": "COCUS30A",
    "crack": "CRKUS30A",
    "meth": "METHAM30N"
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


def join_years(years: str, drug: Literal["cannabis", "cocaine", "crack", "meth"]) -> pd.DataFrame:
    """
    Function which converts a string range of years to a list of year ints.
    Following this, each corresponding NSDUH year is loaded and merged with the
    NSDUH data.
    """
    if "-" not in years:
        return load_and_process_nsduh(int(years), drug)
    years = years.split("-")
    years = range(int(years[0]), int(years[1]) + 1)
    
    assert min(years) >= 2012 and max(years) <= 2019, "Invalid year range. Valid range is: 2012-2019."
    assert drug in ["cannabis", "cocaine", "crack", "meth"], "Invalid drug type, must be one of [cannabis, cocaine, crack, meth]"

    # Load NSDUH data
    nsduh_df = load_and_process_nsduh(years[0], drug)

    # Load each year of NSDUH data
    for year in years[1:]:
        nsduh_df = nsduh_df.append(load_and_process_nsduh(year, drug), ignore_index=True)

    return nsduh_df

def load_and_process_nsduh(year: int, drug: str) -> pd.DataFrame:
    # This is the NSDUH dataset. Only AGE/RACE/SEX/MJDAY30A are selected.
    nsduh_df = pd.read_csv(data_path / "NSDUH" / f"NSDUH_{year}_Tab.tsv",
                        sep="\t", usecols=columns_to_use+[usage_column_dict[drug]])

    nsduh_df.rename(columns={
        "NEWRACE2": "race",
        "CATAG3": "age",
        "IRSEX": "sex",
        "POVERTY3": "poverty",
        "COUTYP4": "urbancounty",
        usage_column_dict[drug]: "usage_30day"
    }, inplace=True)

    nsduh_df["race"] = nsduh_df.race.map(race_dict)
    nsduh_df["age"] = nsduh_df.age.map(age_dict)
    nsduh_df["sex"] = nsduh_df.sex.map(sex_dict)
    nsduh_df["poverty"] = nsduh_df.poverty.map(poverty_dict)
    nsduh_df["urbancounty"] = nsduh_df.urbancounty.map(urbancounty_dict)
    nsduh_df["usage_binary"] = nsduh_df["usage_30day"].map(binary_usage)
    nsduh_df["usage_30day"] = nsduh_df["usage_30day"].map(usage)
    nsduh_df = nsduh_df[nsduh_df.race != "other/mixed"]
    nsduh_df["year"] = year

    return nsduh_df


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument("--drug", help="drug type", default="cannabis")

    args=parser.parse_args()

    df = join_years(args.year, args.drug)

    df.to_csv(data_path / "NSDUH" / f"nsduh_processed_{args.drug}_{args.year}.csv")



