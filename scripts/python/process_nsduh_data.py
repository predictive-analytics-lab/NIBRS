"""This python file loads the NSDUH dataset and formats it for the project needs."""

import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "data"


# This is the NSDUH dataset. Only AGE/RACE/SEX/MJDAY30A are selected.
nsduh_df = pd.read_csv(data_path / "NSDUH" / "NSDUH_2019_Tab.txt",
                       sep="\t", usecols=["NEWRACE2", "CATAG3", "IRSEX", "MJDAY30A"])

nsduh_df.rename(columns={
    "NEWRACE2": "RACE",
    "CATAG3": "AGE",
    "IRSEX": "SEX"
}, inplace=True)

# This is the NSDUH dataset. Only AGE/RACE/SEX/MJDAY30A are selected.
nsduh_df = pd.read_csv(data_path / "NSDUH" / "NSDUH_2019_Tab.txt",
                       sep="\t", usecols=["NEWRACE2", "CATAG3", "IRSEX", "MJDAY30A"])

nsduh_df.rename(columns={
    "NEWRACE2": "RACE",
    "CATAG3": "AGE",
    "IRSEX": "SEX"
}, inplace=True)

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


nsduh_df.rename(columns={
    "NEWRACE2": "RACE",
    "CATAG3": "AGE",
    "IRSEX": "SEX"
}, inplace=True)

nsduh_df.RACE = nsduh_df.RACE.map(race_dict)
nsduh_df = nsduh_df[nsduh_df.RACE != "other/mixed"]
nsduh_df.AGE = nsduh_df.AGE.map(age_dict)
nsduh_df.SEX = nsduh_df.SEX.map(sex_dict)
nsduh_df["MJBINARY"] = nsduh_df.MJDAY30A.map(binary_usage)
nsduh_df.MJDAY30A = nsduh_df.MJDAY30A.map(usage)

nsduh_df.to_csv(data_path / "NSDUH" /
                "processed_cannabis_usage.csv")
