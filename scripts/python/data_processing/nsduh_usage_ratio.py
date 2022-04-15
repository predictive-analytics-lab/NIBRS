"""Get the usage ratio for NSDUH race categories."""
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

data_path = Path(__file__).parents[3] / \
    "scripts" / "R" / "downloaded_data"
output_path = Path(__file__).parents[3] / "data" / "NSDUH"

drugs_post2015 = {"MJDAY30A": "cannabis", "CRKUS30A": "crack",
                  "COCUS30A": "cocaine", "HER30USE": "heroin", "METHAM30N": "meth"}
drugs_pre2015 = {"MJDAY30A": "cannabis", "CRKUS30A": "crack",
                 "COCUS30A": "cocaine", "HER30USE": "heroin", "MTDAYPMO": "meth"}

dui = ["DRVINALCO2", "DRVINMARJ2", "DRVINDRG", "DRVINDROTMJ",
       "DRVINALDRG", "DRVALDR", "DRVAONLY", "DRVDONLY"]
dui_negative = [2, 0, 81, 91, 99]

drunk = ["ALCPDANG", "ALCSERPB"]
drunk_negative = [2, 91, 83, 93]

drugs = [*drugs_post2015.values(), "dui", "drunkeness"]


def _annual_usage_to_val(row: pd.Series, col_list: List[str], negative_list: List[int]):
    if any(x == 1 for x in row[col_list].values):
        return 1
    if any(x in negative_list for x in row[col_list].values):
        return 0
    return np.nan


def _monthly_usage_to_val(x):
    if x <= 30:
        return int(x)
    if x == 91 or x == 93:
        return 0
    else:
        return np.nan


def load_nsduh_data(year: str):
    extension = ".tsv" if int(year) < 2019 else ".txt"
    if int(year) < 2015:
        drugs_dict = drugs_pre2015
    else:
        drugs_dict = drugs_post2015
    df = pd.read_csv(
        data_path / f"nsduh_{year}" / f"NSDUH_{year}_Tab{extension}", delimiter="\t", engine="python")
    # MAP race and drop others
    df["race"] = df.NEWRACE2.map({1: "white", 2: "black"})
    df = df[~df.race.isnull()]
    df = df.drop(columns=["NEWRACE2"])
    for k, v in drugs_dict.items():
        df[v] = df[k].apply(_monthly_usage_to_val)
        df = df.drop(columns=[k])
    dui_i = list(set(dui).intersection(set(df.columns)))
    drunk_i = list(set(drunk).intersection(set(df.columns)))
    df["dui"] = df.apply(lambda row: _annual_usage_to_val(
        row, dui_i, dui_negative), axis=1)
    df["drunkeness"] = df.apply(lambda row: _annual_usage_to_val(
        row, drunk_i, drunk_negative), axis=1)
    df["year"] = year
    return df[["year", "race", *drugs_post2015.values(), "drunkeness", "dui"]]


def get_nsduh_usage(df: pd.DataFrame, days_of_use: bool = True, years: bool = True):
    groupers = ["race"]

    def _users(group):
        new_group = {}
        for drug in drugs:
            new_group[drug] = [group[group[drug] > 0]
                               [drug].count() / group[drug].count()]
        return pd.DataFrame(new_group)
    if years:
        groupers += ["year"]
    if days_of_use:
        df = df.groupby(groupers).apply(
            lambda x: x.sum() / x.count()).reset_index()
    else:
        df = df.groupby(groupers).apply(
            _users).reset_index()
    return df


def get_usage_ratio(df: pd.DataFrame, years: bool = True) -> pd.DataFrame:
    indicies = ["drug"]
    melts = ["race"]
    output = ["drug", "usage_ratio"]
    if years:
        indicies += ["year"]
        melts += ["year"]
        output += ["year"]
    df = df.melt(id_vars=melts, value_vars=list(drugs),
                 value_name="usage", var_name="drug")
    df = df.pivot(index=indicies, columns="race", values="usage").reset_index()
    df["usage_ratio"] = df.black / df.white
    return df


def main(days_of_use: bool):
    df = pd.DataFrame()
    for x in tqdm(list(range(2010, 2021))):
        df_year = load_nsduh_data(str(x))
        df_year = get_nsduh_usage(df_year, days_of_use=days_of_use, years=True)
        df_year = get_usage_ratio(df_year, years=True)
        df = pd.concat([df, df_year])
    return df


if __name__ == "__main__":
    df_dou = main(days_of_use=True)
    df_dou.to_csv(output_path / "nsduh_usage_ratio_dou.csv", index=False)
    df = main(days_of_use=False)
    df.to_csv(output_path / "nsduh_usage_ratio.csv", index=False)
