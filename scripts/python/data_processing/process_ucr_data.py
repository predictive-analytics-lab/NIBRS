"""
This python script processes, and modifies, the NIBRS data output from the SQL query.

"""
import functools
from typing import List, Union, Tuple
from typing_extensions import Literal
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

data_path = Path(__file__).parent.parent.parent.parent / "data"
ucr_arrests = data_path / "UCR" / "ucr_arrests.csv"


def parse_year(years: Union[str, int]) -> List[int]:
    if isinstance(years, str):
        if "-" in years:
            years = years.split("-")
            years = [int(year)
                     for year in range(int(years[0]), int(years[1]) + 1)]
        else:
            years = [int(years)]
    elif isinstance(years, list):
        years = [int(x) for x in years]
    return years


def parse_column(colname: str) -> str:
    if "black" in colname:
        return "black"
    elif "white" in colname:
        return "white"


def get_ucr_data(
    years: Union[str, int, list], target: Literal["dui", "drunkeness", "ucr_possesion"]
) -> pd.DataFrame:
    """
    Returns a dataframe of the UCR arrests data.

    Args:
        years: The years to include in the dataframe.
        target: The target to include in the dataframe.

    Returns:
        pd.DataFrame: The UCR arrests data.
    """
    if target == "ucr_possesion":
        target = "possesion"
    df = pd.read_csv(ucr_arrests, index_col=False)
    df = df[df["year"].isin(parse_year(years))]

    leaic_df = pd.read_csv(
        data_path / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "FIPS"],
        dtype={"FIPS": object},
    )

    leaic_df = leaic_df.rename(columns={"ORI9": "ori"})

    df = df.merge(leaic_df, on="ori")

    df = df.groupby("FIPS").sum().reset_index()

    target_columns = [x for x in df.columns if target in x]
    df = df[["FIPS", "year", *target_columns]]

    rename_dict = {k: parse_column(k) for k in target_columns}

    df = df.rename(columns=rename_dict)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="year, or year range.", default="2019")
