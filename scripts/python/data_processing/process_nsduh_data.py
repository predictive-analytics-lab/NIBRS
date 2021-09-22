"""This python file runs the R script to produce the usage model."""
from typing import List, Union
import pandas as pd

import argparse
import subprocess

from pathlib import Path

data_path = Path(__file__).parents[3] / "data" / "NSDUH"

# Rscript scripts/R/usage_model_on_nsduh.R 1 1 1

def get_file(poverty: bool, urban: bool, hispanic: bool = False) -> pd.DataFrame:
    filename = "nsduh_usage_2007_2019_nohisp"
    if poverty:
        filename += "_poverty"
    if urban:
        filename += "_urban"
    filename+= ".csv"
    return pd.read_csv(data_path / filename, index_col=False)


def get_nsduh_data(years: Union[str, List[Union[str, int]]], poverty: bool = False, urban: bool = False, hispanic: bool = False):
    df = get_file(poverty, urban, hispanic)
    if "-" in years:
        years = years.split("-")
        years = [int(y) for y in range(int(years[0]), int(years[-1]) + 1)]
    else:
        years = [int(years)]   
    df = df[df.year.isin(years)]
    if poverty:
        df = df.rename({"poverty_level": "poverty"}, axis=1)
    if urban:
        df = df.rename({"is_urban": "urbancounty"}, axis=1)
    return df

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.")

    args=parser.parse_args()
    
    df = get_nsduh_data(years=args.year)
    
    df.to_csv(data_path / f"nsduh_processed_{args.year}.csv", index=False)
    

    



