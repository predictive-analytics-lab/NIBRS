"""This python file runs the R script to produce the usage model."""
from typing import List, Union
import pandas as pd

import argparse
import subprocess

from pathlib import Path

data_path = Path(__file__).parent.parent.parent.parent / "data"

# Rscript scripts/R/usage_model_on_nsduh.R 1 1 1

def call_r(script_location: Path, args: list) -> pd.DataFrame:
    p = subprocess.Popen(["Rscript", str(script_location), *args], stdout=subprocess.PIPE)
    out, err = p.communicate()
    df = pd.read_csv(out, index_col=False)
    Path(out).unlink()
    return df

def get_nsduh_data(years: Union[str, List[Union[str, int]]], poverty: bool = True, urban: bool = True, hispanic: bool = True):
    r_script = Path(__file__).parents[2] / "R" / "process_usage_data_nsduh.R"

    if isinstance(years, str):
        if "-" not in years:
            return call_r(r_script, [int(poverty), int(urban), int(hispanic), years, years])
        years = years.split("-")
        years = [int(y) for y in years]
    year_min = min(years)
    year_max = max(years)
    assert year_min >= 2012 and year_max <= 2019, "Invalid year range. Valid range is: 2012-2019."
    return call_r(r_script, [int(poverty), int(urban), int(hispanic), year_min, year_max])
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.")

    args=parser.parse_args()
    
    df = get_nsduh_data(years=args.year)
    df.to_csv(data_path / "NSDUH" / f"nsduh_processed_{args.year}.csv")
    

    



