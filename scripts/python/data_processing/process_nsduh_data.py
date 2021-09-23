"""This python file runs the R script to produce the usage model."""
from typing import List, Union
from typing_extensions import Literal
import pandas as pd

import argparse
import subprocess

from pathlib import Path

data_path = Path(__file__).parents[3] / "data" / "NSDUH"

# Rscript scripts/R/usage_model_on_nsduh.R 1 1 1


def get_file(poverty: bool, urban: bool, hispanic: bool = False) -> pd.DataFrame:
    filename = "nsduh_usage_2012_2019"
    if hispanic:
        filename += "_hispincluded"
    else:
        filename += "_nohisp"
    if poverty:
        filename += "_poverty"
    if urban:
        filename += "_urban"
    filename += ".csv"
    return pd.read_csv(data_path / filename, index_col=False)


target_variables = {
    "using": ["mean_usage_day", "mean_usage_day_se"],
    "buying": ["mean_bought_day", "mean_bought_day_se"],
    "buying_outside": ["mean_bought_outside_day", "mean_bought_outside_day_se"],
    "traded": ["mean_traded_day", "mean_traded_day_se"],
    "traded_outside": ["mean_traded_outside_day", "mean_traded_outside_day_se"],
}

var_names = ["MJ", "MJ_SE"]


def get_nsduh_data(
    years: Union[str, List[Union[str, int]]],
    poverty: bool = False,
    urban: bool = False,
    hispanic: bool = False,
    target: Literal[
        "using", "buying", "buying_outside", "traded", "traded_outside"
    ] = "using",
):
    vars_to_keep = ["year", "age", "race", "sex", "MJ", "MJ_SE"]
    df = get_file(poverty, urban, hispanic)
    if "-" in years:
        years = years.split("-")
        years = [int(y) for y in range(int(years[0]), int(years[-1]) + 1)]
    else:
        years = [int(years)]

    if poverty:
        df = df.rename({"poverty_level": "poverty"}, axis=1)
        vars_to_keep.insert(3, "poverty")
    if urban:
        df = df.rename({"is_urban": "urbancounty"}, axis=1)
        vars_to_keep.insert(3, "urbancounty")

    if target == "using":
        df = df[df.year.isin(years)]
    else:
        yintersect = set(years).intersection({2015, 2016, 2017})
        if len(yintersect) > 0:
            tdf = pd.DataFrame()
            for year in yintersect:
                mdf = df[df.year.isin([2014, 2018])]
                mj, mj_se = target_variables[target]
                temp_tdf = (
                    mdf.groupby(vars_to_keep[1:-2])
                    .agg({mj: "mean", mj_se: "mean"})
                    .reset_index()
                )
                temp_tdf["year"] = year
                tdf = tdf.append(temp_tdf)
            years = set(years) - {2015, 2016, 2017}
            df = tdf.append(df[df.year.isin(years)])
        else:
            df = df[df.year.isin(years)]

    df = df.rename(columns={m: v for m, v in zip(target_variables[target], var_names)})
    return df[vars_to_keep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default="2019")
    parser.add_argument(
        "--target",
        type=str,
        help="""The target to use, options are: 
        using, buying, buying_outside, traded, traded_outside""",
        default="using",
    )
    parser.add_argument(
        "--hispanic",
        help="Whether to include hispanics in the model. Default = false.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--poverty",
        help="Whether to include poverty information in the model. Default = true.",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--urban",
        help="Whether to include urban information in the model. Default = false.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    df = get_nsduh_data(
        years=args.year,
        poverty=args.poverty,
        urban=args.urban,
        hispanic=args.hispanic,
        target=args.target,
    )

    df.to_csv(data_path / f"nsduh_processed_{args.year}.csv", index=False)
