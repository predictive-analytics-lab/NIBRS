from pathlib import Path

import pandas as pd

output_path = Path(__file__).parents[3] / "data" / "output"


def n_counties_condition(df: pd.DataFrame, cond_var):
    if (
        cond_var == "lb"
    ):  # special case for lower bound - bit annoying but not worth implementing anything better
        cond1 = df["ub"] < 0.8
    else:
        cond1 = df[cond_var] < 0.8
    cond2 = df[cond_var] > 1.25
    cond3 = df[cond_var] > 2
    cond4 = df[cond_var] > 5
    return [len(df[cond1]), len(df[cond2]), len(df[cond3]), len(df[cond4])]


def remove_nonsense(df):
    if "black_arrests" in df.columns:
        df = df[df.black_arrests > 0.5]
        df = df[df.white_arrests > 0.5]
    else:
        df = df[df.black_incidents > 0.5]
        df = df[df.white_incidents > 0.5]
    return df


if __name__ == "__main__":
    poverty = pd.read_csv(
        output_path
        / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv"
    )

    poverty = remove_nonsense(poverty)

    purchase_public = pd.read_csv(
        output_path
        / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv"
    )

    purchase_public = remove_nonsense(purchase_public)

    pov_value = n_counties_condition(poverty, "selection_ratio")
    pov_95 = n_counties_condition(poverty, "lb")

    pur_value = n_counties_condition(purchase_public, "selection_ratio")
    pur_95 = n_counties_condition(purchase_public, "lb")

    result = pd.DataFrame(
        [pov_value, pov_95, pur_value, pur_95],
        columns=["ER < 0.8", "ER > 1.25", "ER > 2", "ER > 5"],
        index=[
            "Usage Value",
            "Usage 95% Conf.",
            "Purchase Public Value",
            "Purchase Public 95% Conf.",
        ],
    )
    result.to_csv(output_path / "table_2.csv")
