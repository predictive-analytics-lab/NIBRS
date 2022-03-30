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


def main(base_path, comp_path, base_name, comp_name):
    base = pd.read_csv(
        output_path
        / base_path
    )

    base = remove_nonsense(base)

    comparitor = pd.read_csv(
        output_path
        / comp_path
    )

    comparitor = remove_nonsense(comparitor)

    base_value = n_counties_condition(base, "selection_ratio")
    base_95 = n_counties_condition(base, "lb")

    comp_value = n_counties_condition(comparitor, "selection_ratio")
    comp_95 = n_counties_condition(comparitor, "lb")

    result = pd.DataFrame(
        [base_value, base_95, comp_value, comp_95],
        columns=["ER < 0.8", "ER > 1.25", "ER > 2", "ER > 5"],
        index=[
            f"{base_name} Value",
            f"{base_name} 95% Conf.",
            f"{comp_name} Value",
            f"{comp_name} 95% Conf.",
        ],
    )
    result.to_csv(output_path / "table_2_meth.csv")

if __name__ == "__main__":
    main(
        "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv", 
        "selection_ratio_county_2017-2019_grouped_bootstraps_1000_meth.csv",
        "Cannabis",
        "Methamphetamines",
    )