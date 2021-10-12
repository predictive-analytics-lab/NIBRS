from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[3] / "data" / "correlates"

legal_states = {
    "08": 2012,
    "25": 2016,
    "26": 2018,
    "41": 2014,
    "50": 2013,
    "53": 2012,
}

model_names = [
    "Dem only",
    "Dem + Pov",
    "Dem + metro",
    "Buying",
    "Buying Outside",
    "Arrests",
]


def reporting_throughout(df: pd.DataFrame):
    reported = (
        (df.groupby("FIPS").size() == df.year.nunique())
        .to_frame("reported")
        .reset_index()
    )
    df = df.merge(reported, on="FIPS")
    return df[df.reported]


def filter_legal(df: pd.DataFrame, legal: bool = False):
    def _remove_legal(key):
        year = legal_states[key]
        condition = (df.year >= year) & (df.FIPS.apply(lambda x: x.startswith(key)))
        if not legal:
            return df[~condition]
        else:
            return df[condition]

    if not legal:
        for key in legal_states.keys():
            df = _remove_legal(key)
        return df
    else:
        df_temp = pd.DataFrame()
        for key in legal_states.keys():
            df_temp = df_temp.append(_remove_legal(key))
        return df_temp


def get_year_data(name: str, log_ratio: bool, legal: bool, reported: bool):
    filename = "selection_ratio_county_2010-2019_bootstraps_1000"
    if name == "Dem only":
        filename += ".csv"
    if name == "Dem + Pov":
        filename += "_poverty.csv"
    if name == "Dem + metro":
        filename += "_metro.csv"
    if name == "Buying":
        filename += "_poverty_buying.csv"
    if name == "Buying Outside":
        filename += "_poverty_buying_outside.csv"
    if name == "Arrests":
        filename += "_poverty_arrests.csv"
    df = pd.read_csv(
        data_path.parent.parent / "turing_output" / filename, dtype={"FIPS": str}
    )
    if legal:
        df = filter_legal(df, legal=True)
    else:
        df = filter_legal(df, legal=False)
    if reported:
        df = reporting_throughout(df)
    if log_ratio:
        df["selection_ratio"] = np.log(df["selection_ratio"])
    y, X = dmatrices(f"selection_ratio ~ year", data=df, return_type="dataframe")
    model = sm.WLS(
        y,
        X,
        weights=1 / df["var_log"],
    )
    model_res = model.fit()
    model_res = model_res.get_robustcov_results(cov_type="HC1")
    coef = model_res.params[1]
    pvalue = model_res.pvalues[1]
    std_error = model_res.HC1_se[1]
    result = f"{coef:.3f} ({std_error:.3f})"
    if pvalue <= 0.05:
        result += "*"
    if pvalue <= 0.01:
        result += "*"
    if pvalue <= 0.001:
        result += "*"
    return result


time_names = [f"Time {i}" for i in range(1, 5)] + [
    f"Time {i} Log SR" for i in range(1, 5)
]

time_results = [
    [get_year_data(name, False, False, True) for name in model_names],
    [get_year_data(name, False, False, False) for name in model_names],
    [get_year_data(name, False, True, True) for name in model_names],
    [get_year_data(name, False, True, False) for name in model_names],
    [get_year_data(name, True, False, True) for name in model_names],
    [get_year_data(name, True, False, False) for name in model_names],
    [get_year_data(name, True, True, True) for name in model_names],
    [get_year_data(name, True, True, False) for name in model_names],
]

time_df = pd.DataFrame(time_results, columns=model_names, index=time_names)
time_df.to_csv(data_path / "time_coefficients.csv")
