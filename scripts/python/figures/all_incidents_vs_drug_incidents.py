"""Script which compares all incident counts per county vs. drug incidents."""
from operator import index
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
tqdm.pandas()

base_path = Path(__file__).parents[3] / "data"


def wilson_error(n_s: int, n: int, z=1.96):
    """
    Wilson score interval

    param n_s: number of successes
    param n: total number of events
    param z: The z-value

    return: The lower and upper bound of the Wilson score interval
    """
    n_f = np.max([1, n - n_s])
    denom = n + z ** 2
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2 / 4))
    return adjusted_p, ci


def wilson_selection(n_s_1, n_1, n_s_2, n_2) -> Tuple[float, float]:
    """
    Get the adjusted selection bias and wilson cis.
    """
    p1, e1 = wilson_error(n_s_1, n_1)
    p2, e2 = wilson_error(n_s_2, n_2)
    sr = p1 / p2
    ci = np.sqrt((e1 / p1) ** 2 + (e2 / p2) ** 2) * sr
    return sr, ci, e1, e2


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def load_drug_incidents_pp(drug_list: List[str] = ["meth", "cocaine", "crack", "heroin"], arrests: bool = False):
    """Get the number of drug incidents per person."""
    def _load_drug(drug: str):
        df = pd.read_csv(base_path / "output" / f"selection_ratio_county_2010-2020_wilson_{drug}{'_arrests' if arrests else ''}.csv", dtype={"FIPS": str}, usecols=[
            "FIPS", "year", "black_incidents", "white_incidents", "black_population", "white_population"])
        return df[["FIPS", "black_incidents", "white_incidents", "year"]]
    df = pd.concat([_load_drug(drug) for drug in drug_list]
                   ).groupby(["FIPS", "year"]).agg({"black_incidents": sum, "white_incidents": sum}).reset_index()
    pop_df = pd.read_csv(base_path / "output" / f"selection_ratio_county_2010-2020_wilson_{drug_list[0]}.csv", dtype={"FIPS": str}, usecols=[
        "FIPS", "year", "black_population", "white_population"])
    df = df.merge(pop_df, on=["FIPS", "year"])
    df["black_incidents_p100k"] = (
        df.black_incidents / df.black_population) * 100_000
    df["white_incidents_p100k"] = (
        df.white_incidents / df.white_population) * 100_000
    return df[["FIPS", "year", "white_incidents_p100k", "black_incidents_p100k", "black_population", "white_population"]]


def load_all_incidents(arrests: bool = False):
    """Get the number of all incidents per county."""
    df = pd.read_csv(base_path / "NIBRS" / "raw" /
                     f"all_incident_count_2010-2020{'_arrests' if arrests else ''}.csv", dtype={"FIPS": str}, usecols=["FIPS", "race", "data_year"])
    df = df.groupby(["FIPS", "race", "data_year"]).progress_apply(
        lambda x: x.shape[0]).to_frame("count").reset_index()
    df = df.pivot(index=["FIPS", "data_year"], columns="race",
                  values="count").reset_index()
    df["incidents"] = df.black + df.white
    df = df.rename(columns={"black": "black_incidents",
                   "white": "white_incidents", "data_year": "year"})
    return df[["FIPS", "year", "incidents", "black_incidents", "white_incidents"]]


def plot(df):
    sns.scatterplot(x="all_black_incidents_p100k",
                    y="black_incidents_p100k", data=df)
    plt.ylabel("Black Drug Incidents per 100k")
    plt.xlabel("All Black Incidents per 100k")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(base_path.parent / "plots" /
                "black_incidents_p100k_vs_incidents.pdf")
    plt.clf()
    sns.scatterplot(x="all_white_incidents_p100k",
                    y="white_incidents_p100k", data=df)
    plt.ylabel("White Drug Incidents per 100k")
    plt.xlabel("All White Incidents per 100k")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(base_path.parent / "plots" /
                "white_incidents_p100k_vs_incidents.pdf")
    plt.clf()
    sns.scatterplot(x="all_ratio", y="drug_ratio", data=df)
    plt.ylabel("Drug Incidents Ratio per 100k")
    plt.xlabel("All Incidents Ratio per 100k")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(base_path.parent / "plots" /
                "incident_ratio_p100k_vs_incident_ratio.pdf")


def main():
    # df["all_black_incidents_p100k"] = ((
    #     df.black_incidents / df.black_population) * 100_000) - df.black_incidents_p100k
    # df["all_white_incidents_p100k"] = ((
    #     df.white_incidents / df.white_population) * 100_000) - df.white_incidents_p100k
    # df["drug_ratio"] = df.black_incidents_p100k / df.white_incidents_p100k
    # df["all_ratio"] = df.all_black_incidents_p100k / \
    #     df.all_white_incidents_p100k
    if (base_path / "nibrs" / "all_incidents_per_county_arrests.csv").exists():
        df = pd.read_csv(base_path / "nibrs" /
                         "all_incidents_per_county_arrests.csv", dtype={"FIPS": str})
    else:
        df = load_all_incidents(True)
        df.to_csv(base_path / "nibrs" /
                  "all_incidents_per_county_arrests.csv", index=False)
    df_drug = load_drug_incidents_pp(drug_list=["heroin"])
    df = df.merge(df_drug, on=["FIPS", "year"])
    plot(df)


if __name__ == "__main__":
    main()
