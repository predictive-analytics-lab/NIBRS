from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

output_dir = Path(__file__).parents[3] / "data" / "output"
data_dir = Path(__file__).parents[3] / "data" / "output"

dataframe_paths = {
    "Crack": data_dir / "selection_ratio_county_2010-2020_wilson_crack.csv",
    "Meth": data_dir / "selection_ratio_county_2010-2020_wilson_meth.csv",
    "Cocaine": data_dir / "selection_ratio_county_2010-2020_wilson_cocaine.csv",
    "Heroin": data_dir / "selection_ratio_county_2010-2020_wilson_heroin.csv",
    "Cannabis": data_dir / "selection_ratio_county_2010-2020_wilson.csv",
    "Non-drug Offenses": data_dir / "other_incidents_2010-2020.csv",
}


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


def usage_ratio():
    df = pd.read_csv(
        Path(__file__).parents[3] / "data" / "NSDUH" / "nsduh_usage_ratio.csv")
    df = df.groupby(["drug"]).rolling(
        3, on="year", center=True).sum().reset_index()
    df["usage_ratio"] = df.black / df.white
    df = df.rename(columns={"drug": "incident_type"})
    df["incident_type"] = df.incident_type.str.title()
    return df[["usage_ratio", "year", "incident_type"]]


def load_data(df: pd.DataFrame, incident_type: str) -> pd.DataFrame:
    df1 = df.melt(id_vars=["FIPS", "year"], value_vars=[
                  "black_incidents", "white_incidents"], var_name="race", value_name="incidents")
    df2 = df.melt(id_vars=["FIPS", "year"], value_vars=[
                  "black_population", "white_population"], var_name="race", value_name="population")
    df1["race"] = df1.race.apply(lambda x: x.split("_")[0])
    df2["race"] = df2.race.apply(lambda x: x.split("_")[0])
    df = df1.merge(df2, on=["FIPS", "year", "race"])

    def rolling(group):
        if len(group) < 3:
            return None
        rolled = group[["year", "incidents", "population"]].rolling(
            3, on="year", center=True).sum()
        rolled["race"] = group.race.values[0]
        rolled["FIPS"] = group.FIPS.values[0]
        return rolled.dropna()
    df = df.groupby(["race", "FIPS"]).apply(rolling).reset_index(drop=True)
    df["incident_rate"] = df.incidents / df.population
    df = df[df.incident_rate < 1]
    df["incident_rate_error"] = df.apply(
        lambda x: wilson_error(x.incidents, x.population)[1], axis=1)

    def weighted_average(group):
        stats = DescrStatsW(group.incident_rate, 1 /
                            group.incident_rate_error, ddof=0)
        return pd.Series({"weighted_incident_rate": stats.mean * 100_000, "weighted_incident_rate_std": (stats.std / np.sqrt(len(group))) * 100_000})
    df = df.groupby(["year", "race"]).apply(weighted_average).reset_index()
    df["log_weighted_incident_rate"] = np.log(
        df["weighted_incident_rate"])
    df["incident_type"] = incident_type
    return df


def weighted_line_plot(df: pd.DataFrame, filename: str):
    g = sns.FacetGrid(df, col="incident_type", hue="race", col_wrap=2,
                      height=4, aspect=1.2, sharey=False)

    def custom_lineplot(x, y, std, race, **kwargs):
        ax = sns.lineplot(x=x, y=y, legend=False, style=race,
                          markers=True, size=2, **kwargs)
        ax.fill_between(x, y - std, y + std, alpha=0.2, **kwargs)
    g.map(custom_lineplot, "year", "weighted_incident_rate",
          "weighted_incident_rate_std", "race")
    g.set(ylabel="Weighted Incidents Per 100K", xlabel="Year")
    g.add_legend(loc="lower center", ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(Path(__file__).parents[3] / "plots" / filename)
    plt.clf()


def add_expected_black(df: pd.DataFrame, use_df: pd.DataFrame):
    breakpoint()
    exp_df = df[df.race == "white"]
    exp_df["race"] = "black (expected)"
    exp_df = exp_df.merge(use_df, on=["year", "incident_type"])
    exp_df["weighted_incident_rate"] = exp_df["weighted_incident_rate"] * \
        exp_df["usage_ratio"]
    exp_df["weighted_incident_rate_std"] = np.log(
        np.exp(exp_df["weighted_incident_rate_std"]) * exp_df["usage_ratio"])
    exp_df["log_weighted_incident_rate"] = np.log(
        exp_df["weighted_incident_rate"])
    return pd.concat([df, exp_df])


def main():
    df = pd.concat([load_data(pd.read_csv(v), k)
                   for k, v in dataframe_paths.items()])
    df = add_expected_black(df, usage_ratio())
    weighted_line_plot(df, "weighted_incident_rate.pdf")


if __name__ == "__main__":
    main()
