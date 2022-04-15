from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

output_dir = Path(__file__).parents[3] / "data" / "output"
data_dir = Path(__file__).parents[3] / "data" / "output"

incident_paths = {
    "Crack": data_dir / "selection_ratio_county_2010-2020_wilson_crack.csv",
    "Meth": data_dir / "selection_ratio_county_2010-2020_wilson_meth.csv",
    "Cocaine": data_dir / "selection_ratio_county_2010-2020_wilson_cocaine.csv",
    "Heroin": data_dir / "selection_ratio_county_2010-2020_wilson_heroin.csv",
    "Cannabis": data_dir / "selection_ratio_county_2010-2020_wilson.csv",
    "Non-drug Offenses": data_dir / "other_incidents_2010-2020.csv",
}

arrest_paths = {
    "Crack": data_dir / "selection_ratio_county_2010-2020_wilson_crack_arrests.csv",
    "Meth": data_dir / "selection_ratio_county_2010-2020_wilson_meth_arrests.csv",
    "Cocaine": data_dir / "selection_ratio_county_2010-2020_wilson_cocaine_arrests.csv",
    "Heroin": data_dir / "selection_ratio_county_2010-2020_wilson_heroin_arrests.csv",
    "Cannabis": data_dir / "selection_ratio_county_2010-2020_wilson_arrests.csv",
    "Non-drug Offenses": data_dir / "other_incidents_2010-2020_arrests.csv",
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


def load_data(incident_path: Path, arrest_path: Path, incident_type: str) -> pd.DataFrame:
    incident_df = pd.read_csv(incident_path, usecols=[
                              "FIPS", "year", "black_incidents", "white_incidents", "white_population", "black_population"])
    pop_df = incident_df.melt(id_vars=["FIPS", "year"], value_vars=[
        "black_population", "white_population"], var_name="race", value_name="population")
    incident_df = incident_df.melt(id_vars=["FIPS", "year"], value_vars=[
        "black_incidents", "white_incidents"], var_name="race", value_name="incidents")

    incident_df["race"] = incident_df.race.apply(lambda x: x.split("_")[0])
    pop_df["race"] = pop_df.race.apply(lambda x: x.split("_")[0])
    incident_df = incident_df.merge(pop_df, on=["FIPS", "year", "race"])
    arrest_df = pd.read_csv(arrest_path, usecols=[
                            "FIPS", "year", "black_arrests", "white_arrests"])
    arrest_df = arrest_df.melt(id_vars=["FIPS", "year"], value_vars=[
        "black_arrests", "white_arrests"], var_name="race", value_name="arrests")
    arrest_df["race"] = arrest_df.race.apply(lambda x: x.split("_")[0])
    df = incident_df.merge(arrest_df, on=["FIPS", "year", "race"])

    def weighted_average(group):
        group = group[group.incidents > 0]
        group = group[group.arrests > 0]
        group["rate"] = group.arrests / group.incidents
        group = group[~group.rate.isnull()]
        stats = DescrStatsW(group.rate, group.population, ddof=0, )
        return pd.Series({"weighted_arrest_rate": stats.mean, "weighted_arrest_rate_std": (stats.std / np.sqrt(len(group)))})
    df = df.groupby(["race", "year"]).apply(weighted_average)
    df["incident_type"] = incident_type
    return df


def arrest_ratio():
    sns.set(font_scale=1.5, rc={'text.usetex': True}, style="whitegrid")

    df = pd.concat([load_data(incident_paths[k], arrest_paths[k], k)
                   for k in incident_paths.keys()]).reset_index()
    g = sns.FacetGrid(df, col="incident_type", hue="race", col_wrap=2,
                      height=4, aspect=0.8, sharey=True)

    def custom_lineplot(x, y, std, race, **kwargs):
        ax = sns.lineplot(x=x, y=y, legend=False, style=race,
                          markers=True, size=2, **kwargs)
        ax.fill_between(x, y - std, y + std, alpha=0.2, **kwargs)
    g.map(custom_lineplot, "year", "weighted_arrest_rate",
          "weighted_arrest_rate_std", "race")
    g.set(ylabel="Weighted Arrest Rate", xlabel="Year")
    g.add_legend(loc="lower center", ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(Path(__file__).parents[3] / "plots" / "arrest_ratio.pdf")


if __name__ == "__main__":
    arrest_ratio()
