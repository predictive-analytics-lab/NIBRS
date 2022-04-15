"""Weighted Enforcement Rate Line Plot"""
from operator import concat
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

output_dir = Path(__file__).parents[3] / "data" / "output"


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


def condition_table():
    def get_size(year_range: str, drug: str, condition: int = 1):
        df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" /
                         f"selection_ratio_county_{year_range}_wilson{'_' + drug if len(drug) > 0 else drug}.csv", dtype={"FIPS": str}, usecols=["year", "FIPS"])
        df = df.groupby(["FIPS"]).filter(lambda x: len(x) >= condition)
        df = df.groupby("year").apply(lambda x: int(
            x.FIPS.nunique())).to_frame("num_counties").reset_index()
        drug = drug if len(drug) > 0 else "cannabis"
        df["condition"] = drug + "_" + str(condition)
        return df
    df = pd.concat([
        get_size("2015-2020", "meth"),
        get_size("2015-2020", "meth", 3),
        get_size("2015-2020", "meth", 6),
        get_size("2010-2020", "crack"),
        get_size("2010-2020", "crack", 3),
        get_size("2010-2020", "crack", 11),
        get_size("2010-2020", "cocaine"),
        get_size("2010-2020", "cocaine", 3),
        get_size("2010-2020", "cocaine", 11),
        get_size("2010-2020", "heroin"),
        get_size("2010-2020", "heroin", 3),
        get_size("2010-2020", "heroin", 11),
        get_size("2010-2020", ""),
        get_size("2010-2020", "", 3),
        get_size("2010-2020", "", 11),
    ], axis=0)
    df["cond_num"] = df.condition.apply(lambda x: int(x.split("_")[1]))
    df["drug"] = df.condition.apply(lambda x: x.split("_")[0])
    df["num_counties"] = df.num_counties.apply(
        lambda x: int(x) if not np.isnan(x) else 0)
    df = df.sort_values(by=["drug", "cond_num"])
    order = df.condition.unique()
    df = df.pivot(index=["year"], columns=["condition"], values="num_counties")
    df = df.reindex(order, axis=1)
    df.to_csv(Path(__file__).parents[3] /
              "data" / "misc" / "condition_table.csv")


def usage_ratio():
    df = pd.read_csv(
        Path(__file__).parents[3] / "data" / "NSDUH" / "nsduh_usage_ratio.csv")
    df = df.groupby(["drug"]).rolling(
        3, on="year", center=True).sum().reset_index()
    df["usage_ratio"] = df.black / df.white
    return df[["usage_ratio", "year", "drug"]]


def usage_ratio_plot(df: pd.DataFrame):
    sns.lineplot(x="year", y="usage_ratio", hue="drug",
                 data=df, markers=True, style="drug")
    plt.legend(loc="lower center",  ncol=5,
               fancybox=True, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.yscale("log")
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel("Usage Ratio")
    plt.ylim(0.1, 10)
    plt.xlabel("Year")
    plt.savefig(Path(__file__).parents[3] / "plots" / "usage_ratio.pdf")


def load_data(drug: str, year_range: str, alt_colname: Optional[str] = None, smoothing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" / f"selection_ratio_county_{year_range}_wilson{'_' + drug if len(drug) > 0 else drug}.csv", dtype={
                     "FIPS": str}, usecols=["year", "FIPS", "black_incidents", "white_incidents", "white_population", "black_population", "coverage", "black_users", "white_users"])
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
    if smoothing:
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
    df["drug"] = drug if not alt_colname else alt_colname
    return df


def weighted_line_plot(df: pd.DataFrame, use_df: pd.DataFrame):

    # for ax, drug in zip(axs.flatten(), df.drug.unique()):
    #     ax.set_title(drug)
    #     df_drug = df[df.drug == drug]
    #     sns.lineplot(x="year", y="weighted_incident_rate", hue="race",
    #                  markers=True, size=2, data=df_drug, ax=ax, err_style=None)
    #     for race in df_drug.race.unique():
    #         df_drug_race = df_drug[df_drug.race == race]
    #         ax.fill_between(df_drug_race.year, df_drug_race.weighted_incident_rate - df_drug_race.weighted_incident_rate_std,
    #                         df_drug_race.weighted_incident_rate + df_drug_race.weighted_incident_rate_std, alpha=0.2, color=color_dict[race])
    #         ax.set_xticks(df.year.unique())
    #     ax.set_ylim(0, (df.weighted_incident_rate +
    #                 df_drug_race.weighted_incident_rate_std).max())
    #     ax.set_xlabel("Year")
    #     ax.set_ylabel("Incident Rate Per 100K")
    exp_df = df[df.race == "white"]
    exp_df["race"] = "black (expected)"
    exp_df = exp_df.merge(use_df, on=["year", "drug"])
    exp_df["weighted_incident_rate"] = exp_df["weighted_incident_rate"] * \
        exp_df["usage_ratio"]
    exp_df["weighted_incident_rate_std"] = np.log(
        np.exp(exp_df["weighted_incident_rate_std"]) * exp_df["usage_ratio"])
    exp_df["log_weighted_incident_rate"] = np.log(
        exp_df["weighted_incident_rate"])

    df = pd.concat([df, exp_df])
    g = sns.FacetGrid(df, col="drug", hue="race", col_wrap=2,
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
    plt.savefig(Path(__file__).parents[3] / "plots" /
                f"weighted_incident_rate_lineplot.pdf")
    plt.clf()


if __name__ == "__main__":
    # condition_table()
    use_df = usage_ratio()
    # usage_ratio_plot(use_df)
    smoothing: bool = True
    meth_df = load_data("meth", "2010-2020", smoothing=smoothing)
    crack_df = load_data("crack", "2010-2020", smoothing=smoothing)
    cocaine_df = load_data(
        "cocaine", "2010-2020", smoothing=smoothing)
    heroin_df = load_data(
        "heroin", "2010-2020", smoothing=smoothing)
    cannabis_df = load_data(
        "", "2010-2020", alt_colname="cannabis", smoothing=smoothing)
    df = pd.concat([cannabis_df, crack_df, cocaine_df, heroin_df, meth_df])
    weighted_line_plot(df, use_df)
