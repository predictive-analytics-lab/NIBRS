"""Percentage change map for drug offenses in the U.S. by state."""
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import seaborn as sns
import numpy as np

base_path = Path(__file__).parents[3] / "data"


def get_population():
    df = pd.read_csv(base_path / "census" / "census-2020-county.csv", usecols=[
                     "STATE", "YEAR", "WA_MALE", "WA_FEMALE", "BA_MALE", "BA_FEMALE", "AGEGRP"], dtype={"STATE": int, "YEAR": int}, engine="python", encoding="ISO-8859-1",)
    df = df[df.YEAR >= 3]
    df["YEAR"] += 2007
    df = df[df.AGEGRP == 0]
    df = df.replace("X", np.nan)
    df["black_population"] = df.BA_MALE.astype(
        float) + df.BA_FEMALE.astype(float)
    df["white_population"] = df.WA_MALE.astype(
        float) + df.WA_FEMALE.astype(float)
    df = df.groupby(["STATE", "YEAR"]).agg(
        {"black_population": "sum", "white_population": "sum"}).reset_index()
    df = df.rename(columns={"STATE": "state", "YEAR": "year",
                   "white_population": "white", "black_population": "black"})
    df = df.melt(id_vars=["state", "year"], value_vars=[
                 "black", "white"], var_name="race", value_name="population").reset_index()
    return df[["state", "year", "race", "population"]]


def get_data(icpsr_path: Path):
    df = pd.read_csv(icpsr_path, usecols=[
                     "RACE", "OFFGENERAL", "RPTYEAR", "STATE"], delimiter="\t")
    fips_abbrv = pd.read_csv(base_path / "misc" /
                             "FIPS_ABBRV.csv", dtype={"FIPS": int})
    popdf = get_population()
    # Only drug crimes
    df = df[df["OFFGENERAL"] == 3]
    df["RACE"] = df.RACE.map({1: "white", 2: "black"})
    df = df[~df.RACE.isnull()]
    df = df[df.RPTYEAR.isin(list(range(2009, 2020)))]
    df = df.groupby(["STATE", "RPTYEAR", "RACE"]).count().reset_index()
    df = df.rename(columns={"OFFGENERAL": "incarcerations", "RACE": "race",
                            "RPTYEAR": "year", "STATE": "state_code"})
    df = df.merge(fips_abbrv, left_on="state_code",
                  right_on="FIPS", how="left")
    df = df.merge(popdf, left_on=["state_code", "year", "race"], right_on=[
                  "state", "year", "race"])
    df = df.rename(columns={"STATE": "state", "ABBRV": "state_abbrv"})
    df["per_100k"] = (df.incarcerations / df.population) * 100_000
    return df[["state_abbrv", "race", "year", "per_100k"]]


def tilegrid_plot(df: pd.DataFrame, race: Optional[str]):
    tile_grid = pd.read_csv(base_path / "misc" / "USA_tilemap.csv")

    tile_grid = tile_grid.merge(
        df, left_on="state", right_on="state_abbrv", how="left")

    row_max = tile_grid.row.max()
    col_max = tile_grid.column.max()

    fig, axs = plt.subplots(
        row_max + 1, col_max + 1, figsize=(20, 12))

    def roundup(x):
        return int(np.ceil(x / 100.0)) * 100

    max_rate = np.ceil(np.max(np.log10(df.per_100k)))
    min_rate = np.floor(np.min(np.log10(df.per_100k)))

    for i in range(row_max + 1):
        for j in range(col_max + 1):
            df = tile_grid[(tile_grid.row == i) & (tile_grid.column == j)]
            if len(df) > 0:
                ax = axs[i, j]
                state = df.state.values[0]
                breakpoint()
                if not np.isnan(df.per_100k.values[0]):
                    ax.text(0.25, 0.2, f"{state}", ha="center",
                            va="center",  transform=ax.transAxes, fontsize=20,)
                    ax.set_yscale("log")

                    ax.plot(df[df.race == "black"].year, df[df.race ==
                            "black"].per_100k.values, color="blue", marker='o', markersize=2)
                    ax.plot(df[df.race == "white"].year, df[df.race ==
                            "white"].per_100k.values, color="green", marker='*', markersize=2)
                    ax.set_ylim([1, 1000])
                    ax.set_xlim([2010, 2019])
                    # ax.plot(df.year, df.per_100k, color="black",
                    #         linestyle='--',)
                    # ax.fill_between(
                    #     df[df.race == "black"].year, df[df.race ==
                    #                                     "black"].per_100k, where=df[df.race ==
                    #                                                                 "black"].per_100k >= 0, interpolate=True, color='blue', alpha=0.3)
                    # ax.fill_between(
                    #     df[df.race == "white"].year, df[df.race ==
                    #                                     "white"].per_100k, where=df[df.race ==
                    #                                                                 "white"].per_100k >= 0, interpolate=True, color='green', alpha=0.3)
                    if state == "WI":
                        ax.get_yaxis().get_major_formatter().labelOnlyBase = False
                        ax.set_xticks([df.year.min(), df.year.max()])
                    else:
                        ax.xaxis.set_major_locator(plt.NullLocator())
                        ax.yaxis.set_major_locator(plt.NullLocator())
                    # if state != "WI":
                    # ax.set_yticks([])
                else:
                    ax.set_facecolor("white")
                    ax.text(0.25, 0.2, state, horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes, fontsize=20, color="black")
                    ax.yaxis.set_major_locator(plt.NullLocator())
                    ax.xaxis.set_major_locator(plt.NullLocator())
            else:
                axs[i, j].axis('off')
    fig.tight_layout()
    # get legend from ax
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=[mpatches.Patch(color='blue', label='Black'), mpatches.Patch(
        color='green', label='White')], loc='lower center', ncol=2, fontsize=20, fancybox=True)
    plt.savefig(base_path.parent / "plots" /
                f"incarceration_map_lineplot_{race if race is not None else 'all'}.pdf")


def main():
    data_path = base_path / "misc" / "processed_incarceration_data_over_years.csv"
    if not data_path.exists():
        df = get_data(base_path / "misc" / "38048-0004-Data.tsv")
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    # tilegrid_plot(df, "black")
    tilegrid_plot(df, "both")


if __name__ == "__main__":
    main()
