"""Percentage change map for drug offenses in the U.S. by state."""
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import seaborn as sns
import numpy as np

base_path = Path(__file__).parents[3] / "data"


def get_data(icpsr_path: Path):
    df = pd.read_csv(icpsr_path, usecols=[
                     "RACE", "OFFGENERAL", "RPTYEAR", "STATE"], delimiter="\t")
    fips_abbrv = pd.read_csv(base_path / "misc" /
                             "FIPS_ABBRV.csv", dtype={"FIPS": int})
    # Only drug crimes
    df = df[df["OFFGENERAL"] == 3]
    df["RACE"] = df.RACE.map({1: "white", 2: "black"})
    df = df[~df.RACE.isnull()]
    df = df[df.RPTYEAR.isin([2009, 2019])]
    df = df.groupby(["STATE", "RPTYEAR", "RACE"]).count().reset_index()
    df = df.rename(columns={"OFFGENERAL": "count", "RACE": "race",
                            "RPTYEAR": "year", "STATE": "state_code"})
    df = df.merge(fips_abbrv, left_on="state_code", right_on="FIPS")
    df = df.rename(columns={"STATE": "state", "ABBRV": "state_abbrv"})
    df = df.pivot(index=["state", "state_abbrv", "race"],
                  columns="year", values="count").reset_index()
    df["perc_change"] = (df[2019] / df[2009]) * 100 - 100
    return df[["state_abbrv", "race", "perc_change"]]


def get_all_race_data(icpsr_path: Path):
    df = pd.read_csv(icpsr_path, usecols=[
                     "OFFGENERAL", "RPTYEAR", "STATE"], delimiter="\t")
    fips_abbrv = pd.read_csv(base_path / "misc" /
                             "FIPS_ABBRV.csv", dtype={"FIPS": int})
    # Only drug crimes
    df = df[df["OFFGENERAL"] == 3]
    df = df[df.RPTYEAR.isin([2009, 2019])]
    df = df.groupby(["STATE", "RPTYEAR"]).count().reset_index()
    df = df.rename(columns={"OFFGENERAL": "count",
                   "RPTYEAR": "year", "STATE": "state_code"})
    df = df.merge(fips_abbrv, left_on="state_code", right_on="FIPS")

    df = df.rename(columns={"STATE": "state", "ABBRV": "state_abbrv"})
    df = df.pivot(index=["state", "state_abbrv"],
                  columns="year", values="count").reset_index()
    df["perc_change"] = (df[2019] / df[2009]) * 100 - 100
    return df[["state_abbrv", "perc_change"]]


def tilegrid_plot(df: pd.DataFrame, race: Optional[str]):
    if race:
        df = df[df.race == race]
    tile_grid = pd.read_csv(base_path / "misc" / "USA_tilemap.csv")

    tile_grid = tile_grid.merge(
        df, left_on="state", right_on="state_abbrv", how="left")

    row_max = tile_grid.row.max()
    col_max = tile_grid.column.max()

    fig, axs = plt.subplots(
        row_max + 1, col_max + 1, figsize=(20, 12))

    # cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True, n=11)
    # normalize cmap
    norm = matplotlib.colors.Normalize(vmin=-100,
                                       vmax=100)

    for i in range(row_max + 1):
        for j in range(col_max + 1):
            df = tile_grid[(tile_grid.row == i) & (tile_grid.column == j)]
            if len(df) > 0:
                ax = axs[i, j]
                state = df.state.values[0]
                if not np.isnan(df.perc_change.values[0]):
                    cmap_val = df.perc_change.values[0]
                    ax.text(0.25, 0.8, f"{state}", ha="center",
                            va="center",  transform=ax.transAxes, fontsize=20,)
                    ax.text(0.7, 0.15, f"{cmap_val:.0f}%", ha="center",
                            va="center",  transform=ax.transAxes, fontsize=15,)
                    ax.set_facecolor(cmap(norm(cmap_val)))
                else:
                    ax.set_facecolor("white")
                    ax.text(0.2, 0.85, state, horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes, fontsize=20, color="black")
                ax.yaxis.set_major_locator(plt.NullLocator())
                ax.xaxis.set_major_locator(plt.NullLocator())
            else:
                axs[i, j].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ticks=np.linspace(-100, 100, 11),
                 boundaries=np.arange(-110, 120, 20), cax=cbar_ax)
    plt.savefig(base_path.parent / "plots" /
                f"incarceration_map_{race if race is not None else 'all'}.pdf")


def main():
    data_path = base_path / "misc" / "processed_incarceration_data.csv"
    all_race_data_path = base_path / "misc" / \
        "processed_incarceration_data_all_races.csv"
    if not data_path.exists():
        df = get_data(base_path / "misc" / "38048-0004-Data.tsv")
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    tilegrid_plot(df, "black")
    tilegrid_plot(df, "white")

    if not all_race_data_path.exists():
        all_df = get_all_race_data(base_path / "misc" / "38048-0004-Data.tsv")
        all_df.to_csv(all_race_data_path, index=False)
    else:
        all_df = pd.read_csv(all_race_data_path)

    tilegrid_plot(all_df, None)


if __name__ == "__main__":
    main()
