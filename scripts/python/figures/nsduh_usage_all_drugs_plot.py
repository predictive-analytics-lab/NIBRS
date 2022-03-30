"""Plot the usage of all drug types."""
from typing import List
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

base_path = Path(__file__).parents[3]
data_path = base_path / "data"
plot_path = base_path / "plots"


def load_drug(drug_col: str, drug_name: str, year: str) -> pd.DataFrame:
    df = pd.read_csv(data_path / "NSDUH" / f"nsduh_processed{drug_col}_{year}.csv", usecols=["race", "MJ", "year"])
    df["drug"] = drug_name.title()
    df["race"] = df.race.str.title()
    df = df.rename(columns={"MJ": "drug_use", "race": "Race"})
    df = df[df.drug_use > 0]
    # df["drug_use"] = np.log(df.drug_use)
    return df

def load_all_drugs() -> pd.DataFrame:
    df = pd.concat([
        load_drug("_crack", "crack", "2016-2020"), 
        load_drug("_cocaine", "cocaine", "2016-2020"), 
        load_drug("_meth", "methamphetamines", "2016-2020"), 
        load_drug("_heroin", "heroin", "2016-2020"),
        load_drug("_using", "cannabis", "2016-2020")], axis=0)
    return df

def barplot(df: pd.DataFrame) -> None:
    """Use seaborn to create a barplot."""
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale("log")
    g = sns.barplot(data=df, x="drug", y="drug_use", hue="Race", ax=ax, hue_order=["White", "Black"], ci=68, capsize=.05, errwidth=1, errcolor=".12")
    plt.ylabel("P(Use|race, drug)")
    plt.xlabel("Drug")
    plt.tight_layout()
    plt.savefig(plot_path / "use_barplot.pdf")

def pointplot(df):
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 10))
    df = df.sort_values(by="year")
    g = sns.catplot(data=df, kind="point", x="year", col="drug", col_wrap=2, y="drug_use", hue="Race", ax=ax, hue_order=["White", "Black", "Other"], ci=68, capsize=.05, errwidth=1, errcolor=".12", facet_kws={'sharey': True, 'sharex': True})
    g.set(ylabel="P(Use|race, drug)", xlabel="Year")
    for ax in g.axes.flat:
        ax.set_yscale("log")
    handles, labels = plt.gca().get_legend_handles_labels()
    g.legend.remove()
    # wide legend, fancy
    g.figure.legend(handles, labels, loc='lower center', ncol=5, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(plot_path / "use_pointplot.pdf")


if __name__ == "__main__":
    df = load_all_drugs()
    df = df[df.Race.isin(["Black", "White"])]
    barplot(df)
    pointplot(df)