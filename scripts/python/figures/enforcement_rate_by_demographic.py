# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

output_dir = Path(__file__).parents[3] / "data" / "output"
plot_dir = Path(__file__).parents[3] / "data" / "plots"


def plot_bars(data: pd.DataFrame) -> None:
    p = "colorblind"  # ["#a7ffeb", "#00695c"]
    g = sns.FacetGrid(
        data,
        row="age",
        col="sex",
        hue="kind",
        sharex=False,
        palette=p,
        height=5,
        aspect=5 / 5,
        gridspec_kws={"hspace": 0.4, "wspace": 0},
    )
    g = g.map(sns.barplot, "race", "value", ci=None).add_legend()
    g.despine(left=True)
    g.set_ylabels("Enforcement rate")
    for ax in g.axes.flat:
        for patch in ax.patches:
            patch.set_edgecolor("black")
    plt.savefig("figS5.pdf")
    plt.close()


def plot_lines(data: pd.DataFrame) -> None:
    p = "colorblind"  # ["#a7ffeb", "#00695c"]
    g = sns.FacetGrid(
        data,
        row="age",
        col="sex",
        hue="race/kind",
        hue_kws={"ls": ["-", "-", "--", "--"]},
        palette=p,
        sharex=False,
        height=5,
        aspect=5 / 5,
        gridspec_kws={"hspace": 0.4, "wspace": 0},
    )
    g = g.map(sns.lineplot, "year", "value", ci=None).add_legend()
    g.despine(left=True)
    g.set_ylabels("Enforcement rate")

    g.fig.subplots_adjust(top=0.85)
    # g.fig.suptitle("Number of Incidents/Arrests per 100,000 Cannabis user-years by demographic")
    for ax in g.axes.flat:
        for patch in ax.patches:
            patch.set_edgecolor("black")
    plt.savefig("figS6.pdf")
    plt.close()


def enforcement_rate(data: pd.DataFrame, kind: str, by_year: bool):
    vars = ["race", "age", "sex"]
    if by_year:
        vars += ["year"]
    data = data.groupby(vars).sum().reset_index()
    data["value"] = data[kind] / data["users"]
    data["kind"] = kind
    data["race/kind"] = data["race"] + " " + data["kind"]
    return data


def load_data(by_year: bool = False):
    incident_data = pd.read_csv(
        output_dir / "selection_ratio_county_2010-2019_poverty_not_aggregated.csv"
    )
    arrest_data = pd.read_csv(
        output_dir
        / "selection_ratio_county_2010-2019_poverty_arrests_not_aggregated.csv"
    )
    incident_er = enforcement_rate(incident_data, "incidents", by_year)
    arrest_er = enforcement_rate(arrest_data, "arrests", by_year)
    return pd.concat([incident_er, arrest_er], axis=0)


data = load_data()
data_by_year = load_data(by_year=True)
plot_bars(data)
plot_lines(data_by_year)
# %%
