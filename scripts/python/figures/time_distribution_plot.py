import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")
sns.set(font_scale=1.2)

from pathlib import Path


data_path = Path(__file__).parents[3] / "data" / "NIBRS" / "raw"
plot_path = Path(__file__).parents[3] / "plots"


def time_distribution(df: pd.DataFrame, drug: str):
    df = df[~df.incident_hour.isnull()]

    df = df.rename(columns={"race": "Race"})

    df = df[df[f"{drug}_count"] > 0]

    sns.set_style("whitegrid")

    ax = sns.histplot(
        data=df,
        x="incident_hour",
        fill=True,
        palette="colorblind",
        alpha=0.5,
        stat="density",
        common_norm=False,
        linewidth=0,
        discrete=True,
    )

    ax.set_xlabel("Time")
    plt.title(f"{drug.title()} Incidents by Time")

    plt.savefig(plot_path / f"time_distribution_nosplit_{drug}.pdf")
    # flush the plot
    plt.clf()

if __name__ == "__main__":
    df = pd.read_csv(data_path / "drug_incidents_2010-2019.csv", usecols=["incident_hour", "data_year", "race", "crack_count", "cocaine_count", "heroin_count", "cannabis_count", "meth_count"])
    time_distribution(df, "meth")
    time_distribution(df, "crack")
    time_distribution(df, "cocaine")
    time_distribution(df, "cannabis")
    time_distribution(df, "heroin")
