import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")
sns.set(font_scale=1.2, rc={"text.usetex": True})

from pathlib import Path


data_path = Path(__file__).parents[3] / "data" / "NIBRS" / "raw"
plot_path = Path(__file__).parents[3] / "plots"

df = pd.read_csv(
    data_path / "cannabis_allyears.csv", usecols=["incident_hour", "data_year", "race"]
)


df = df[~df.incident_hour.isnull()]

df = df.rename(columns={"race": "Race"})

sns.set_style("whitegrid")

ax = sns.histplot(
    data=df,
    x="incident_hour",
    fill=True,
    hue="Race",
    palette="colorblind",
    alpha=0.5,
    stat="proportion",
    common_norm=False,
    linewidth=0,
    discrete=True,
)

ax.set_xlabel("Time")

plt.savefig(plot_path / "time_distribution.pdf")
