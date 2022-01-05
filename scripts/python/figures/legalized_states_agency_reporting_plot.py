import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")

from pathlib import Path

plots_path = Path(__file__).parents[3] / "plots"
data_path = Path(__file__).parents[3] / "data" / "misc"


df = pd.read_csv(data_path / "county_reporting_legal_states.csv", index_col=0)

sns.set_style("white")

df["ry"] = df["reporting_year"]

sns.set_context("paper", font_scale=1.7)
sns.set_style("whitegrid")

g = sns.catplot(
    data=df,
    kind="bar",
    x="year",
    y="ry",
    col="state",
    col_wrap=2,
    color="#37474F",
    height=4,
    aspect=1.5,
    sharey=False,
    sharex=True,
)

g.set_xticklabels(rotation=45)

info = [
    {"name": "Colorado", "count": 64, "y": 6, "c": "white"},
    {"name": "Massachusetts", "count": 14, "y": 25, "c": "black"},
    {"name": "Michigan", "count": 83, "y": 25, "c": "white"},
    {"name": "Oregon", "count": 36, "y": 3.7, "c": "white"},
    {"name": "Vermont", "count": 14, "y": 20, "c": "black"},
    {"name": "Washington", "count": 39, "y": 6, "c": "white"},
]

for i, ax in enumerate(g.axes):
    infoi = info[i]
    ax.set_title(f"{infoi['name']}, Total Counties: {infoi['count']}")
    ax.set_ylim([0, infoi["count"]])
    ax.axhline(infoi["count"])


g.set_axis_labels(y_var="Number of reporting agencies", x_var="Year")
plt.savefig(plots_path / "agency_count.pdf")
