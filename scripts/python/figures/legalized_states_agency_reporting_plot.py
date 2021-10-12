import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plots_path = Path(__file__).parents[3] / "plots"

plt.style.use("ggplot")

from pathlib import Path

sns.set(font_scale=1.1, rc={"text.usetex": True})

df = pd.read_csv("/home/dev/Desktop/ttt.csv", index_col=0)

sns.set_style("white")

df["ry"] = df["reporting_year"]

g = sns.catplot(
    data=df,
    kind="bar",
    x="year",
    y="ry",
    col="state",
    col_wrap=3,
    color="#37474F",
    height=3,
    aspect=1.3,
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
    for p in ax.patches:
        ax.text(
            p.get_x() + 0.17,
            infoi["y"],
            f"$\\frac{{{int(p.get_height())}}}{{{infoi['count']}}}$",
            color=infoi["c"],
            rotation="horizontal",
            size="large",
        )
        ax.set_title(infoi["name"])


g.set_axis_labels(y_var="Number of reporting agencies", x_var="Year")
plt.savefig(plots_path / "agency_count.pdf")
