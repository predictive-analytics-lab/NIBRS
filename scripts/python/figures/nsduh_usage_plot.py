import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plot_path = Path(__file__).parents[3] / "plots"

df = pd.read_csv(
    Path(__file__).parents[3] / "data" / "NSDUH" / "nsduh_processed_2019.csv"
)

sns.set_context("paper", font_scale=5)
sns.set_style("whitegrid")

g = sns.catplot(
    data=df[(~df.poverty.isnull()) & (df.race != "other")],
    x="race",
    y="MJ",
    hue="poverty",
    col="sex",
    row="age",
    kind="bar",
    palette="colorblind",
    height=11,
    aspect=1.5,
    legend=False
)

g.set_axis_labels(y_var="P(Use|race,age,sex,poverty)")

for ax in g.axes.flat:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                    box.width, box.height * 0.8])

# Put a legend below current axis
ax = plt.gca()
ax.legend(loc='upper center', bbox_to_anchor=(0, -0.15),
          fancybox=True, shadow=True, ncol=5)

plt.savefig(plot_path / "usage_plot_2019.pdf")
