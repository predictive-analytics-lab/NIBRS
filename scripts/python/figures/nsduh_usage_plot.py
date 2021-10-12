import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plot_path = Path(__file__).parents[3] / "plots"

df = pd.read_csv(
    Path(__file__).parents[3] / "data" / "NSDUH" / "nsduh_processed_2019.csv"
)
ax = sns.catplot(
    data=df[(~df.poverty.isnull()) & (df.race != "other")],
    x="race",
    y="MJ",
    hue="poverty",
    row="sex",
    col="age",
    kind="bar",
    palette="colorblind",
)
ax.set_axis_labels(y_var="P(Use|race,age,sex,poverty)")

plt.savefig(plot_path / "usage_plot_2019.pdf")
