# %%

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

df = pd.read_csv("/home/dev/Desktop/ttt.csv")

# %%

sns.set_style("white")

sns.catplot(
    data=df,
    y="reporting_year",
    color="red",
    sharey=False,
    col="state",
    alpha=0.3,
    kind="bar",
    x="year",
    height=4,
    aspect= 3
)
plt.axis("off")

plt.show()
# %%
