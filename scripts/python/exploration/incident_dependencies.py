# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data_path = Path(__file__).parent.parent.parent.parent / "data"
data = pd.read_csv(data_path / "output" / "selection_ratio_county_2019.csv", index_col=0)
# %%
cmap = sns.diverging_palette(240, 10, l=65, center="dark", as_cmap=True)
data["Population"] = data["black"] + data["white"]
data["selection_ratio"] = np.log10(data["selection_ratio"])
data = data[np.abs(data.selection_ratio) < 2]
sns.scatterplot(data=data, x="Population", y="incidents", hue="selection_ratio", size="selection_ratio", palette=cmap)
# plt.ylim(0.01, 10)
plt.xscale("log")
# plt.ylabel("Selection Ratio")
# plt.xlabel("Incidents")
plt.show()
# %%
