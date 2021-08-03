# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


data_path = Path(__file__).parent.parent.parent / "data"
agency_lemas = pd.read_csv(data_path / "output" / "agency_lemas.csv", index_col=0)

agency_lemas = agency_lemas[agency_lemas.incidents >= 50]


# %%


### NEW COLUMNS ###


agency_lemas["bw_force_ratio"] = (agency_lemas["PERS_BLACK_MALE"] + agency_lemas["PERS_BLACK_FEM"]) / (agency_lemas["PERS_WHITE_MALE"] + agency_lemas["PERS_WHITE_FEM"])
agency_lemas["fm_force_ratio"] = (agency_lemas["PERS_BLACK_FEM"] + agency_lemas["PERS_WHITE_FEM"]) / (agency_lemas["PERS_WHITE_MALE"] + agency_lemas["PERS_BLACK_MALE"])

# %%

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
# rs = np.random.RandomState(33)


# Compute the correlation matrix
corr = agency_lemas.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(40, 35))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig("/home/dev/Desktop/heatmap.png")
# %%

sns.scatterplot(data=agency_lemas, y="selection_ratio", x="bwratio")
plt.xlim([0, 2])

# %%

from pandas_profiling import ProfileReport

profile = ProfileReport(agency_lemas, title=f"{data_path.stem}Report", minimal=False)
profile.to_file(Path(__file__).parent.parent.parent / "reports" / "report.html")
# %%
profile.to_file(Path(__file__).parent.parent.parent / "reports" / "report.html")

# %%
