# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from pathlib import Path

data_path = Path(__file__).parents[3] / 'data'

sb_df = pd.read_csv(data_path / "output" / 'selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv', dtype={"FIPS": str})

sb_df["selection_ratio_log10"] = np.log10(sb_df["selection_ratio"])

# sb_df = sb_df[sb_df["selection_ratio"] / sb_df["ci"] >= 2]

# %%

res = stats.linregress(x=sb_df.bwratio, y=sb_df.selection_ratio_log10)

g = sns.jointplot(data=sb_df, x="bwratio", y="selection_ratio_log10", kind="reg")
plt.text(x=0.5, y=2.2, s=f"slope: {res.slope:.3f}\n p-value: {res.pvalue:.3f}")

g.ax_joint.set_ylabel("log10(selection_ratio)")
g.ax_joint.set_xlabel("Black / White Population")
# g.ax_joint.set_xlim([0, 0.2])
# g.ax_joint.set_ylim([0, 2])

plt.show()
# %%
