# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")

from pathlib import Path

data_path = Path(__file__).parents[3] / "data"
plots_path = Path(__file__).parents[3] / "plots"

poverty = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)

poverty["Model"] = "\\texttt{Use}\\textsubscript{Dmg+Pov}"
poverty["selection_ratio"] = np.log(poverty["selection_ratio"])


all_incidents = pd.read_csv(
    data_path.parent
    / "turing_output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_all_incidents.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio"],
)

all_incidents["Model"] = "\\texttt{Use}\\textsubscript{Dmg+Pov, All Incidents}"
all_incidents["selection_ratio"] = np.log(all_incidents["selection_ratio"])

data = pd.concat([all_incidents.copy(), poverty.copy()], ignore_index=True)


poverty = poverty.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "Model": "Model_base",
    }
)

all_incidents = poverty.merge(all_incidents, on="FIPS")
all_incidents["diff"] = (
    all_incidents["selection_ratio"] - all_incidents["selection_ratio_base"]
)
# %%

sns.set(font_scale=1.2, rc={"text.usetex": True, "legend.loc": "upper left"})


sns.set_style("ticks")
ax = sns.histplot(
    data=all_incidents,
    x="diff",
    fill=True,
    palette="colorblind",
    color="red",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-5, 5])

ax.set_ylabel("")
ax.set_xlabel(
    "Log(enforcement ratio Use\\textsubscript{Dmg+Pov}) - Log(enforcement ratio Use\\textsubscript{Dmg+Pov, All Incidents})"
)
plt.tight_layout()
plt.savefig(
    plots_path / "all_incidents_difference_distribution.pdf", bbox_inches="tight"
)
plt.clf()
plt.cla()
plt.close()  # plt.show()
# %%

sns.set(font_scale=1.2, rc={"text.usetex": True, "legend.loc": "upper left"})


sns.set_style("ticks")
ax = sns.histplot(
    data=data,
    x="selection_ratio",
    hue="Model",
    fill=True,
    palette="colorblind",
    alpha=0.5,
    linewidth=0,
)
ax.set_xlim([-5, 5])

ax.set_ylabel("")
ax.set_xlabel("Log(Differential Enforcement Ratio)")

plt.savefig(plots_path / "all_incidents_distribution.pdf", bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()
# plt.savefig(plots_path / "all_incidents_distribution.pdf")
# plt.show()
# %%
from scipy import stats

sns.set_style("white")

all_incidents["srb"] = all_incidents["selection_ratio_base"]
all_incidents["sr"] = all_incidents["selection_ratio"]

ax = sns.lmplot(
    data=all_incidents,
    x="srb",
    y="sr",
    scatter_kws={"color": "black"},
    line_kws={"color": "red"},
)
# plt.plot(np.linspace(-4, 8), np.linspace(-4, 8), color="black")

ax.ax.set_ylabel(
    "Log(\\texttt{Use}\\textsubscript{Dmg+Pov, All Incidents} Differential Enforcement Ratio)"
)
ax.ax.set_xlabel(
    "Log(\\texttt{Use}\\textsubscript{Dmg+Pov} Differential Enforcement Ratio)"
)

ax.ax.set_xlim([-4, 8])
ax.ax.set_ylim([-4, 8])

ax.ax.spines["right"].set_visible(True)
ax.ax.spines["top"].set_visible(True)
# res = stats.linregress(x=all_incidents.selection_ratio_base, y=all_incidents.selection_ratio)
# plt.text(x=0.5, y=8, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

plt.savefig(plots_path / "all_incidents_correlation.pdf", bbox_inches="tight")
# plt.show()
plt.clf()
plt.cla()
plt.close()
# %%

import statsmodels.api as sm
from patsy import dmatrices

y, X = dmatrices(
    f"selection_ratio ~ selection_ratio_base",
    data=all_incidents,
    return_type="dataframe",
)
model = sm.OLS(
    y,
    X,
)
model_res = model.fit()
model_res = model_res.get_robustcov_results(cov_type="HC1")
print(model_res.summary())
# %%
