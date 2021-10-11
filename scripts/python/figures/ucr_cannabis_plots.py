# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")

from pathlib import Path

# %%

data_path = Path(__file__).parents[3] / "data"
plots_path = Path(__file__).parents[3] / "plots"


arrests = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "black_arrests", "white_arrests"],
)

arrests["Model"] = "\\texttt{Use}\\textsubscript{Dmg+Pov, Arrests}"
arrests["selection_ratio"] = np.log(arrests["selection_ratio"])


possesion = pd.read_csv(
    data_path
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_ucr_possesion.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "black_incidents", "white_incidents"],
)

possesion["Model"] = "\\texttt{Possesion}\\textsubscript{Dmg+Pov, Arrests}"
possesion["selection_ratio"] = np.log(possesion["selection_ratio"])

possesion = possesion[possesion.FIPS.isin(arrests.FIPS)]
arrests = arrests[arrests.FIPS.isin(possesion.FIPS)]


def remove_nonsense(df):
    if "black_arrests" in df.columns:
        df = df[df.black_arrests > 0.5]
        df = df[df.white_arrests > 0.5]
    else:
        df = df[df.black_incidents > 0.5]
        df = df[df.white_incidents > 0.5]
    return df


arrests = remove_nonsense(arrests)
possesion = remove_nonsense(possesion)

data = pd.concat([possesion.copy(), arrests.copy()], ignore_index=True)


arrests = arrests.rename(
    columns={
        "selection_ratio": "selection_ratio_base",
        "Model": "Model_base",
    }
)

possesion = arrests.merge(possesion, on="FIPS")
possesion["diff"] = possesion["selection_ratio"] - possesion["selection_ratio_base"]
# %%

sns.set(font_scale=1.2, rc={"text.usetex": True, "legend.loc": "upper left"})


sns.set_style("ticks")
ax = sns.histplot(
    data=possesion,
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
    "Log(enforcement ratio Use\\textsubscript{Dmg+Pov, Arrests}) - Log(enforcement ratio Possesion\\textsubscript{Dmg+Pov, Arrests})"
)
plt.tight_layout()
plt.savefig(plots_path / "possesion_difference_distribution.pdf", bbox_inches="tight")
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

plt.savefig(plots_path / "possesion_distribution.pdf", bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()
# plt.savefig(plots_path / "possesion_distribution.pdf")
# plt.show()
# %%
from scipy import stats

sns.set_style("white")

possesion["srb"] = possesion["selection_ratio_base"]
possesion["sr"] = possesion["selection_ratio"]

ax = sns.lmplot(
    data=possesion,
    x="srb",
    y="sr",
    scatter_kws={"color": "black"},
    line_kws={"color": "red"},
)
# plt.plot(np.linspace(-4, 8), np.linspace(-4, 8), color="black")

ax.ax.set_ylabel(
    "Log(\\texttt{Possesion}\\textsubscript{Dmg+Pov, Arrests} Differential Enforcement Ratio)"
)
ax.ax.set_xlabel(
    "Log(\\texttt{Use}\\textsubscript{Dmg+Pov, Arrests} Differential Enforcement Ratio)"
)

ax.ax.set_xlim([-4, 8])
ax.ax.set_ylim([-4, 8])

ax.ax.spines["right"].set_visible(True)
ax.ax.spines["top"].set_visible(True)
# res = stats.linregress(x=possesion.selection_ratio_base, y=possesion.selection_ratio)
# plt.text(x=0.5, y=8, s=f"slope: {res.slope:.6f}\n ci: {res.pvalue:.6f}")

plt.savefig(plots_path / "possesion_correlation.pdf", bbox_inches="tight")
# plt.show()
plt.clf()
plt.cla()
plt.close()
# %%

import statsmodels.api as sm
from patsy import dmatrices

y, X = dmatrices(
    f"selection_ratio ~ selection_ratio_base", data=possesion, return_type="dataframe"
)
model = sm.OLS(
    y,
    X,
)
model_res = model.fit()
model_res = model_res.get_robustcov_results(cov_type="HC1")
print(model_res.summary())
# %%
