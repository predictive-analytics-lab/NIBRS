"""Script which produces some visualizations for Fig.2 + SI. 

Specifically:
    Model distribution, difference distribution, and regression plots between baseline and comparison models.
"""
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

plt.style.use("ggplot")

data_path = Path(__file__).parents[3] / "data"
plots_path = Path(__file__).parents[3] / "plots"


data_dict = {
    "poverty": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg+Pov}",
    },
    "poverty_all": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_all_incidents.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg+Pov, All Incidents}",
    },
    "hispanic": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_hispanics.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg+Pov, Hispanics}",
    },
    "dui": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_dui.csv",
        "model": "\\texttt{DUI}\\textsubscript{Dmg+Pov, Arrests}",
    },
    "drunkeness": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_hispanics_drunkeness.csv",
        "model": "\\texttt{Drunkeness}\\textsubscript{Dmg+Pov, Arrests}",
    },
    "ucr_possesion": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_ucr_possesion.csv",
        "model": "\\texttt{Possesion}\\textsubscript{Dmg+Pov, Arrests}",
    },
    "arrests": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg+Pov, Arrests}",
    },
    "metro": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_metro.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg+Metro}",
    },
    "dems": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv",
        "model": "\\texttt{Use}\\textsubscript{Dmg}",
    },
    "buying_outside": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv",
        "model": "\\texttt{Purchase}\\textsubscript{Public}",
    },
    "buying": {
        "dataset": "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv",
        "model": "\\texttt{Purchase}",
    },
}


def load_csv(dataset_name: str, log: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        data_path / "output" / data_dict[dataset_name]["dataset"],
        dtype={"FIPS": str},
        usecols=["FIPS", "selection_ratio"],
    )
    df = df.rename(columns={"selection_ratio": "sr"})
    if log:
        df["sr"] = np.log(df["sr"])
    df["Model"] = data_dict[dataset_name]["model"]
    return df


def model_difference_data(
    base_dataset_name: str, comp_dataset_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_dataset = load_csv(base_dataset_name)
    comp_datasets = [load_csv(dataset_name) for dataset_name in comp_dataset_names]
    long_data = pd.concat(
        [base_dataset.copy(), *[ci.copy() for ci in comp_datasets]], ignore_index=True
    )
    base_dataset = base_dataset.rename(
        columns={
            "sr": "basesr",
            "Model": "Model_base",
        }
    )
    for i in range(len(comp_datasets)):
        comp_datasets[i] = base_dataset.merge(comp_datasets[i], on="FIPS")
    wide_data = pd.concat([ci.copy() for ci in comp_datasets], ignore_index=True)
    wide_data["difference"] = wide_data["sr"] - wide_data["basesr"]
    return long_data, wide_data


def model_difference_plots(
    base_dataset_name: str, comp_dataset_names: List[str], filename_prefix: str
) -> None:
    sns.set(
        font_scale=1.2,
        style="ticks",
        rc={"text.usetex": True, "legend.loc": "upper left"},
    )
    long_data, wide_data = model_difference_data(base_dataset_name, comp_dataset_names)
    difference_distribution(
        wide_data, base_dataset_name, comp_dataset_names, filename_prefix
    )
    distribution(long_data, filename_prefix)
    if len(comp_dataset_names) == 1:
        reg_plot(wide_data, base_dataset_name, comp_dataset_names, filename_prefix)
        get_model_coef(wide_data)


def difference_distribution(
    data: pd.DataFrame,
    base_dataset_name: str,
    comp_dataset_names: List[str],
    prefix: str,
) -> None:
    ax = sns.kdeplot(
        data=data,
        x="difference",
        hue="Model",
        fill=True,
        palette="colorblind",
        color="red",
        alpha=0.5,
        linewidth=0,
    )
    ax.set_xlim([-2, 2])

    ax.set_ylabel("")
    if len(comp_dataset_names) > 1:
        ax.set_xlabel(
            f"Log(enforcement ratio) - Log({data_dict[base_dataset_name]['model']})"
        )
    else:
        ax.set_xlabel(
            f"Log({data_dict[comp_dataset_names[0]]['model']}) - Log({data_dict[base_dataset_name]['model']})"
        )
    plt.tight_layout()
    plt.savefig(
        plots_path / f"{prefix}_difference_distribution.pdf", bbox_inches="tight"
    )
    plt.clf()
    plt.cla()
    plt.close()


def distribution(data: pd.DataFrame, prefix) -> None:
    ax = sns.histplot(
        data=data,
        x="sr",
        hue="Model",
        fill=True,
        palette="colorblind",
        alpha=0.5,
        linewidth=0,
    )
    ax.set_ylabel("")
    ax.set_xlabel("Log(Differential Enforcement Ratio)")
    ax.set_xlim([-5, 5])
    plt.tight_layout()
    plt.savefig(plots_path / f"{prefix}_distribution.pdf", bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def reg_plot(
    data: pd.DataFrame,
    base_dataset_name: str,
    comp_dataset_names: List[str],
    prefix: str,
) -> None:
    g = sns.lmplot(
        data=data,
        x="basesr",
        y="sr",
        scatter_kws={"color": "black"},
        line_kws={"color": "red"},
    )
    g.ax.set_ylabel(f"Log({data_dict[comp_dataset_names[0]]['model']})")
    g.ax.set_xlabel(f"Log({data_dict[base_dataset_name]['model']})")
    g.ax.spines["right"].set_visible(True)
    g.ax.spines["top"].set_visible(True)
    plt.savefig(plots_path / f"{prefix}_correlation.pdf", bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def get_model_coef(data) -> None:
    y, X = dmatrices(
        f"sr ~ basesr",
        data=data,
        return_type="dataframe",
    )
    model = sm.OLS(
        y,
        X,
    )
    model_res = model.fit()
    model_res = model_res.get_robustcov_results(cov_type="HC1")
    print(model_res.summary())


if __name__ == "__main__":
    model_difference_plots(
        "poverty",
        ["dems", "metro", "buying", "buying_outside", "arrests"],
        "all_models",
    )
    model_difference_plots("poverty", ["hispanic"], "hispanic")
    model_difference_plots("poverty", ["poverty_all"], "all_incidents")
    model_difference_plots("arrests", ["dui"], "dui")
    model_difference_plots("arrests", ["drunkeness"], "drunkeness")
    model_difference_plots("arrests", ["ucr_possesion"], "ucr_possesion")
