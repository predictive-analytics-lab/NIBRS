
from typing import Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def load_and_merge_data(data_path: Path, drug_dict: dict, year_spec: str = "2017-2019_grouped") -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _load_drug(drug: str, drug_fn: str) -> pd.DataFrame:
        drug_df = pd.read_csv(
            data_path / f"selection_ratio_county_{year_spec}_bootstraps_1000{'_' + drug_fn if len(drug_fn) > 0 else drug_fn}.csv",
            dtype={"FIPS": str},
            usecols=["black_incidents", "black_users", "white_incidents", "white_users", "FIPS"],
            )
        drug_df["drug"] = drug

        drug_df["enforcement_rate_black"] = np.log(drug_df.black_incidents / drug_df.black_users)
        drug_df["enforcement_rate_white"] = np.log(drug_df.white_incidents / drug_df.white_users)
        return drug_df[["FIPS", "drug", "enforcement_rate_black", "enforcement_rate_white"]]
    df = pd.concat([_load_drug(drug, drug_dict[drug]) for drug in drug_dict], axis=0)
    black_df = df.pivot(index="FIPS", columns="drug", values="enforcement_rate_black")
    white_df = df.pivot(index="FIPS", columns="drug", values="enforcement_rate_white")
    return black_df, white_df

def corr_heatmap(df: pd.DataFrame, race: str):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.color_palette("mako", as_cmap=True, )
    sns.heatmap(corr, mask=mask, cmap=cmap,
                square=True, linewidths=.5, vmin=0, vmax=1, annot=True, annot_kws={'size': 15},)
    # set ax font size large
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.suptitle(f"Drug Correlation Heatmap -  {race} Individuals")
    plt.savefig(data_path.parent.parent / "plots" / f"corr_heatmap_{race}.pdf")


if __name__ == "__main__":
    drug_dict = {
        "cannabis": "",
        "cocaine": "cocaine",
        "crack": "crack",
        "heroin": "heroin",
        "meth": "meth",
    }
    data_path = Path(__file__).parents[3] / "data" / "output"
    black_df, white_df = load_and_merge_data(data_path, drug_dict)
    corr_heatmap(black_df, "Black")
    corr_heatmap(white_df, "White")

