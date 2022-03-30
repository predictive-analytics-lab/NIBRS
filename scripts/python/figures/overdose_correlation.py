"""Various plots and tables using the drug overdose dataset."""
from typing import List
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joypy

def load_overdose_dataset(years: list = [2015]):
    """Load the overdose dataset."""
    df =  pd.read_csv(Path(__file__).parents[3] / "data" / "misc" / "NCHS_-_Drug_Poisoning_Mortality_by_County__United_States.csv", usecols=["FIPS", "Estimated Age-adjusted Death Rate, 16 Categories (in ranges)", "Year"], dtype={"FIPS": str})
    df = df.rename(columns={"Estimated Age-adjusted Death Rate, 16 Categories (in ranges)": "Death Rate", "Year": "year"})

    df = df[df["year"].isin(years)]
    def float_conv(x):
        try:
            return float(x) if ">" not in x else 30
        except:
            return np.nan
    df["lower"] = df["Death Rate"].str.split("-").str[0].apply(float_conv)
    df["upper"] = df["Death Rate"].str.split("-").str[1].apply(float_conv)
    return df

def load_enforcement_rates(drug: str, years: List[int] = [2015], threshold: int = 10):
    """Load the enforcement rates dataset."""
    years = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" / f"selection_ratio_county_{years}_bootstraps_1000{drug}.csv", dtype={"FIPS": str})
    df = df[(df["black_incidents"] + df["white_incidents"]) > threshold]
    df["enforcement_rate"] = (df["black_incidents"] + df["white_incidents"]) / (df["black_users"] + df["white_users"])
    #normalize between 0 and 1 - use upper quartile rather than max
    return df[["enforcement_rate", "FIPS", "year"]]

def plot_overdose_boxplot(df: pd.DataFrame, name: str):
    """Plot the correlation between overdose rates and enforcement rates."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # sort df by lower
    df = df.sort_values(by="lower")

    # drop rows where lower = 0
    df = df[df["lower"] != 0]

    # normalize enforcement rate
    df["enforcement_rate"] = df["enforcement_rate"] / df["enforcement_rate"].quantile(0.95)
    df["enforcement_rate"] = df["enforcement_rate"].clip(upper=1)

    sns.boxplot(x="lower", y="enforcement_rate", data=df, ax=ax, linewidth=2.5, showfliers=False, whis=0.5)
    sns.swarmplot(x="lower", y="enforcement_rate", data=df, color=".25", alpha=0.5, ax=ax, linewidth=0, size=3.5)
    plt.ylim(0, 1)
    # plt.vlines(x=df["enforcement_rate"].median(), ymin=-1, ymax=2, linewidth=2, color="black", linestyle="--")
    plt.xlabel("Death Rate")
    plt.ylabel("Enforcement Rate")
    plt.savefig(Path(__file__).parents[3] / "plots" / f"overdose_dist_{name}.png")
    plt.clf()

def calc_year_gradients(df: pd.DataFrame):
    from scipy.stats import linregress
    def regress_wrapper(x, y):
        return linregress(x, y).slope
    er_df = df.groupby(["FIPS"]).apply(lambda x: regress_wrapper(x["year"], x["enforcement_rate"])).to_frame("er_slope").reset_index()
    dr_df = df.groupby(["FIPS"]).apply(lambda x: regress_wrapper(x["year"], x["lower"])).to_frame("dr_slope").reset_index()
    return er_df.merge(dr_df, on="FIPS").fillna(0)


def gradient_scatter(df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x="er_slope", y="dr_slope", data=df, ax=ax)
    plt.xlabel("Enforcement Rate Gradient")
    plt.ylabel("Death Rate Gradient")
    plt.savefig(Path(__file__).parents[3] / "plots" / f"overdose_grad_{name}.png")
    plt.clf()

def main(drug_filename: str, drug_savename):
    df = load_overdose_dataset()
    df = df.merge(load_enforcement_rates(drug_filename), on="FIPS")
    plot_overdose_boxplot(df, drug_savename)

def main_grad(drug_filename: str, drug_savename):
    years = list(range(2010, 2016))
    df = load_overdose_dataset(years)
    odf = df.merge(load_enforcement_rates(drug_filename, years), on=["FIPS", "year"])
    df = calc_year_gradients(odf)
    gradient_scatter(df, drug_savename)


if __name__ == "__main__":
    # main("", "cannabis")
    # main("_cocaine", "cocaine")
    main_grad("_heroin", "heroin")
    main_grad("_crack", "crack")
    # main_grad("_meth", "meth")
    # main_grad("_all", "all_drug")