"""Box plots of enforcement rates and racial enforment ratio over time."""

from cProfile import run
from pathlib import Path
from typing import List
from xmlrpc.client import FastMarshaller
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def temporal_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth the temporal data."""
    def temp(x):
        return x.rolling(window=3, center=True).mean().shift(-3).dropna(subset=["selection_ratio"])
    df = df.groupby(["FIPS"]).apply(temp).reset_index(drop=True)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["year"] = df["year"].astype(int)
    return df

def filter_counties_entire_year_range(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the counties to only those that have data for the entire year range."""
    df = df.groupby(["FIPS"]).filter(lambda x: len(x) == df.year.nunique())
    return df

def load_data(drug: str, years: List[int] = list(range(2010, 2021)), threshold: int = 10, smoothing: bool = False):
    """Load the enforcement rates dataset."""
    if drug != "_meth":
        years = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    else:
        years = f"{max(2015, years[0])}-{years[-1]}" if len(years) > 1 else str(years[0])
    if drug == "_meth":
            df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" / f"selection_ratio_county_{years}_wilson{drug}.csv", dtype={"FIPS": str})
    else:
        df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" / f"selection_ratio_county_{years}_bootstraps_1000{drug}.csv", dtype={"FIPS": str})
    df = df[(df["black_incidents"] + df["white_incidents"]) > threshold]
    df = filter_counties_entire_year_range(df)
    if smoothing:
        df = temporal_smoothing(df)
    df["enforcement_rate"] = (df["black_incidents"] + df["white_incidents"]) / (df["black_users"] + df["white_users"])

    df["black_enforcement_rate"] = df["black_incidents"] / df["black_users"]
    df["white_enforcement_rate"] = df["white_incidents"] / df["white_users"]
    
    if drug == "_meth":
            df["black_enforcement_rate_er"] = df.ber_error
            df["white_enforcement_rate_er"] = df.wer_error
    else:
        df["black_enforcement_rate_er"] = df["black_enforcement_rate"] * np.sqrt((df["black_users_variance"] / df["black_users"]) ** 2 + (df["black_incident_variance"] / df["black_incidents"]) ** 2)
        df["white_enforcement_rate_er"] = df["white_enforcement_rate"] * np.sqrt((df["white_users_variance"] / df["white_users"]) ** 2 + (df["white_incident_variance"] / df["white_incidents"]) ** 2)
        df = enforcement_variance(df)
        # df["enforcement_rate"] = df["enforcement_rate"] / df["enforcement_rate"].max()
    df["log_selection_ratio"] = np.log(df["selection_ratio"])
    df = df[df.enforcement_rate < df.enforcement_rate.quantile(0.99)]
    df = df[df.enforcement_rate > df.enforcement_rate.quantile(0.01)]
    df["log_enforcement_rate"] = np.log(df["enforcement_rate"])
    #normalize between 0 and 1 - use upper quartile rather than max
    return df[["log_selection_ratio", "log_enforcement_rate", "selection_ratio", "black_enforcement_rate", "white_enforcement_rate", "white_enforcement_rate_er", "black_enforcement_rate_er", "enforcement_rate", "FIPS", "year"]]


def enforcement_variance(df):
    df["incidents"] = df["black_incidents"] + df["white_incidents"]
    df["users"] = df["black_users"] + df["white_users"]
    df["incident_variance"] = df["black_incident_variance"] + df["white_incident_variance"]
    df["user_variance"] = df["black_users_variance"] + df["white_users_variance"]
    df["enf_var"] = np.exp((1 / ((df.incidents ** 2) * df.incident_variance)) + (1 / ((df.users ** 2) * df.user_variance)))
    return df

def box_plot(df: pd.DataFrame, colname: str, name: str, drugname: str):
    """Plot the correlation between overdose rates and enforcement rates."""
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 5), ncols=1, nrows=2, sharex=True)

    sns.boxplot(x="year", y="log_enforcement_rate", data=df, linewidth=2.5, showfliers=False, whis=0.5, ax=ax1)
    sns.swarmplot(x="year", y="log_enforcement_rate", data=df, color=".25", alpha=0.7, ax=ax1, linewidth=0, size=1)

    sns.boxplot(x="year", y="log_selection_ratio", data=df, linewidth=2.5, showfliers=False, whis=0.5, ax=ax2)
    sns.swarmplot(x="year", y="log_selection_ratio", data=df, color=".25", alpha=0.7, ax=ax2, linewidth=0, size=1)

    ax1.set_ylabel("Log Enforcement Rate")
    ax2.set_ylabel("Log Enforcement Ratio")

    ax1.set_xlabel("")
    ax2.set_xlabel("Year")

    plt.savefig(Path(__file__).parents[3] / "plots" / f"{drugname}_over_time.pdf")
    plt.clf()

def regression(df: pd.DataFrame, target_col: str, target_var_col: str):
    from patsy import dmatrices
    import statsmodels.api as sm

    y, X = dmatrices(f"{target_col} ~ year", data=df, return_type="dataframe")
    model = sm.WLS(
        y,
        X,
        weights= 1 / df[target_var_col],
    )
    model_res = model.fit()
    model_res = model_res.get_robustcov_results(cov_type="HC1")
    coef = model_res.params[1]
    pvalue = model_res.pvalues[1]
    std_error = model_res.HC1_se[1]
    result = f"{coef:.6f} ({std_error:.3f})"
    if pvalue <= 0.05:
        result += "*"
    if pvalue <= 0.01:
        result += "*"
    if pvalue <= 0.001:
        result += "*"
    return result

def split_race_line_plot_all(temporal_smoothing: bool):
    df = load_data("_all", smoothing=temporal_smoothing)
    df1 = df.melt(id_vars=["FIPS", "year"], value_vars=["black_enforcement_rate", "white_enforcement_rate"], var_name="race", value_name="enforcement_rate")
    df2 = df.melt(id_vars=["FIPS", "year"], value_vars=["black_enforcement_rate_er", "white_enforcement_rate_er"], var_name="race", value_name="enforcement_rate_er")
    df1["race"] = df1.race.apply(lambda x: x.split("_")[0])
    df2["race"] = df2.race.apply(lambda x: x.split("_")[0])
    df = df1.merge(df2, on=["FIPS", "year", "race"])
    df["enforcement_rate_plus_error"] = list(zip(df.enforcement_rate, df.enforcement_rate_er))
    df["enforcement_rate"] = df.enforcement_rate / df.enforcement_rate_er
    sns.lineplot(
        data=df, x="year", y="enforcement_rate", hue="race", style="race",
        markers=True, dashes=False, err_style="band", palette="tab10", linewidth=1,
    )
    
    plt.ylabel("Enforcement Rate")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(Path(__file__).parents[3] / "plots" / f"split_race_drugs_line_plot{'_smoothed' if temporal_smoothing else ''}.pdf")
    plt.clf()

def split_race_line_plot(temporal_smoothing: bool):
    cannabis_df = load_data("", smoothing=temporal_smoothing)
    crack_df = load_data("_crack", smoothing=temporal_smoothing)
    heroin_df = load_data("_heroin", smoothing=temporal_smoothing)
    meth_df = load_data("_meth", smoothing=temporal_smoothing)
    cocaine_df = load_data("_cocaine", smoothing=temporal_smoothing)
    cannabis_df["drug"] = "cannabis"
    crack_df["drug"] = "crack"
    heroin_df["drug"] = "heroin"
    meth_df["drug"] = "meth"
    cocaine_df["drug"] = "cocaine"
    df = pd.concat([cannabis_df, crack_df, heroin_df, cocaine_df, meth_df])
    df = df.reset_index(drop=True)
    df1 = df.melt(id_vars=["FIPS", "year", "drug"], value_vars=["black_enforcement_rate", "white_enforcement_rate"], var_name="race", value_name="enforcement_rate")
    df2 = df.melt(id_vars=["FIPS", "year", "drug"], value_vars=["black_enforcement_rate_er", "white_enforcement_rate_er"], var_name="race", value_name="enforcement_rate_er")
    df1["race"] = df1.race.apply(lambda x: x.split("_")[0])
    df2["race"] = df2.race.apply(lambda x: x.split("_")[0])
    df = df1.merge(df2, on=["FIPS", "year", "race", "drug"])
    df["enforcement_rate"] = df.enforcement_rate * (df.enforcement_rate_er / df.enforcement_rate_er.sum())
    sns.relplot(
        kind="line", data=df, x="year", y="enforcement_rate", hue="race", style="race", col="drug", col_wrap=2,
        markers=True, dashes=False, err_style="band", palette="tab10", linewidth=1, estimator=sum, facet_kws={'sharey': False}
    )
    plt.ylabel("Enforcement Rate")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(Path(__file__).parents[3] / "plots" / f"multi_drug_line_plot{'_smoothed' if temporal_smoothing else ''}.pdf")
    plt.clf()


def all_drugs_line_plot(temporal_smoothing: bool):
    """Line plot with shared y axis"""
    cannabis_df = load_data("", smoothing=temporal_smoothing)
    crack_df = load_data("_crack", smoothing=temporal_smoothing)
    heroin_df = load_data("_heroin", smoothing=temporal_smoothing)
    meth_df = load_data("_meth", smoothing=temporal_smoothing)
    cocaine_df = load_data("_cocaine", smoothing=temporal_smoothing)
    cannabis_df["drug"] = "cannabis"
    crack_df["drug"] = "crack"
    heroin_df["drug"] = "heroin"
    meth_df["drug"] = "meth"
    cocaine_df["drug"] = "cocaine"
    # use concat rather than append
    df = pd.concat([cannabis_df, crack_df, heroin_df, meth_df, cocaine_df])
    df = df.reset_index(drop=True)

    # two subplots on seperate plots
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    #style plot
    # sns.set_style("whitegrid")
    # sns.set_context("paper")


    # sns.lineplot(
    #     data=df,
    #     x="year", y="log_selection_ratio", hue="drug", style="drug", ax=ax,
    #     markers=True, dashes=False, err_style="band", palette="tab10", linewidth=1
    # )
    sns.lineplot(
        data=df,
        x="year", y="enforcement_rate", hue="drug", style="drug", ax=ax2,
        markers=True, dashes=False, err_style="band", palette="tab10", linewidth=1
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    # wide legend, fancy
    fig.legend(handles, labels, loc='lower center', ncol=5, fancybox=True, shadow=True)

    #turn off ax legend
    ax.legend_.remove()
    ax2.legend_.remove()


    ax.set_ylabel("Log (Racial) Enforcemenet Ratio")
    ax2.set_ylabel("Enforcement rate")
    plt.tight_layout()
    # make space for title
    plt.subplots_adjust(top=0.9, bottom=0.18)
    fig.suptitle(f"Drug Log (Racial) Enforcement Ratio and Enforcement Rates Over Time{', Temporally smoothed.' if temporal_smoothing else ''}")
    plt.savefig(Path(__file__).parents[3] / "plots" / f"all_drugs_line_plot{'_smoothed' if temporal_smoothing else ''}.pdf")
    plt.clf()



def run_drug(drug_fn: str, drug_name: str):
    df = load_data(drug_fn)
    box_plot(df, "enforcement_rate", "Enforcement Rate", drug_name)
    box_plot(df, "log_selection_ratio", "Log Selection Ratio", drug_name)
    sr = regression(df, "log_selection_ratio", "var_log")
    er = regression(df, "enforcement_rate", "enf_var")
    return {"Racial ER": sr, "ER": er}

def main():
    results = []
    results.append({"Drug": "All", **run_drug("_all", "all")})
    results.append({"Drug": "Cannabis", **run_drug("", "cannabis")})
    results.append({"Drug": "Cocaine", **run_drug("_cocaine", "cocaine")})
    results.append({"Drug": "Crack", **run_drug("_crack", "crack")})
    results.append({"Drug": "Heroin", **run_drug("_heroin", "heroin")})
    results.append({"Drug": "Meth", **run_drug("_meth", "meth")})
    results = pd.DataFrame(results)
    print(results.to_latex(index=False))


if __name__ == "__main__":
    # main()
    # split_race_line_plot(temporal_smoothing=False)
    split_race_line_plot(temporal_smoothing=True)
    # all_drugs = load_data("_all", smoothing=False)
    # box_plot(all_drugs, "enforcement_rate", "Enforcement Rate", "all")
    # box_plot(all_drugs, "log_selection_ratio", "Log Selection Ratio", "all")


