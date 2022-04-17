"""Test for correlations between lemas and incident rates."""
# %%
from email.mime import base
from fileinput import filename
from re import X
from this import d
from typing import List, Optional, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


base_path = Path(__file__).parents[3] / "data"


def wilson_error(n_s: int, n: int, z=1.96):
    """
    Wilson score interval

    param n_s: number of successes
    param n: total number of events
    param z: The z-value

    return: The lower and upper bound of the Wilson score interval
    """
    n_f = np.max([1, n - n_s])
    denom = n + z ** 2
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2 / 4))
    return adjusted_p, ci


def load_nibrs_data() -> pd.DataFrame:
    """Load NIBRS data."""
    def _load_df(drug: str):
        if drug == "other_incidents":
            df = pd.read_csv(base_path / "output" / "other_incidents_2010-2020.csv", dtype={"FIPS": str}, usecols=[
                "FIPS", "year", "black_incidents", "white_incidents", "black_population", "white_population"])
        else:
            df = pd.read_csv(base_path / "output" / f"selection_ratio_county_2010-2020_wilson{'_' + drug if drug != 'cannabis' else ''}.csv", dtype={"FIPS": str}, usecols=[
                "FIPS", "year", "black_incidents", "white_incidents", "black_population", "white_population"])
        df["black_incidents_p100k"] = (
            df.black_incidents / df.black_population) * 100_000
        df["white_incidents_p100k"] = (
            df.white_incidents / df.white_population) * 100_000
        df["black_error"] = df.apply(
            lambda x: wilson_error(x.black_incidents, x.black_population)[1], axis=1) * 100_000
        df["white_error"] = df.apply(
            lambda x: wilson_error(x.white_incidents, x.white_population)[1], axis=1) * 100_000
        df["drug"] = drug
        df = df[df.year == 2016]
        return df[["FIPS", "drug", "black_incidents_p100k", "white_incidents_p100k", "black_error", "white_error"]]
    return pd.concat([_load_df(drug) for drug in ["cannabis", "cocaine", "heroin", "crack", "meth", "other_incidents", "dui", "drunkeness"]])


def load_OA() -> pd.DataFrame:
    df = pd.read_csv(base_path / "correlates" / "OA_processed.csv", dtype={"FIPS": str}, usecols=[
                     "FIPS", "perc_republican_votes", "density_county_ratio", "bwratio", "income_county_ratio", "income_bw_ratio"])
    df["bwratio"] = df["bwratio"].rank(pct=True).astype(float)
    return df


def load_lemas_data() -> pd.DataFrame:
    """Load Lemas Data."""
    df = pd.read_csv(base_path / "agency" /
                     "lemas_fips.csv", dtype={"FIPS": str})
    df = df.rename(columns={k: k.replace(
        " ", "_").replace("-", "_") for k in df.columns})
    return df


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load data."""
    lemas = load_lemas_data()
    nibrs = load_nibrs_data()
    census = load_census_data()
    oa = load_OA()
    return nibrs.merge(lemas, on="FIPS").merge(census, on="FIPS").merge(oa, on="FIPS"), list(set(lemas.columns) - {"FIPS", "incidents", "Unnamed:_1", "population_covered_proportion", "bwratio", "population_covered", "selection ratio std"})


def load_census_data() -> pd.DataFrame:
    df = pd.read_csv(base_path / "census" / "census-2020-county.csv", dtype={"STATE": str, "COUNTY": str, "YEAR": str},
                     engine="python", encoding="ISO-8859-1", usecols=["YEAR", "STATE", "COUNTY", "BA_MALE", "BA_FEMALE", "WA_MALE", "WA_FEMALE"])
    df["FIPS"] = df.STATE + df.COUNTY
    df = df[df.YEAR == "6"]

    def convert_to_int(x):
        try:
            return int(x)
        except:
            return 0
    df["WA_MALE"] = df.WA_MALE.apply(convert_to_int)
    df["WA_FEMALE"] = df.WA_FEMALE.apply(convert_to_int)
    df["BA_FEMALE"] = df.BA_FEMALE.apply(convert_to_int)
    df["BA_MALE"] = df.BA_MALE.apply(convert_to_int)

    df = df.groupby("FIPS").agg(
        {"WA_MALE": "sum", "WA_FEMALE": "sum", "BA_MALE": "sum", "BA_FEMALE": "sum"}).reset_index()
    df["white_population"] = df.WA_MALE + df.WA_FEMALE
    df["black_population"] = df.BA_MALE + df.BA_FEMALE
    return df[["FIPS", "white_population", "black_population"]]


def calculate_results(drug_df: pd.DataFrame, correlates: List[str], target_col: str, target_variance_col: str, signif_record: float = 0.001, missing_threshold: float = 1) -> pd.DataFrame:
    results = []
    signif_vars = []
    for name in drug_df.drug.unique():
        df = drug_df[drug_df.drug == name]
        for col in correlates:
            if "__missing" in col:
                continue
            df_copy = df.copy()
            df_copy = df_copy[~df_copy[col].isnull()]
            df_copy = df_copy[~df_copy[target_col].isnull()]
            if "__" in col:
                base_col = col.split("__")[0]
                if missing_threshold <= 0:
                    df_copy = df_copy[df_copy[f"{base_col}__missing"]
                                      <= missing_threshold]
                else:
                    df_copy = df_copy[df_copy[f"{base_col}__missing"]
                                      < missing_threshold]
            if len(df_copy) <= 5:
                continue
            y, X = dmatrices(
                f"{target_col} ~ {col}", data=df_copy, return_type="dataframe"
            )
            model = sm.WLS(
                y,
                X,
                weights=1 / df_copy[target_variance_col],
            )
            model_res = model.fit()
            model_res = model_res.get_robustcov_results(cov_type="HC1")
            coef = model_res.params[1]
            pvalue = model_res.pvalues[1]
            std_error = model_res.HC1_se[1]
            result = f"{coef:.3f} ({std_error:.3f})"
            if pvalue <= 0.05:
                result += "*"
            if pvalue <= 0.01:
                result += "*"
            if pvalue <= 0.001:
                result += "*"
            if pvalue <= signif_record:
                signif_vars.append(col)
            results += [[name, col, result]]
    result_df = pd.DataFrame(
        results, columns=["model", "variable", "coef (p-value)"])
    pivot_df = result_df.pivot(
        index="variable", columns="model", values="coef (p-value)")
    return pivot_df, list(set(signif_vars))


def boxplot_variable(df: pd.DataFrame, vars: List[str], target: str, col_wrap: int, missing_threshold: float = 1) -> None:
    """Boxplot variable."""
    sns.set(style="whitegrid")

    n_rows = len(vars) // col_wrap + 1
    n_cols = col_wrap
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for var, ax in zip(vars, axes.flatten()):
        if "__" in var:
            base_col = var.split("__")[0]
            df_copy = df.copy()
            if missing_threshold <= 0:
                df_copy = df_copy[df_copy[f"{base_col}__missing"]
                                  <= missing_threshold]
            else:
                df_copy = df_copy[df_copy[f"{base_col}__missing"]
                                  < missing_threshold]
            sns.boxplot(x="drug", y=target, hue=var, data=df_copy, ax=ax)
        else:
            ax.set_xscale("log")
            sns.scatterplot(x=var, y=target, hue="drug", data=df, ax=ax)
            ax.set_xlim(1, df[var].max())
        ax.set_title(f"{var}")
        ax.set_xlabel("")
        ax.set_ylabel(f"{target}")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(base_path.parent / "plots" /
                f"{target}_{'no_missing' if missing_threshold <= 0 else 'partial_missing'}_boxplot.png")


def agency_distribution(df: pd.DataFrame, drug: str):
    df["BLACK_OFFICERS"] = df["PERS_BLACK_MALE"] + df["PERS_BLACK_FEM"]
    df["WHITE_OFFICERS"] = df["PERS_WHITE_MALE"] + df["PERS_WHITE_FEM"]

    df = df[(df.BLACK_OFFICERS + df.WHITE_OFFICERS) > 0]

    df["BW_OFFICER_RATIO"] = df["BLACK_OFFICERS"] / df["WHITE_OFFICERS"]

    df["BLACK_OFFICER_RATE"] = df["BLACK_OFFICERS"] / df["black_population"]
    df["WHITE_OFFICER_RATE"] = df["WHITE_OFFICERS"] / df["white_population"]

    df["OFFICER_RATE_RATIO"] = df["BLACK_OFFICER_RATE"] / df["WHITE_OFFICER_RATE"]

    df.OFFICER_RATE_RATIO.replace(np.inf, np.nan, inplace=True)
    df = df[~df.OFFICER_RATE_RATIO.isnull()]

    def categorize_orr(orr: float) -> str:
        if orr > 0.75:
            return "balanced"
        elif orr > 0.25:
            return "low"
        elif orr > 0:
            return "very-low"
        else:
            return "none"

    df["Officer Rate Category"] = df["OFFICER_RATE_RATIO"].apply(
        categorize_orr)

    df = df[df.drug == drug]

    sns.set(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

    sns.kdeplot(data=df, x="black_enforcement_rate",
                hue="Officer Rate Category", shade=True, ax=ax1)
    sns.kdeplot(data=df, x="white_enforcement_rate",
                hue="Officer Rate Category", shade=True, ax=ax2)

    # get legend
    handles, labels = ax1.get_legend_handles_labels()

    # add legend to bottom of figure
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=True)

    # clear legend
    ax1.legend_.remove()
    # ax2.legend_.remove()

    ax1.set_xlabel("Black Enforcement Rate")
    ax2.set_xlabel("White Enforcement Rate")
    plt.suptitle(
        f"{drug.title()} Enforcement Split by (Pop-Weighted) Officer Race Ratio")
    plt.tight_layout()
    # add space for legend
    # plt.subplots_adjust(bottom=0.1)
    plt.savefig(base_path.parent / "plots" / f"{drug}_kde.pdf")
    plt.clf()


def bin_plot(df, var: str, ncol: int):
    df["BLACK_OFFICERS"] = df["PERS_BLACK_MALE"] + df["PERS_BLACK_FEM"]
    df["WHITE_OFFICERS"] = df["PERS_WHITE_MALE"] + df["PERS_WHITE_FEM"]
    df["BLACK_OFFICER_PERCENTAGE"] = df["BLACK_OFFICERS"] / \
        (df["BLACK_OFFICERS"] + df["WHITE_OFFICERS"])

    def _categorize(x):
        if x >= 0.76:
            return "76-100"
        elif x >= 0.51:
            return "51-75"
        elif x >= 0.26:
            return "26-50"
        elif x >= 0.01:
            return "01-25"
        else:
            return "0"
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow))
    nrow = df.drug.nunique() // ncol + 1
    for drug, ax in zip(df.drug.unique(), axes.flatten()):
        drug_df = df[df.drug == drug]
        ax.set_yscale("log")
        # bin var__yes
        drug_df["black_enforcement_rate"] = np.exp(
            drug_df["black_enforcement_rate"])
        drug_df["white_enforcement_rate"] = np.exp(
            drug_df["white_enforcement_rate"])
        drug_df = drug_df.melt(id_vars=["drug", "BLACK_OFFICER_PERCENTAGE", "FIPS"], value_vars=[
                               "black_enforcement_rate", "white_enforcement_rate"], var_name="race", value_name="enforcement_rate").reset_index()
        drug_df["race"] = drug_df.race.map(
            {"black_enforcement_rate": "black", "white_enforcement_rate": "white"})
        sns.boxplot(y="enforcement_rate", x="race", hue="BLACK_OFFICER_PERCENTAGE", data=drug_df,
                    ax=ax, palette="Set2", whis=0.25, fliersize=0, hue_order=["0-50%", "50-100%"])
        sns.stripplot(x="race", y="enforcement_rate", hue="BLACK_OFFICER_PERCENTAGE", data=drug_df, jitter=True,
                      palette="Set2", ax=ax, split=True, linewidth=1, alpha=.25, hue_order=["0-50%", "50-100%"])
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        # sns.scatterplot(x=x, y=y, data=df, **kwargs)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.125)
    fig.legend(handles[:2], labels[:2],
               title=f"{var}", loc='lower center', ncol=2, fancybox=True)
    plt.show()


def get_usage_ratio():
    df = pd.read_csv(
        Path(__file__).parents[3] / "data" / "NSDUH" / "nsduh_usage_ratio.csv")
    df = df.fillna(df.mean())
    df = df.groupby(["drug"]).rolling(
        3, on="year", center=True).sum().reset_index()
    df["usage_ratio"] = df.black / df.white
    return df[["usage_ratio", "year", "drug"]]


def plot_bool(df: pd.DataFrame, var: str, option: Optional[str] = None, bins: Optional[list[float]] = None, bin_options: Optional[list[str]] = None, missing_threshold: float = 1, normalize: bool = False, ncol: int = 2, title: Optional[str] = None, option_title: Optional[str] = None, filename: str = "plot.pdf"):
    if option:
        if missing_threshold <= 0:
            df = df[df[f"{var}__missing"] <= missing_threshold]
        else:
            df = df[df[f"{var}__missing"] < missing_threshold]
    if normalize:
        cols = [col for col in df.columns if col.startswith(
            var) and not col.endswith("__missing")]
        df["norm"] = df[cols].sum(axis=1)
        df.loc[:, cols] = df.loc[:, cols].apply(
            lambda x: x / df["norm"], axis=0)
    sns.set(style="whitegrid")
    if not bins:
        bins = [-np.inf, 0.5, np.inf]
        bin_options = ["0-50%", "50-100%"]
    nrow = df.drug.nunique() // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 5 * nrow))
    usage_ratio = get_usage_ratio()
    usage_ratio = usage_ratio[usage_ratio.year == 2016]
    for drug, ax in zip(df.drug.unique(), axes.flatten()):
        usage = usage_ratio[usage_ratio.drug == drug]
        drug_df = df[df.drug == drug]
        if len(usage) > 0:
            drug_df["expected_black_incidents_per_100k"] = drug_df.white_incidents_p100k * \
                usage.usage_ratio.values[0]
        else:
            drug_df["expected_black_incidents_per_100k"] = drug_df.white_incidents_p100k
        ax.set_yscale("log")
        # bin var__yes
        if option:
            drug_df["binned"] = pd.cut(
                drug_df[f"{var}__{option}"], bins=bins, labels=False)
        else:
            drug_df["binned"] = pd.cut(drug_df[var], bins=bins, labels=False)
        drug_df["binned"] = drug_df.binned.map(
            {i: bin_options[i] for i in range(len(bin_options))}.get)
        drug_df.sort_values(by="binned", inplace=True)
        drug_df = drug_df.melt(id_vars=["drug", "binned", "FIPS"], value_vars=[
                               "black_incidents_p100k", "white_incidents_p100k", "expected_black_incidents_per_100k"], var_name="race", value_name="incidents_p100k").reset_index()
        drug_df["race"] = drug_df.race.map(
            {"black_incidents_p100k": "black", "white_incidents_p100k": "white", "expected_black_incidents_per_100k": "expected black"})
        drug_df = drug_df[drug_df.incidents_p100k > 0]
        if len(usage) <= 0:
            drug_df = drug_df[drug_df.race.isin(["black", "white"])]
        sns.boxplot(y="incidents_p100k", hue="race", x="binned",
                    data=drug_df, ax=ax, palette="Set2", whis=0.25, fliersize=0, hue_order=["white", "black", "expected black"])
        sns.stripplot(hue="race", y="incidents_p100k", x="binned", data=drug_df,
                      jitter=True, palette="Set2", ax=ax, split=True, linewidth=1, alpha=.25, hue_order=["white", "black", "expected black"])

        # for i, bin in enumerate(drug_df.binned.unique()):
        #     bin_df = drug_df[drug_df.binned == bin]
        #     white_median = bin_df[bin_df.race ==
        #                           "white"].incidents_p100k.median()
        #     adjusted_median = bin_df[bin_df.race == "white"].incidents_p100k.median(
        #     ) / usage.usage_ratio.values[0]
        #     length = adjusted_median - white_median
        #     # ax.arrow(-0.2 + (i * 1), black_median, 0, adjusted_median -
        #     #          black_median, width=0.02, head_width=0.4, color="#212121", alpha=0.8, zorder=10, head_length=0.4, length_includes_head=True)
        #     ax.annotate("", xy=(0.2 + (i * 1), adjusted_median), xytext=(0.2 + (i * 1), white_median),
        #                 arrowprops=dict(arrowstyle="->", color="#212121", alpha=0.8, zorder=10, lw=3))
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        # ax.legend(handles[0:2], labels[0:2], title=f"{var} ({option})")

        # sns.swarmplot(y="enforcement_rate", x="binned", hue="race", data=drug_df,  alpha=1,palette="Set2", size=3, ax=ax)
        ax.set_ylabel(f"Incidents Per 100K")
        if option_title:
            ax.set_xlabel(option_title)
        else:
            if option is not None:
                ax.set_xlabel(f"% {var} = ({option}) in County")
            else:
                ax.set_xlabel(f"% {var}) in County")
        ax.set_title(drug)
    if option:
        plt.suptitle(
            f"Normalized Incident Rates for U.S. Counties with Police Agencies consisting of either 0-50% or 50-100% {var} = {option}, split by Race.")
    else:
        plt.suptitle(title)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.125)
    fig.legend(handles[:3], [x.title() for x in labels[:3]],
               title=f"Race", loc='lower center', ncol=len(bins), fancybox=True)
    plt.savefig(base_path.parent / "plots" / filename)
    plt.show()

# %%
df, correlates = load_data()

# %%
black_results, black_sig = calculate_results(
    df, correlates, "black_enforcement_rate", "black_enforcement_rate_error")
white_results, white_sig = calculate_results(
    df, correlates, "white_enforcement_rate", "white_enforcement_rate_error")

black_results.to_csv(base_path / "output" /
                     "black_lemas_results_partial_missing.csv")
white_results.to_csv(base_path / "output" /
                     "white_lemas_results_partial_missing.csv")

boxplot_variable(df, black_sig, "black_enforcement_rate", 6)
boxplot_variable(df, white_sig, "white_enforcement_rate", 6)

black_results, black_sig = calculate_results(
    df, correlates, "black_enforcement_rate", "black_enforcement_rate_error", missing_threshold=0)
white_results, white_sig = calculate_results(
    df, correlates, "white_enforcement_rate", "white_enforcement_rate_error", missing_threshold=0)

boxplot_variable(df, black_sig, "black_enforcement_rate",
                 6, missing_threshold=0)
boxplot_variable(df, white_sig, "white_enforcement_rate",
                 6, missing_threshold=0)

black_results.to_csv(base_path / "output" /
                     "black_lemas_results_no_missing.csv")
white_results.to_csv(base_path / "output" /
                     "white_lemas_results_no_missing.csv")
# %%
# plot_bool(df, "black_enforcement_rate", "CP_PLAN", "yes", missing_threshold=1, normalize=True)
df["BLACK_OFFICERS"] = df["PERS_BLACK_MALE"] + df["PERS_BLACK_FEM"]
df["WHITE_OFFICERS"] = df["PERS_WHITE_MALE"] + df["PERS_WHITE_FEM"]
df["BLACK_OFFICER_PERCENTAGE"] = df["BLACK_OFFICERS"] / \
    (df["BLACK_OFFICERS"] + df["WHITE_OFFICERS"])
# %%
df["BLACK_POP_PERCENTAGE"] = df["black_population"] / \
    (df["white_population"] + df["black_population"])
df["BLACK_REL_OFFICERS"] = df["BLACK_OFFICER_PERCENTAGE"] / \
    df["BLACK_POP_PERCENTAGE"]
plot_bool(df,
          "BLACK_REL_OFFICERS",
          bins=[-np.inf, 0.25, np.inf],
          bin_options=["Bottom 50%", "Top 50%"],
          normalize=False,
          title="Incidents per 100K population split by number of black officers relative to population and race.",
          option_title="Relative Black Officer Category",
          filename="relative_black_officers.pdf")

# %%

df["CALLS_P100K"] = (
    df["OPER_CFS"] / (df["black_population"] + df["white_population"])) * 100000
df["CALLS_P100K"] = df["CALLS_P100K"].rank(pct=True).astype(float)

plot_bool(
    df,
    "CALLS_P100K",
    bins=[-np.inf, 0.5, np.inf],
    bin_options=["Bottom 50%", "Top 50%"],
    normalize=False,
    title="Incidents Per 100K Population split by calls p100k and Race.",
    option_title="Operation Calls P/100K",
    filename="")

# %%

df["DIS_P100K"] = (
    df["OPER_DIS"] / (df["black_population"] + df["white_population"])) * 100000
df["DIS_P100K"] = df["DIS_P100K"].rank(pct=True).astype(float)

plot_bool(
    df,
    "DIS_P100K",
    bins=[-np.inf, 0.5, np.inf],
    bin_options=["Bottom 50%", "Top 50%"],
    normalize=False,
    title="Incidents Per 100K Population split by dispatches p100k and Race.",
    option_title="Operation Dispatches P/100K",
    filename="")

# %%
plot_bool(df, "TECH_TYP_FACEREC", "yes", normalize=True, missing_threshold=1)
# %%
plot_bool(
    df,
    "bwratio",
    bins=[-np.inf, 0.5, np.inf],
    bin_options=["Bottom 50%", "Top 50%"],
    normalize=False,
    title="Incidents Per 100K Population split by County Black / White Population ratio and Race.",
    option_title="Black / White Population Ratio",
    filename="")

# %%
plot_bool(
    df,
    "density_county_ratio",
    bins=[-np.inf, 0.5, np.inf],
    bin_options=["Bottom 50%", "Top 50%"],
    normalize=False,
    title="Incidents Per 100K Population split by Population Density and Race.",
    option_title="County Population Density",
    filename="county_density.pdf")

# %%
plot_bool(
    df,
    "income_county_ratio",
    bins=[-np.inf, 0.5, np.inf],
    bin_options=["Bottom 50%", "Top 50%"],
    normalize=False,
    title="Incidents Per 100K Population split by Average County Income and Race.",
    option_title="Average County Income Category",
    filename="income_ratio.pdf")

# %%
plot_bool(df, "EQ_BDYARM_NOAUTH", "no", normalize=True, missing_threshold=1)

# %%
