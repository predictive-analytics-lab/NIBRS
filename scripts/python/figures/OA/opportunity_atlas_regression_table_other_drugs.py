from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[4] / "data" / "correlates"

correlates = [
    "bwratio",
    "income_county_ratio",
    "income_bw_ratio",
    "incarceration_county_ratio",
    "incarceration_bw_ratio",
    "perc_republican_votes",
    "density_county_ratio",
    "hsgrad_county_ratio",
    "hsgrad_bw_ratio",
    "collegegrad_county_ratio",
    "collegegrad_bw_ratio",
    "employment_county_ratio",
    "employment_bw_ratio",
    "birthrate_county_ratio",
    "birthrate_bw_ratio",
    "census_county_ratio",
]

correlate_titles = [
    "B/W population ratio",
    "Income",
    "Income B/W ratio",
    "Incarceration",
    "Incarceration B/W ratio",
    "% Republican Vote Share",
    "Population density",
    "High school graduation rate",
    "High school graduation rate B/W ratio",
    "College graduation rate",
    "College graduation rate B/W ratio",
    "Employment rate at 35",
    "Employment rate at 35 B/W ratio",
    "Teenage birth rate",
    "Teenage birth rate B/W ratio",
    "Census Response rate",
]


correlate_conv = {c: t for c, t in zip(correlates, correlate_titles)}

def _load_and_munge(file: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    col_to_rename = list(df.columns)[-1]
    df = df.rename(columns={col_to_rename: name})
    df["FIPS"] = df["cty"].apply(lambda x: x[3:])
    df = df.drop(columns=["Name", "cty"])
    return df


def parse_dfs(directory: Path) -> List[pd.DataFrame]:
    combined_df = pd.DataFrame()
    for file in directory.rglob(f"*_all.csv"):
        name = file.name.split("_")[0]
        target = f"{name}_county_ratio"
        df = _load_and_munge(file, name)
        df[target] = df[name] / df[name].mean()
        df[target] = df[target].rank(pct=True).astype(float)

        if (directory / f"{name}_black.csv").exists():
            black_df = _load_and_munge(directory / f"{name}_black.csv", f"black_{name}")
            white_df = _load_and_munge(directory / f"{name}_white.csv", f"white_{name}")
            df = pd.merge(df, black_df, on="FIPS", how="left")
            df = pd.merge(df, white_df, on="FIPS", how="left")
            bw_target = f"{name}_bw_ratio"
            df[bw_target] = df[f"black_{name}"] / df[f"white_{name}"]
            df[bw_target] = df[bw_target].rank(pct=True).astype(float)
        df = df.drop(
            columns=list(
                {name, f"black_{name}", f"white_{name}"}.intersection(set(df.columns))
            )
        )
        if len(combined_df) == 0:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on="FIPS", how="left")
    return combined_df

def calculate_er(df: pd.DataFrame) -> pd.DataFrame:
    df["enforcement_rate_black"] = df["black_incidents"] / df["black_users"]
    df["enforcement_rate_white"] = df["white_incidents"] / df["white_users"]
    df["enforcement_rate_black_error"] = df.enforcement_rate_black * np.sqrt((df["black_incident_variance"] / df["black_incidents"]) ** 2 + (df["black_users_variance"] / df["black_users"]) ** 2)
    df["enforcement_rate_white_error"] = df.enforcement_rate_white * np.sqrt((df["white_incident_variance"] / df["white_incidents"]) ** 2 + (df["white_users_variance"] / df["white_users"]) ** 2)
    return df

def load_and_merge_data(data_path: Path, drug_dict: dict, year_spec: str = "2017-2019_grouped") -> Tuple[pd.DataFrame, list]:
    df = parse_dfs(data_path.parent / "correlates")
    correlates = [*df.columns, "perc_republican_votes"]
    correlates.remove("FIPS")
    def _load_drug(drug: str, drug_fn: str) -> pd.DataFrame:
        drug_df = pd.read_csv(
            data_path / f"selection_ratio_county_{year_spec}_bootstraps_1000{'_' + drug_fn if len(drug_fn) > 0 else drug_fn}.csv",
            dtype={"FIPS": str},
            usecols=[
                "FIPS", 
                "selection_ratio", 
                "var_log", "bwratio", 
                "urban_code", "black_incidents", 
                "black_users", "white_incidents", 
                "white_users", "black_incident_variance", 
                "black_users_variance", "white_incident_variance", 
                "white_users_variance", "bwratio", "black", "white"
                ],
            )
        drug_df["drug"] = drug
        drug_df = calculate_er(drug_df)
        drug_df["log_selection_ratio"] = np.log(drug_df["selection_ratio"])
        drug_df["log_enforcement_rate_black"] = np.log(drug_df["enforcement_rate_black"])
        drug_df["log_enforcement_rate_white"] = np.log(drug_df["enforcement_rate_white"])
        drug_df["log_selection_ratio_error"] = np.abs(drug_df.log_selection_ratio / drug_df.var_log)
        drug_df.rename(columns={"black_users": "black_uses", "white_users": "white_uses", "black": "black_population", "white": "white_population"}, inplace=True)
        return drug_df[["FIPS", "log_enforcement_rate_black", "enforcement_rate_black_error", "enforcement_rate_white_error",  "log_enforcement_rate_white", "drug", "black_incidents", "white_incidents", "bwratio", "black_uses", "white_uses", "black_population", "white_population", "urban_code"]]
    drug_df = pd.concat([_load_drug(drug, drug_dict[drug]) for drug in drug_dict], axis=0)
    election = pd.read_csv(
        data_path.parent / "misc" / "election_results_x_county.csv",
        dtype={"FIPS": str},
        usecols=["year", "FIPS", "perc_republican_votes"],
    )
    election = election[election.year == 2020]
    election = election.drop(columns={"year"})
    df = df.merge(election, on="FIPS", how="left")
    df = pd.merge(drug_df, df, on="FIPS", how="left")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def calculate_results(drug_df: pd.DataFrame, correlates: List[str], target_col: str, target_variance_col: str) -> pd.DataFrame:
    results = []
    for name in drug_df.drug.unique():
        df = drug_df[drug_df.drug == name]
        for col in correlates:
            df_copy = df.copy()
            df_copy = df_copy[~df_copy[col].isnull()]
            df_copy = df_copy[~df_copy[target_col].isnull()]
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
            results += [[name, col, result]]
    result_df = pd.DataFrame(results, columns=["model", "correlate", "coef (p-value)"])
    pivot_df = result_df.pivot(index="correlate", columns="model", values="coef (p-value)")
    pivot_df = pivot_df.loc[correlates]
    pivot_df = pivot_df.rename(correlate_conv, axis=0)
    return pivot_df


def main():
    drug_dict = {
        "cannabis": "",
        "cocaine": "cocaine",
        "crack": "crack",
        "heroin": "heroin",
        "meth": "meth",
    }
    drug_df = load_and_merge_data(data_path.parent / "output", drug_dict)
    drug_df.to_csv(data_path.parent / "correlates" / "OA_FIPS.csv", index=False)
    # selection_ratio_results = calculate_results(drug_df, correlates, "log_selection_ratio", "log_selection_ratio_error")
    enforcement_rate_black_results = calculate_results(drug_df, correlates, "log_enforcement_rate_black", "enforcement_rate_black_error")
    enforcement_rate_white_results = calculate_results(drug_df, correlates, "log_enforcement_rate_white", "enforcement_rate_white_error")
    enforcement_rate_black_results.to_csv(data_path.parent / "correlates" / "OA_FIPS_black.csv")
    enforcement_rate_white_results.to_csv(data_path.parent / "correlates" / "OA_FIPS_white.csv")

if __name__ == "__main__":
    main()