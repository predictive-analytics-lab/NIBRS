from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[4] / "data" / "correlates"


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


df = parse_dfs(data_path)

poverty_using = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log", "bwratio", "urban_code"],
)

poverty_buying = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

poverty_buying_outside = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_buying_outside.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

dem_only_using = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

metro_using = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_metro.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

arrests = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty_arrests.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)

arrests["selection_ratio_log_arrests"] = np.log(arrests["selection_ratio"])
arrests["var_log_arrests"] = arrests["var_log"]

poverty_using["selection_ratio_log_poverty"] = np.log(poverty_using["selection_ratio"])
poverty_using["var_log_poverty"] = poverty_using["var_log"]


poverty_buying["selection_ratio_log_buying"] = np.log(poverty_buying["selection_ratio"])
poverty_buying["var_log_buying"] = poverty_buying["var_log"]


poverty_buying_outside["selection_ratio_log_buying_outside"] = np.log(
    poverty_buying_outside["selection_ratio"]
)
poverty_buying_outside["var_log_buying_outside"] = poverty_buying_outside["var_log"]

dem_only_using["selection_ratio_log_dem_only"] = np.log(
    dem_only_using["selection_ratio"]
)
dem_only_using["var_log_dem_only"] = dem_only_using["var_log"]

metro_using["selection_ratio_log_metro"] = np.log(metro_using["selection_ratio"])
metro_using["var_log_metro"] = metro_using["var_log"]


poverty_using = poverty_using.drop(columns=["selection_ratio", "var_log"])
poverty_buying = poverty_buying.drop(columns=["selection_ratio", "var_log"])
arrests = arrests.drop(columns=["selection_ratio", "var_log"])

poverty_buying_outside = poverty_buying_outside.drop(
    columns=["selection_ratio", "var_log"]
)
dem_only_using = dem_only_using.drop(columns=["selection_ratio", "var_log"])
metro_using = metro_using.drop(columns=["selection_ratio", "var_log"])

df = df.merge(poverty_using, on="FIPS")
df = df.merge(poverty_buying, on="FIPS")
df = df.merge(poverty_buying_outside, on="FIPS")
df = df.merge(dem_only_using, on="FIPS")
df = df.merge(metro_using, on="FIPS")
df = df.merge(arrests, on="FIPS")

election = pd.read_csv(
    data_path.parent / "misc" / "election_results_x_county.csv",
    dtype={"FIPS": str},
    usecols=["year", "FIPS", "perc_republican_votes"],
)

election = election[election.year == 2020]

election = election.drop(columns={"year"})

df = df.merge(election, on="FIPS", how="left")

dfs = [
    dem_only_using,
    poverty_using,
    metro_using,
    poverty_buying,
    poverty_buying_outside,
]
names = ["dem_only", "poverty", "metro", "buying", "buying_outside", "arrests"]

results = []

df = df.replace([np.inf, -np.inf], np.nan)

correlates = df.columns[1:]
correlates = [c for c in correlates if not c.startswith("selection_ratio")]
correlates = [c for c in correlates if not c.startswith("var_log")]

for name in names:
    for col in correlates:
        tfi = df.copy()
        tfi = tfi[~tfi[col].isnull()]
        y, X = dmatrices(
            f"selection_ratio_log_{name} ~ {col}", data=tfi, return_type="dataframe"
        )
        model = sm.WLS(
            y,
            X,
            weights=1 / tfi[f"var_log_{name}"],
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

correlate_order = [
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

correlate_names = [
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

model_names = [
    "Dem only",
    "Dem + Pov",
    "Dem + metro",
    "Buying",
    "Buying Outside",
    "Arrests",
]


name_conv = {k: v for k, v in zip(correlate_order, correlate_names)}
model_conv = {k: v for k, v in zip(names, model_names)}


pivot_df = result_df.pivot(index="correlate", columns="model", values="coef (p-value)")
pivot_df = pivot_df.reindex(names, axis=1)
pivot_df = pivot_df.loc[correlate_order]

pivot_df = pivot_df.rename(name_conv, axis=0)
pivot_df = pivot_df.rename(model_conv, axis=1)

print(pivot_df.to_latex())

pivot_df.to_csv(data_path / "OA_coef_pivot.csv")
df.to_csv(data_path / "OA_processed.csv")
