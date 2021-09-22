# %%
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path(__file__).parents[3] / "data" / "correlates"


def _load_and_munge(file: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    col_to_rename = list(df.columns)[-1]
    df = df.rename(columns={col_to_rename: name})
    df["FIPS"] = df["cty"].apply(lambda x: x[3:])
    df = df.drop(columns=["Name", "cty"])
    return df


def parse_dfs(directory: Path, logx: List[str] = []) -> List[pd.DataFrame]:
    combined_df = pd.DataFrame()
    for file in directory.rglob(f"*_all.csv"):
        name = file.name.split("_")[0]
        target = f"{name}_county_ratio"
        df = _load_and_munge(file, name)
        df[target] = df[name] / df[name].mean()
        if target in logx:
            df[target] = np.log10(df[target])
        if (directory / f"{name}_black.csv").exists():
            black_df = _load_and_munge(directory / f"{name}_black.csv", f"black_{name}")
            white_df = _load_and_munge(directory / f"{name}_white.csv", f"white_{name}")
            df = pd.merge(df, black_df, on="FIPS", how="left")
            df = pd.merge(df, white_df, on="FIPS", how="left")
            bw_target = f"{name}_bw_ratio"
            df[bw_target] = df[f"black_{name}"] / df[f"white_{name}"]
            if bw_target in logx:
                df[bw_target] = np.log10(df[bw_target])
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


# %%
df = parse_dfs(data_path)

poverty_using = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
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

urban_using = pd.read_csv(
    data_path.parent
    / "output"
    / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_urban.csv",
    dtype={"FIPS": str},
    usecols=["FIPS", "selection_ratio", "var_log"],
)


poverty_using["selection_ratio_log10_poverty"] = np.log10(
    poverty_using["selection_ratio"]
)
poverty_using["var_log_poverty"] = poverty_using["var_log"]


poverty_buying["selection_ratio_log10_buying"] = np.log10(
    poverty_buying["selection_ratio"]
)
poverty_buying["var_log_buying"] = poverty_buying["var_log"]


poverty_buying_outside["selection_ratio_log10_buying_outside"] = np.log10(
    poverty_buying_outside["selection_ratio"]
)
poverty_buying_outside["var_log_buying_outside"] = poverty_buying_outside["var_log"]

dem_only_using["selection_ratio_log10_dem_only"] = np.log10(
    dem_only_using["selection_ratio"]
)
dem_only_using["var_log_dem_only"] = dem_only_using["var_log"]

urban_using["selection_ratio_log10_urban"] = np.log10(urban_using["selection_ratio"])
urban_using["var_log_urban"] = urban_using["var_log"]


poverty_using = poverty_using.drop(columns=["selection_ratio", "var_log"])
poverty_buying = poverty_buying.drop(columns=["selection_ratio", "var_log"])
poverty_buying_outside = poverty_buying_outside.drop(
    columns=["selection_ratio", "var_log"]
)
dem_only_using = dem_only_using.drop(columns=["selection_ratio", "var_log"])
urban_using = urban_using.drop(columns=["selection_ratio", "var_log"])

df = df.merge(poverty_using, on="FIPS")
df = df.merge(poverty_buying, on="FIPS")
df = df.merge(poverty_buying_outside, on="FIPS")
df = df.merge(dem_only_using, on="FIPS")
df = df.merge(urban_using, on="FIPS")

election = pd.read_csv(
    data_path.parent / "misc" / "election_results_x_county.csv",
    dtype={"FIPS": str},
    usecols=["year", "FIPS", "perc_republican_votes"],
)

election = election[election.year == 2020]

election = election.drop(columns={"year"})

df = df.merge(election, on="FIPS", how="left")

# %%
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
from matplotlib import pyplot as plt


dfs = [
    dem_only_using,
    poverty_using,
    urban_using,
    poverty_buying,
    poverty_buying_outside,
]
names = ["dem_only", "poverty", "urban", "buying", "buying_outside"]

results = []

df = df.replace([np.inf, -np.inf], np.nan)

correlates = df.columns[1:]
correlates = [c for c in correlates if not c.startswith("selection_ratio")]
correlates = [c for c in correlates if not c.startswith("var_log")]

for name in names:
    for col in correlates:
        tfi = df.copy()
        tfi = tfi[~tfi[col].isnull()]
        y, X = dmatrices(f"var_log_{name} ~ {col}", data=tfi, return_type="dataframe")
        model = sm.WLS(
            y,
            X,
            weights=1 / tfi[f"var_log_{name}"],
        )
        model_res = model.fit()
        coef = model_res.params[0]
        pvalue = model_res.pvalues[0]
        # g = sns.jointplot(
        #     data=tfi,
        #     x=col,
        #     y=f"selection_ratio_log10_{name}",
        #     kind="reg",
        # )
        # plt.text(x=0.5, y=2.2, s=f"slope: {coef:.3f}\n ci: {pvalue:.3f}")
        # g.ax_joint.set_ylabel("log10(selection_ratio)")
        # g.ax_joint.set_xlabel(col)
        # plt.title(f"{name} {col}")
        # plt.show()
        result = f"{coef:.3f} ({pvalue:.6f})"
        results += [[name, col, result]]

result_df = pd.DataFrame(results, columns=["model", "correlate", "coef (p-value)"])
# %%

pivot_df = result_df.pivot(index="correlate", columns="model", values="coef (p-value)")
pivot_df = pivot_df.reindex(names, axis=1)
# %%

df.to_csv(data_path / "processed_correlates.csv")

# %%
