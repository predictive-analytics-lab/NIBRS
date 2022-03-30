from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[4] / "data" / "correlates"

relative_enforcment = pd.read_csv(data_path.parent / "output" / "relative_drug_enforcement.csv", dtype={"FIPS": str})
# right pad FIPS to 5 digits
relative_enforcment["FIPS"] = relative_enforcment["FIPS"].str.rjust(5, fillchar="0")
relative_enforcment["relative_enforcement"] = np.log(relative_enforcment["relative_enforcement"].astype(float))


df = pd.read_csv(data_path / "OA_processed_other_drugs.csv", dtype={"FIPS": str})
df = df.replace([np.inf, -np.inf], np.nan)

df = relative_enforcment.merge(df, on="FIPS", how="left")

# drop columns starting with 'selection_ratio'
df = df.drop(columns=df.filter(regex="^selection_ratio"))
df = df.drop(columns=df.filter(regex="^var_log"))
df = df.drop(columns=df.filter(regex="^Unnamed"))
df = df.drop(columns=df.filter(regex="^SR"))

correlates = df.columns[1:]
correlates = [c for c in correlates if not c.startswith("relative_enforcement") and not c.startswith("drug") and not c.startswith("FIPS")]

results = []

names = ["cannabis", "cocaine", "heroin", "crack", "meth"]


for name in names:
    for col in correlates:
        tfi = df.copy()
        tfi = tfi[~tfi[col].isnull()]
        tfi = tfi[tfi.drug == name]
        # drop drug col
        tfi = tfi.drop(columns=["drug"])
        y, X = dmatrices(
            f"relative_enforcement ~ {col}", data=tfi, return_type="dataframe"
        )
        model = sm.WLS(
            y,
            X
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
    "Cannabis",
    "Cocaine",
    "Heroin",
    "Crack",
    "Meth",
]


name_conv = {k: v for k, v in zip(correlate_order, correlate_names)}
model_conv = {k: v for k, v in zip(names, model_names)}


pivot_df = result_df.pivot(index="correlate", columns="model", values="coef (p-value)")
pivot_df = pivot_df.reindex(names, axis=1)
pivot_df = pivot_df.loc[correlate_order]

pivot_df = pivot_df.rename(name_conv, axis=0)
pivot_df = pivot_df.rename(model_conv, axis=1)

pivot_df.to_csv(data_path / "OA_coef_RE_other_drugs.csv")
df.to_csv(data_path / "OA_RE_other_drugs.csv")

print(pivot_df.to_latex())
# df = df.rename(columns={c: c.split("_")[-1] for c in df.columns if c.startswith("enforcement_ratio")})
# cors = df[names].corr()
# mask = np.zeros_like(cors)
# mask[np.triu_indices_from(mask, 1)] = True

# with sns.axes_style("white"):
#     f, ax = plt.subplots(figsize=(7, 5))
#     ax = sns.heatmap(cors, mask=mask, vmin=0, vmax=1, square=True)

# plt.savefig(data_path / "OA_correlation_heatmap_other_drugs.png")