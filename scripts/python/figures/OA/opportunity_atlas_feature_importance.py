# %%
from pathlib import Path
from typing import List, Tuple
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices

names = ["dem_only", "poverty", "metro", "buying", "buying_outside", "arrests"]
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

def get_model_data(df: pd.DataFrame, model: str, bw_ratio: bool = False, drop_var: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = df.loc[:, f"selection_ratio_log_{model}"]
    if not drop_var:
        var = df.loc[:, f"var_log_{model}"]
    drop_cols = [col for col in df.columns if col.startswith("var_log") or col.startswith("selection_ratio_log")] + ["FIPS"]
    if not bw_ratio:
        drop_cols += [col for col in df.columns if col.endswith("bw_ratio")]
    else:
        drop_cols += [col for col in df.columns if not col.endswith("bw_ratio")]
    df = df.drop(columns=drop_cols)
    drop_index = df[df.isnull().any(axis=1)].index
    df = df.drop(index=drop_index)
    y = y.drop(index=drop_index)
    if not drop_var:
        var = var.drop(index=drop_index)
    return df.values, y.values, df.columns, pd.concat([df, y, var], axis=1)


def forest_importance(X: np.ndarray, y: np.ndarray, seed: int, model: str, columns: List[str], jobs: int) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    forest = RandomForestRegressor(random_state=seed)
    forest.fit(X_train, y_train)
    print(f"--------- {model} ----------")
    print(metrics.r2_score(y_test, forest.predict(X_test)))
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=jobs)
    importance_mean =  pd.Series(result.importances_mean, index=columns, name="mean")
    importance_std = pd.Series(result.importances_std, index=columns, name="std")
    return pd.concat([importance_mean, importance_std], axis=1)

def lm_coefs(df: pd.DataFrame) -> pd.DataFrame:
    target_col = [col for col in df.columns if col.startswith("selection_ratio_log")][0]
    var_col = [col for col in df.columns if col.startswith("var_log")][0]
    other_cols = [col for col in df.columns if not col.startswith("selection_ratio_log") and not col.startswith("var_log")]
    y, X = dmatrices(f"{target_col} ~ {' + '.join(other_cols)}", data=df, return_type="dataframe")
    X = sm.add_constant(X)
    model = sm.WLS(
        y,
        X,
        weights=1 / df[var_col],
    )
    model_res = model.fit()
    model_res = model_res.get_robustcov_results(cov_type="HC1")
    names = [x[0] for x in model_res.summary().tables[1].data if x[0] != ""]
    results = {}
    for coef, pval, std_err, name in zip(model_res.params, model_res.pvalues, model_res.HC1_se, names):
        if name == "Intercept":
            continue
        result = f"{coef:.3f} ({std_err:.3f})"
        if pval <= 0.05:
            result += "*"
        if pval <= 0.01:
            result += "*"
        if pval <= 0.001:
            result += "*"
        if name in name_conv:
            results[name_conv[name]] = result
    return results


def plot_feature_importance(data_path: Path, models: List[str], bw_ratio: bool = False):
    df = pd.read_csv(data_path, index_col=0)
    output_df = pd.DataFrame()
    for model in models:
        X,  y, cols, df = get_model_data(df=df, model=model, bw_ratio=False)
        importance_df = forest_importance(X=X, y=y, seed=0, model=model, columns=cols, jobs=1)
        importance_df["model"] = model
        output_df = output_df.append(importance_df)

    output_df = output_df.reset_index()
    output_df = output_df.rename(columns={"index": "feature"})
    output_df["feature"] = output_df["feature"].map(column_map)

    def errplot(x, y, yerr, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        ax = data.plot(x=x, y=y, xerr=yerr, ax=ax, kind="barh", facecolor="green", edgecolor="black", width=1, **kwargs)
        ax.set_ylabel("Mean Feature Importance")
        ax.set_xlabel("Feature")
    
    g = sns.FacetGrid(output_df, col="model", col_wrap=2, sharex=False, sharey=False)
    g.map_dataframe(errplot, "feature", "mean", "std")
    g.set_axis_labels("Feature", "Mean Feature Importance")
    plt.show()


def lm_results(data_path: Path, models: List[str]):
    df = pd.read_csv(data_path, index_col=0)
    results = {}
    for model in models:
        X,  y, cols, rdf = get_model_data(df=df, model=model, bw_ratio=True, drop_var=False)
        results[model_conv[model]] = lm_coefs(rdf)
    results = pd.DataFrame(results)
    overlap = [n for n in correlate_names if n in results.index]
    results = results.loc[overlap]
    results = results.reindex(model_names, axis=1)
    return results

        


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "correlates" / "OA_processed.csv"
    # plot_feature_importance(data_path, ["poverty", "buying_outside", "buying", "arrests", "dem_only", "metro"], True)
    res = lm_results(data_path, ["poverty", "buying_outside", "buying", "arrests", "dem_only", "metro"])
    print(res.to_latex())
    # res.to_csv(Path(__file__).parents[3] / "data" / "correlates" / "lm_results.csv")
# %%
