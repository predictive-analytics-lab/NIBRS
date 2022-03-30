"""Predict number of incidents for unfilled counties."""
from typing import List, Tuple
import pandas as pd
from pathlib import Path
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectFpr, f_regression, SelectKBest
from sklearn.impute import IterativeImputer, SimpleImputer
import warnings
warnings.filterwarnings("ignore")

base_path = Path(__file__).parents[3] / "data"

def load_medicare_data() -> pd.DataFrame:
    df = pd.read_csv(base_path / "misc" / "Medicare.csv", dtype={"FIPS": str})
    df["FIPS"] = df.FIPS.str.zfill(5)
    return df


def load_ers_data() -> pd.DataFrame:
    csv_names = ["Veterans.csv", "Jobs.csv", "People.csv", "Income.csv", "County Classifications.csv"]
    data_path = base_path / "misc"
    df = pd.DataFrame()
    for csv in csv_names:
        if csv == "County Classifications.csv":
            delim = "\t"
        else:
            delim = ","
        # ignre errors
        loaded_df = pd.read_csv(data_path / csv, dtype={"FIPS": str}, delimiter=delim, engine='python')
        loaded_df = loaded_df.drop(columns=["State", "County"])
        if len(df) > 0:
            df = df.merge(loaded_df, on="FIPS")
        else:
            df = loaded_df
    return df

def load_lemas_data() -> pd.DataFrame:
    data_path = base_path / "output"
    df = pd.read_csv(data_path / "aggregated_lemas_FIPS.csv", dtype={"FIPS": str})
    df["FIPS"] = df.FIPS.str.zfill(5)
    df.drop(columns=["population_covered", "selection_ratio", "selection ratio std", "incidents"], inplace=True)
    return df

def form_data(target_col: str, omit_cols: List[str], drug: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load OA data."""
    data_path = base_path / "correlates"
    df = pd.read_csv(data_path / "OA_FIPS.csv", dtype={"FIPS": str})
    # ers = load_ers_data()
    # lemas = load_lemas_data()
    # medicare = load_medicare_data()
    # df = df.merge(ers, on="FIPS")
    # df = df.merge(lemas, on="FIPS")
    # df = df.merge(medicare, on="FIPS")
    bw_ratio_cols = [col for col in df.columns if "bw_ratio" in col]
    df = df.drop(columns=bw_ratio_cols)
    df = df[df.drug == drug]

    # convert to quantiles
    # drop null
    # df = df[~df.isnull().any(axis=1)]
    # remove padding
    y = df[target_col] - 0.5
    X = df.drop(columns=omit_cols + [target_col])
    X = X.drop(columns=[col for col in X.columns if "_incidents" in col])
    return X, y

def fit_model(X: pd.DataFrame, y: pd.Series, target: str, drug: str) -> XGBRegressor:
    """Fit model with 10 CV folds."""
    pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=0, imputation_order='random', n_nearest_features=50)),
        ("selector", SelectFpr(f_regression, alpha=0.01)),
        ("model", XGBRegressor(n_estimators=100, random_state=0)),

    ], verbose=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipeline, X, y, scoring="r2", cv=cv, verbose=0, n_jobs=1)
    print(f"{target} {drug} R2 Score: {np.mean(scores)}")
    pipeline.fit(X, y)
    return pipeline, np.mean(scores)

def importance_plot(model: Pipeline, X: pd.DataFrame, target_col: str, drug: str) -> None:
    """Plot feature importance."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    importances = model.named_steps["model"].feature_importances_
    # get 15 most importance features
    indices = np.argsort(importances)[::-1][:15]
    selected = model.named_steps["selector"].get_support()
    plt.figure(figsize=(10, 10))
    plt.title(f"Feature Importance for {drug}")
    sns.barplot(x=importances[indices], y=X.columns[selected][indices], orient="h")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    (base_path.parent / "plots" / "xg_importance").mkdir(parents=True, exist_ok=True)
    plt.savefig(base_path.parent / "plots" / "xg_importance" / f"{target_col}_importance_{drug}.pdf")

def main():
    """Run main function."""
    drugs = ["cannabis", "meth", "heroin", "cocaine", "crack"]
    target_cols = ["white_incidents", "black_incidents"]
    omit_cols = ["FIPS", "drug", "black_population", "black_uses", "white_uses", "white_population"]#, "black_population", "black_uses", "BlackNonHispanicNum2010"]
    results = []
    for target_col in target_cols:
        for drug in drugs:
            X, y = form_data(target_col, omit_cols, drug)
            model, score = fit_model(X, y, target_col, drug)
            importance_plot(model, X, target_col, drug)
            results.append({"drug": drug, "target": target_col, "r2": score})
    results_df = pd.DataFrame(results)
    results_df["target"] = results_df["target"].map({"white_incidents": "White", "black_incidents": "Black"})
    results_df["drug"] = results_df["drug"].map({"cannabis": "Cannabis", "meth": "Meth", "heroin": "Heroin", "cocaine": "Cocaine", "crack": "Crack"})
    results_df["r2"] = results_df["r2"].round(4)
    results_df.rename(columns={"r2": "R2", "target": "Race", "drug": "Drug"}, inplace=True)
    results_df = results_df.pivot_table(index="Drug", columns="Race", values="R2")
    results_df.to_csv(base_path / "output" / "xg_r2.csv")

    
if __name__ == "__main__":
    main()
