import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path(__file__).parents[3] / "data" / "output"

def confidence_categorization(df: pd.DataFrame, value_col: str, ci_col: str) -> pd.DataFrame:
    def _categorization(v, ci):
        if v - ci > 5:
            return "S>5"
        if v - ci > 2:
            return "S>2"
        if v - ci > 1:
            return "S>1"
        if v + ci < 1:
            return "S<1"
        if v + ci < 0.5:
            return "S<0.5"
        if v + ci < 0.2:
            return "S<0.2"
        return "Low confidence"
    df["cat"] = df.apply(lambda x: _categorization(x[value_col], x[ci_col]), axis=1)
    return df

def additions(df: pd.DataFrame) -> pd.DataFrame:
    # df = confidence_categorization(df, "selection_ratio", "ci")

    # df["frequency"] = df["frequency"].apply(lambda x: f'{int(x):,}')
    # df["bwratio"] = df["bwratio"].apply(lambda x: f'{x:.3f}')

    # df["slci"] = df["selection_ratio"].round(3).astype(str) + " ± " + df["ci"].round(3).astype(str)
    df["selection_ratio_log10"] = np.log10(df["selection_ratio"])

    return df

if __name__ == "__main__":
    df = pd.read_csv(str(data_path / "cannabis_unsmoothed.csv"), index_col=False, dtype={"FIPS": str})
    df = additions(df)
    df.to_csv(data_path / "cannabis_unsmoothed.csv", index=False)