import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path


if __name__ == "__main__":
    data_path = Path("../data/NIBRS_202011271157.csv").resolve()
    assert data_path.exists(), f"Data does not exist at path {data_path}"
    df = pd.read_csv(data_path)
    profile = ProfileReport(df, title=f"{data_path.stem} Report")
    profile.to_file(f"../reports/{data_path.stem}.html")