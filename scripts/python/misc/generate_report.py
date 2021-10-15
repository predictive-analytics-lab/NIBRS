"""Script which uses the pandas-profiling library to create a HTML report of a given dataset."""

import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path
from sys import argv


if __name__ == "__main__":
    try:
        data_path = Path(argv[1]).expanduser().resolve()
    except IndexError:
        raise ValueError("No file specified.")
    assert data_path.exists(), f"Data does not exist at path {data_path}"
    df = pd.read_csv(data_path)
    minimal = False
    profile = ProfileReport(df, title=f"{data_path.stem}Report", minimal=minimal)
    profile.to_file(f"../reports/{data_path.stem}{'_min' if minimal else ''}.html")
