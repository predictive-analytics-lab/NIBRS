"""
Script which joins the agency output (with selection bias) to the lemas dataset.

IMPORTANT: Script currently replaces all infinite values with zero
(for the selection bias) - to avoid aggregation issues downstream.

"""

import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path(__file__).parent.parent.parent.parent / "data"

def load_and_merge_data() -> pd.DataFrame:

    ## LOAD DATASETS

    agency_df = pd.read_csv(data_path / "output" / "agency_output.csv", index_col=0)

    agency_df.replace(np.inf, 0, inplace=True)

    lemas_df = pd.read_csv(data_path / "agency" / "lemas_processed.csv", index_col=0)

    subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                            dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

    lemas_df["FIPS"] = lemas_df.FIPS.apply(lambda x: str(x).rjust(5, "0"))

    lemas_df = pd.merge(lemas_df, subregion_df, on="FIPS")

    lemas_df["state_region"] = lemas_df["State"] + " : " + lemas_df["Region"]


    agency_df = pd.merge(lemas_df, agency_df, on="ori", how="left")

    return agency_df

if __name__ == "__main__":
    agency_df = load_and_merge_data()
    agency_df.to_csv(data_path / "output" / "agency_lemas.csv")