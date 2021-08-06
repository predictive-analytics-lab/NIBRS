"""
This script is used to explore the large biased agencies in agency_lemas.csv dataset.

"""

import pandas as pd
from pathlib import Path


def filter_agencies(df: pd.DataFrame) -> pd.DataFrame:
    df["force_bw"] = (df["PERS_BLACK_MALE"] + df["PERS_BLACK_FEM"]) / (df["PERS_WHITE_MALE"] + df["PERS_WHITE_FEM"])
    
    # Get the 100 highest population agencies, then sort by selection bias ratio
    filtered_agency = df.iloc[df.iloc[df.population.sort_values(ascending=False).iloc[:100].index].selection_ratio.sort_values(ascending=False).index].reset_index()
    
    # Further, filter these that have at least 100 incidents.
    filtered_agency = filtered_agency[filtered_agency["incidents"] > 100]
    
    return filtered_agency


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent.parent / "data"
    agency_df = pd.read_csv(data_path / "output" / "agency_lemas.csv")
    filtered_agency = filter_agencies(agency_df)
    filtered_agency.to_csv(data_path / "output" / "highest_agencies.csv")