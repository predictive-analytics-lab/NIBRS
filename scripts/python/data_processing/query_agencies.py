from os import PathLike
from process_census_data import load_and_process_census_data
from typing import Generator, List, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm

data_dir = Path(__file__).resolve().parents[3] / "data_downloading" / "downloads"
data_path = Path(__file__).resolve().parents[3] / "data"

def search_filename(name: str, dir: Path) -> Path:
    for filename in dir.iterdir():
        if name in filename.name.lower():
            return filename

def query_one_year(state_year_dir: Path) -> pd.DataFrame:
    """Combine a single year-state combination into the desired df form."""
    dirs = [f for f in state_year_dir.iterdir() if f.is_dir()]
    if len(dirs) > 0:
        # Sometimes we end up with /year-state/state/
        state_year_dir = dirs[0]

    def read_csv(name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        filepath = search_filename(name, state_year_dir)
        try:
            df = pd.read_csv(filepath)
        except:
            breakpoint()
        df.columns = df.columns.str.lower()
        if usecols is not None:
            df = df[usecols]
        return df
        
    return read_csv("agencies", usecols=["ori", "population"])


def combine_agencies(years: List[str]) -> pd.DataFrame:
    combined_df = pd.DataFrame()
    for sy_dir in tqdm(list(data_dir.iterdir())):
        data_year = sy_dir.stem.split("-")[-1]
        if data_year not in years:
            continue
        df = query_one_year(sy_dir)
        df["year"] = data_year
        combined_df = combined_df.append(df)
    return combined_df


def coverage(agency_df: pd.DataFrame) -> pd.DataFrame:
    # Add FIPS to agency df
    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t", usecols=["ORI9", "FIPS"], dtype={'FIPS': object})
    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})
    agency_df = agency_df.merge(fips_ori_df, on="ori")
    agency_df["year"] = agency_df.year.astype(str)
    # Load Census Remove ORI duplicates
    census = load_and_process_census_data(list(agency_df.year.unique()))
    census = census.drop(columns=["ori"])
    census = census.drop_duplicates()
    census["year"] = census.year.astype(str)

    census = census.groupby(["FIPS", "year"]).frequency.sum().to_frame("frequency").reset_index()
    
    agency_df = agency_df.groupby(["FIPS", "year"]).population.sum().to_frame("population").reset_index()
    
    census = census.merge(agency_df, on=["FIPS", "year"], how="left")
    census = census.fillna(0)
    
    census["coverage"] = census.population / census.frequency
    
    return census[["FIPS", "year", "coverage", "population", "frequency"]]


    
    

if __name__ == "__main__":
    df = combine_agencies([str(x) for x in range(2012, 2020)])
    county_coverage = coverage(df)
    df.to_csv(data_path / "misc" / "agencies.csv")
    county_coverage.to_csv(data_path / "misc" / "county_coverage.csv")
