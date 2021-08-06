"""This python script processes, and modifies, the NIBRS data output from the SQL query."""
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent.parent / "data"

nibrs_df = pd.read_csv(data_path / "NIBRS" / "cannabis_agency_2019_20210608.csv", usecols=[
                       "dm_offender_race_ethnicity", "dm_offender_age", "dm_offender_sex", "arrest_type", "cannabis_mass", "ori"])

nibrs_df.rename(columns={
    "dm_offender_race_ethnicity": "RACE",
    "dm_offender_age": "AGE",
    "dm_offender_sex": "SEX"
}, inplace=True)

nibrs_df = nibrs_df[nibrs_df["RACE"] != "hispanic/latino"]

fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t",
                          usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                           dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

nibrs_df = pd.merge(nibrs_df, fips_ori_df, on="ori")
nibrs_df = pd.merge(nibrs_df, subregion_df, on="FIPS")
nibrs_df["STATEREGION"] = nibrs_df["State"] + "-" + nibrs_df["Region"]

nibrs_df.to_csv(data_path / "NIBRS" / "cannabis_processed.csv")

nibrs_arrests = nibrs_df[nibrs_df.arrest_type != "No Arrest"]

nibrs_arrests.to_csv(data_path / "NIBRS" / "cannabis_arrests_processed.csv")
