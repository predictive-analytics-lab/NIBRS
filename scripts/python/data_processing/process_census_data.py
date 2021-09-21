"""
Script which injests county level census information,
reconfigures the age boundaries, and adds FIPS + state sub-region information.

Important: races aren't in combination with mixed.
e.g. WA_MALE (white male) is JUST white males. Does not include white mixed males.
"""
import argparse
import functools

from pathlib import Path
from typing import List, Union
from matplotlib import pyplot as plt, use
import pandas as pd
import numpy as np


data_path = Path(__file__).parent.parent.parent.parent / "data"


############ RECODING FUNCTIONS ##############

def RACE(x: pd.Series):
    """
    Function which injests a row of a census dataframe and
    changes the converts from the RACESEX config -> RACE.
    
    :param x: a row of the census dataframe
    """
    if x["RACESEX"] == "WA_MALE" or x["RACESEX"] == "WA_FEMALE":
        return "White"
    return "Black"


def SEX(x: pd.Series):
    """
    Function which injests a row of a census dataframe and
    changes the converts from the RACESEX config -> SEX.
    
    :param x: a row of the census dataframe
    """
    if x["RACESEX"] == "WA_MALE" or x["RACESEX"] == "BA_MALE":
        return "Male"
    return "Female"


def AGE(x: pd.Series):
    """
    Function which injests a row of a census dataframe and
    changes the converts the age group format to string format.
    
    :param x: a row of the census dataframe
    """
    if x == 1:
        return "0-4"
    if x == 2:
        return "5-9"
    if x == 3:
        return "10-14"
    if x == 4:
        return "15-19"
    if x == 5:
        return "20-24"
    if x == 6:
        return "25-29"
    if x == 7:
        return "30-34"
    if x == 8:
        return "35-39"
    if x == 9:
        return "40-44"
    if x == 10:
        return "45-49"
    if x == 11:
        return "50-54"
    if x == 12:
        return "55-59"
    if x == 13:
        return "60-64"
    if x == 14:
        return "65-69"
    if x == 15:
        return "70-74"
    if x == 16:
        return "75-79"
    if x == 17:
        return "80-84"
    if x == 18:
        return "85+"
    return "total"

rural_urban = {"Nonmetro": "rural",
               "Large metro": "urban",
               "Small metro": "urban"}

def reporting_coverage(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    agency_df = pd.read_csv(data_path / "misc" / "agency_participation.csv", usecols=["ori", "nibrs_participated", "population", "data_year"])
    agency_df = agency_df[agency_df.data_year == 2019]
    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t", usecols=["ORI9", "FIPS"], dtype={'FIPS': object})
    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})
    agency_df = pd.merge(agency_df, fips_ori_df, on="ori")
    agency_df = agency_df[agency_df["nibrs_participated"] != "Y"]
    coverage = (agency_df.groupby(resolution).population.sum() / df.groupby(resolution).frequency.sum()).clip(upper=1).to_frame("coverage").reset_index()
    df = df.merge(coverage, on=resolution, how="left")
    return df.fillna(0)

def disjunction(*conditions):
    """
    Function which takes a list of conditions and returns the logical OR of them.
    """
    return functools.reduce(np.logical_or, conditions)

def load_and_process_census_data(years: Union[str, List[str]]) -> pd.DataFrame:
    """
    Function which takes a list of years and returns a dataframe of the census data of the given years.
    
    In particular, it takes the following steps:
     - Reads in the data from the census data file
     - Recodes the RACESEX and AGE columns
     - Adds FIPS and state sub-region columns
    """
    if isinstance(years, str):
        if "-" in years:
            years = years.split("-")
            years = range(int(years[0]), int(years[1]) + 1)
            try:
                years = [12 - (2019 - int(yi)) for yi in years]
            except:
                print("invalid year format.")
        else:
            try:
                years = [12 - (2019 - int(years))]
            except:
                print("invalid year format.")
    elif isinstance(years, list):
        years = [12 - (2019 - int(y)) for y in years]


    #LOAD DATA
    df = pd.read_csv(data_path / "census" / f"census-2019-county.csv", encoding="ISO-8859-1", index_col=False, engine='python', usecols=[
                    "AGEGRP", "COUNTY", "STATE", "WA_MALE", "WA_FEMALE", "BA_MALE", "BA_FEMALE", "YEAR"], dtype={'STATE': object, "COUNTY": object})  # RACES NOT IN COMBINATION

    # FILTER YEARS
    df = df[disjunction(*[df.YEAR == yi for yi in years])]


    df = pd.melt(df, id_vars=["AGEGRP", "COUNTY", "STATE", "YEAR"], value_vars=[
                "WA_MALE", "WA_FEMALE", "BA_MALE", "BA_FEMALE"], var_name="RACESEX", value_name="frequency")

    df["SEX"] = df.apply(SEX, axis=1)
    df["RACE"] = df.apply(RACE, axis=1)
    df["AGE"] = df["AGEGRP"].map(AGE)
    df["FIPS"] = df["STATE"] + df["COUNTY"]
    df["YEAR"] += 2007 # Not quite sure why 2007 is the magic number.. but ¯\_(ツ)_/¯
    
    # if coverage:
    #     df = reporting_coverage(df, "FIPS")
    #     df.frequency = np.floor(df.frequency * df.coverage).astype(int)
    #     df = df.drop(["coverage"], axis=1)

    df = df[df.AGE != "total"]

    df = df.drop(["AGEGRP", "RACESEX"], axis=1)

    df.frequency /= 4

    def age_lowerbound(x):
        return int(x["AGE"].split("-")[0].split("+")[0])

    df.AGE = df.apply(age_lowerbound, axis=1)

    df_2 = df.copy(deep=True)
    df_3 = df.copy(deep=True)
    df_4 = df.copy(deep=True)

    df_2.AGE += 1
    df_3.AGE += 2
    df_4.AGE += 3

    df = pd.concat([df, df_2, df_3, df_4])

    def age_agg(x):
        if x["AGE"] < 12:
            return "drop"
        if x["AGE"] < 18:
            return "12-17"
        if x["AGE"] < 26:
            return "18-25"
        if x["AGE"] < 35:
            return "26-34"
        if x["AGE"] < 50:
            return "35-49"
        return "50+"

    df["AGE"] = df.apply(age_agg, axis=1)
    df = df[df["AGE"] != "drop"]

    df = df.groupby(["AGE", "SEX", "RACE", "FIPS", "YEAR"]).sum().reset_index()

    subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                            dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

    df = pd.merge(df, subregion_df, how='left', on='FIPS')

    fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t",
                            usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    df = pd.merge(fips_ori_df, df, on="FIPS")


    df["state_region"] = df["State"] + "-" + df["Region"]
    df = df.dropna(subset=["state_region"], how="all")

    df["SEX"] = df.SEX.apply(lambda x: x.lower())
    df["RACE"] = df.RACE.apply(lambda x: x.lower())

    # Clean up
    df.rename(columns={"AGE":"age", "SEX":"sex", "RACE":"race", "YEAR": "year", "State":"state"}, inplace=True)
    df.drop(["Region"], axis=1, inplace=True)

    return df

def add_poverty_info(df: pd.DataFrame):
    poverty_df = pd.read_csv(data_path / "misc" / "poverty_data.csv", index_col=False, dtype={"FIPS":str})
    poverty_df = poverty_df.fillna(0)
    poverty_df["FIPS"] = poverty_df.FIPS.apply(lambda x: str(x).rjust(5, "0"))
    poverty_df = poverty_df.rename(columns={"poverty_income higher than poverty level":"income higher than poverty threshold",
                               "poverty_living in poverty":"living in poverty"})
    df = df.merge(poverty_df, how="left", on=["FIPS", "age", "race", "sex"])
    df["income higher than poverty threshold"] *= df.frequency
    df["living in poverty"] *= df.frequency
    df = df.drop(columns=["frequency"])
    df = pd.melt(df, id_vars=["ori", "FIPS", "year", "state", "state_region", "age", "race", "sex"], value_vars=["living in poverty", "income higher than poverty threshold"], var_name="poverty", value_name="frequency")
    df = df.fillna(0)
    return df

def add_urban_info(df: pd.DataFrame):    
    df_pre_2015 = df[df.year < 2015]
    df_post_2015 = df[df.year >= 2015]
    
    def _add_urban(df_year: pd.DataFrame, filename_year: str) -> pd.DataFrame:
        if len(df_year) == 0:
            return pd.DataFrame()
        urban_df = pd.read_csv(data_path / "misc" / f"urban_codes_x_county_{filename_year}.csv", index_col=False, dtype={"FIPS":str}, usecols=["FIPS", "urbancounty"])
        urban_df["FIPS"] = urban_df.FIPS.apply(lambda x: str(x).rjust(5, "0"))
        urban_df["urbancounty"] = urban_df.urbancounty.map(rural_urban)
        return df_year.merge(urban_df, how="left", on="FIPS")
    
    df_pre_2015 = _add_urban(df_pre_2015, "2003")
    df_post_2015 = _add_urban(df_post_2015, "2013")
    return df_pre_2015.append(df_post_2015)


def get_census_data(years: str, poverty: bool, urban: bool):
    df = load_and_process_census_data(years)
    if poverty:
        df = add_poverty_info(df)
    if urban:
        df = add_urban_info(df)
    return df


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--year", help="year, or year range.", default=2019)
    parser.add_argument("--poverty", type=bool, help="Add poverty information.", default=True)
    parser.add_argument("--urban", type=bool, help="Add rural/urban information.", default=True)

    # parser.add_argument("--coverage", help="population scaled by coverage", default=False)

    args=parser.parse_args()
        
    df = get_census_data(str(args.year), args.poverty, args.urban)
    
    year = args.year if args.year else "2019"
    df.to_csv(data_path / "census" / f"census_processed_{year}.csv")

