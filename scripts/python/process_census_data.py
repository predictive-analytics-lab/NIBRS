from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

data_path = Path(__file__).parent.parent.parent / "data"


df = pd.read_csv(data_path / "demographics" / "census-2019-county.csv", encoding="ISO-8859-1", engine='python', usecols=[
                 "AGEGRP", "COUNTY", "STATE", "WA_MALE", "WA_FEMALE", "BA_MALE", "BA_FEMALE", "YEAR"], dtype={'STATE': object, "COUNTY": object})  # RACES NOT IN COMBINATION
df = df[df["YEAR"] == 12]
df = pd.melt(df, id_vars=["AGEGRP", "COUNTY", "STATE"], value_vars=[
             "WA_MALE", "WA_FEMALE", "BA_MALE", "BA_FEMALE"], var_name="RACESEX", value_name="frequency")


def RACE(x):
    if x["RACESEX"] == "WA_MALE" or x["RACESEX"] == "WA_FEMALE":
        return "White"
    return "Black"


def SEX(x):
    if x["RACESEX"] == "WA_MALE" or x["RACESEX"] == "BA_MALE":
        return "Male"
    return "Female"


def AGE(x):
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


df["SEX"] = df.apply(SEX, axis=1)
df["RACE"] = df.apply(RACE, axis=1)
df["AGE"] = df["AGEGRP"].map(AGE)
df["FIPS"] = df["STATE"] + df["COUNTY"]

df = df[df.AGE != "total"]

df = df.drop(["AGEGRP", "RACESEX"], axis=1)

df = pd.read_csv(data_path / "demographics" /
                 "counties_processed.csv", dtype={'FIPS': object}, index_col=0)

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
df = df.groupby(["AGE", "SEX", "RACE", "FIPS"]).sum().reset_index()
df = df.drop(["COUNTY", "STATE"], axis=1)

subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                           dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

df = pd.merge(df, subregion_df, how='left', on='FIPS')

df["STATEREGION"] = df["State"] + "-" + df["Region"]
df = df.dropna(subset=["STATEREGION"], how="all")

df["SEX"] = df.SEX.apply(lambda x: x.lower())
df["RACE"] = df.RACE.apply(lambda x: x.lower())

df.to_csv(data_path / "demographics" / "county_census.csv")
