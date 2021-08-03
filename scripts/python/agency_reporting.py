# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


data_path = Path(__file__).parent.parent.parent / "data"

ap_cols = [
    "ori",
    "nibrs_participated",
    "county_name",
    "population",
    "suburban_area_flag",
    "male_officer",
    "female_officer",
    "data_year"
]
agency_df = pd.read_csv(data_path / "misc" / "agency_participation.csv", usecols=ap_cols)

agency_df = agency_df[agency_df.data_year == 2019]

fips_ori_df = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t",
                          usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

agency_df = pd.merge(agency_df, fips_ori_df, on="ori")

subregion_df = pd.read_csv(data_path / "misc" / "subregion_counties.csv",
                           dtype={'FIPS': object}, usecols=["State", "Region", "FIPS"])

agency_df = pd.merge(agency_df, subregion_df, on="FIPS")

agency_df["state_region"] = agency_df["State"] + " : " + agency_df["Region"]

lemas_df = pd.read_csv(data_path / "agency" / "lemas_processed.csv", usecols=["FIPS", "PERS_EDU_MIN"])

lemas_df["FIPS"] = lemas_df.FIPS.apply(lambda x: str(x).rjust(5, "0"))

agency_df = pd.merge(agency_df, lemas_df, on="FIPS", how="left")

# %%

def coverage(df: pd.DataFrame, resolution: str, nibrs: bool = True, lemas: bool = False):
    agencies = df.groupby(resolution).size()
    df_c = df.copy()

    if lemas:
        df_c = df_c[~df_c["PERS_EDU_MIN"].isnull()]
    
    if nibrs:
        df_c = df_c[df_c["nibrs_participated"] == "Y"]
        
    conditioned_agencies = df_c.groupby(resolution).size()

    population_covered = df_c.groupby(resolution).population.sum() / df.groupby(resolution).population.sum()
    population_covered = population_covered.fillna(0)
    reporting_proportion = conditioned_agencies / agencies
    reporting_proportion = reporting_proportion.fillna(0)
    
    return reporting_proportion, population_covered
# %%

agencies_per_county, reporting_agencies_per_county, population_covered_per_county = coverage(agency_df, "FIPS")
agencies_per_region, reporting_agencies_per_region, population_covered_per_region = coverage(agency_df, "state_region")
agencies_per_state, reporting_agencies_per_state, population_covered_per_state = coverage(agency_df, "State")

# %%


agency_prop, pop_prop = coverage(agency_df, "State", nibrs=True, lemas=True)

# %%
agency_prop_lf, pop_prop_lf = coverage(agency_df, "State", nibrs=True, lemas=False)

# %%
