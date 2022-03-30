# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data_path = Path(__file__).parent.parent.parent / "data"

lemas_columns = [
    "PERS_EDU_MIN",
    "PERS_MIL",
    "PERS_CITZN",
    "PERS_TRN_ACAD",
    "PERS_TRN_FIELD",
    'PERS_TRN_INSVC',
    'PERS_BACKINV',
    'PERS_CREDHIS',
    'PERS_CRIMHIS',
    'PERS_DRIVHIS',
    'PERS_SOCMED',
    'PERS_INTERVW',
    'PERS_PERSTEST',
    'PERS_POLY',
    'PERS_PSYCH',
    'PERS_VOICE',
    'PERS_APTEST',
    'PERS_PROBSOLV',
    'PERS_CULTURE',
    'PERS_CONFLICT',
    'PERS_DRUG',
    'PERS_MED',
    'PERS_VISN',
    'PERS_PHYS',
    "PERS_BILING_SWN",
    "PERS_NEW_WHT",
    "PERS_NEW_BLK",
    "PERS_NEW_TOTR",
    "PERS_NEW_MALE",
    "PERS_NEW_FEM",
    "PERS_WHITE_MALE",
    "PERS_WHITE_FEM",
    "PERS_BLACK_MALE",
    "PERS_BLACK_FEM",
    "PERS_CHF_SEX",
    "PERS_CHF_RACE",
    "PERS_SUP_INTM_WH",
    "PERS_SUP_INTM_BK",
    "PERS_SUP_SGT_WH",
    "PERS_SUP_SGT_BK",
    "OPER_CFS",
    "OPER_DIS",
    "CP_MISSION",
    "CP_PLAN",
    "CP_TECH",
    "CP_CPACAD",
    "CP_PSP_ADVGRP",
    "CP_PSP_NEIGH",
    "CP_SURVEY",
    "EQ_PRM_NOAUTH",
    "EQ_BCK_NOAUTH",
    "EQ_BDYARM_NOAUTH",
    "EQ_SEMI_NOAUTH",
    "EQ_REV_NOAUTH",
    'EQ_AUTH_OHAND',
    'EQ_AUTH_CHAND',
    'EQ_AUTH_TKDWN',
    'EQ_AUTH_NECK',
    'EQ_AUTH_LEG',
    'EQ_AUTH_OC',
    'EQ_AUTH_CHEM',
    'EQ_AUTH_BTN',
    'EQ_AUTH_BLNT',
    'EQ_AUTH_CED',
    'EQ_AUTH_EXP',
    "EQ_VID_CAR",
    "EQ_VID_BWC",
    'TECH_WEB_NONE',
    'TECH_WEB_STAT',
    'TECH_WEB_STOP',
    'TECH_WEB_ARR',
    'TECH_WEB_REPORT',
    'TECH_WEB_ASK',
    'TECH_WEB_COMPL',
    "TECH_COMP_CRMANL",
    "TECH_TYP_FACEREC",
    "TECH_EIS",
    "POL_RACPROF",
    "POL_STFRSK",
    "POL_MSCOND",
    "POL_BWC",
    "POL_CULTAW",
    "ISSU_ADDR_BIAS",
    "ISSU_ADDR_DRUG_ENF",
]

agency_lemas = pd.read_csv(data_path / "output" / "agency_lemas.csv", dtype={"FIPS": str}, usecols=lemas_columns + ["FIPS", "population"])
# %%

binary_columns = set(agency_lemas[lemas_columns].columns[[agency_lemas[lemas_columns].nunique() <=5]])
cont_columns = set(agency_lemas[lemas_columns].columns[[agency_lemas[lemas_columns].nunique() >= 10]])
categorical_columns = set(lemas_columns) - binary_columns - cont_columns - {"FIPS", "population"}
# %%

def binary_convert(r):
    r.loc[r == 1] = "_yes"
    r.loc[r == 2] = "_no"
    r.loc[r == -88] = "_no"
    r.loc[r == -9] = np.nan
    r.loc[r == -8] = np.nan
    return r

def cont_convert(r):
    r.loc[r < 0] = np.nan
    return r

for col in binary_columns:
    agency_lemas[col] = binary_convert(agency_lemas[col])
    
for col in cont_columns:
    agency_lemas[col] = cont_convert(agency_lemas[col])


# %%

issu_addr_dict = {
    1: "_personnel",
    2: "_personnel",
    3: "_addressed",
    4: "_not_addressed",
    5: "_not_a_problem",
    -9: np.nan
}

chief_race_dict = {
    1: "_white",
    2: "_black",
    3: "_hispanic",
    4: "_other",
    5: "_other",
    6: "_other",
    7: "_other", 
    8: "_other",
    -9: np.nan
}

edu_dict = {
    1: "_college",
    2: "_college",
    3: "_college",
    4: "_high-school",
    5: "_none",
    -9: np.nan
}

agency_lemas["ISSU_ADDR_BIAS"] = agency_lemas["ISSU_ADDR_BIAS"].map(issu_addr_dict.get)
agency_lemas["ISSU_ADDR_DRUG_ENF"] = agency_lemas["ISSU_ADDR_DRUG_ENF"].map(issu_addr_dict.get)
agency_lemas["PERS_CHF_RACE"] = agency_lemas["PERS_CHF_RACE"].map(chief_race_dict.get)
agency_lemas["PERS_EDU_MIN"] = agency_lemas["PERS_EDU_MIN"].map(edu_dict.get)



# %%

def continuous_aggregation(x: pd.DataFrame, column: str):
    denom = x.population[x[column] != "_missing"].sum()
    if denom == 0:
        weight_mult = 1
    else:
        weight_mult = x.population.sum() / denom
    weight = (((x.population / x.population.sum())[x[column] != "_missing"]) * weight_mult)[x[column] != "_missing"]
    x = x[x[column] != "_missing"]
    # x = x[~x[column].isna()]
    if len(x) <= 0:
        return np.nan
    return round((x[column] * weight).sum())



def categorical_aggregation(x: pd.DataFrame, column: str):
    matching_cols = [col for col in x if col.startswith(column)]
    prop_pop = x.population / x.population.sum()
    return (x[matching_cols].mul(prop_pop.values, axis=0)).sum(axis=0)

def aggregate_columns(group: pd.DataFrame):
    output = pd.DataFrame()
    for col in cont_columns:
        output[col] = [continuous_aggregation(group, col)]
    for col in categorical_columns.union(binary_columns):
        cat_output = categorical_aggregation(group, col)
        output = pd.concat([output, pd.DataFrame([cat_output])], axis=1)
    return output


# %%

from tqdm.notebook import tqdm
tqdm.pandas()


def aggregate(df: pd.DataFrame):
    df = df.fillna(value=np.nan)
    df.loc[:, lemas_columns] = df.loc[:, lemas_columns].fillna(value="_missing")
    agency_lemas_1hot = pd.get_dummies(df, columns=list(categorical_columns.union(binary_columns)))
    agency_lemas_1hot = agency_lemas_1hot.groupby("FIPS").progress_apply(aggregate_columns)
    return agency_lemas_1hot

output = aggregate(agency_lemas)
# %%
