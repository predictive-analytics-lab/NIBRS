# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data_path = Path(__file__).parent.parent.parent / "data"

agency_lemas = pd.read_csv(data_path / "output" / "agency_lemas.csv", index_col=0)

lemas_columns = [
    "ISSU_ADDR_BIAS",
    "ISSU_ADDR_DRUG_ENF",
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

# %%

binary_columns = set(agency_lemas[lemas_columns].columns[[agency_lemas[lemas_columns].nunique() <=5]])
cont_columns = set(agency_lemas[lemas_columns].columns[[agency_lemas[lemas_columns].nunique() >= 10]])
categorical_columns = set(lemas_columns) - binary_columns - cont_columns
# %%

binary_conversion_5 = {
    1: 1,
    2: 1,
    -88: 0,
    -9: -1,
    -8: -1
}

binary_conversion_4 = {
    1: 1,
    2: 0,
    -9: -1,
    -8: -1
}

def binary_convert(r):
    r = list(r.values)
    if -88 in r:
        return list(map(binary_conversion_5.get, r))
    else:
        return list(map(binary_conversion_4.get, r))

def cont_convert(r):
    r = list(r.values)
    return [np.max([-1, ri]) for ri in r]



for col in binary_columns:
    agency_lemas[col] = binary_convert(agency_lemas[col])
    
for col in cont_columns:
    agency_lemas[col] = cont_convert(agency_lemas[col])

# %%

categorical_columns
# %%

issu_addr_dict = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 3,
    -9: -1
}

chief_race_dict = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 4,
    7: 4, 
    8: 4,
    -9: -1
}

edu_dict = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 3,
    -9: -1
}

agency_lemas["ISSU_ADDR_BIAS"] = agency_lemas["ISSU_ADDR_BIAS"].map(issu_addr_dict.get)
agency_lemas["ISSU_ADDR_DRUG_ENF"] = agency_lemas["ISSU_ADDR_DRUG_ENF"].map(issu_addr_dict.get)
agency_lemas["PERS_CHF_RACE"] = agency_lemas["PERS_CHF_RACE"].map(chief_race_dict.get)
agency_lemas["PERS_EDU_MIN"] = agency_lemas["PERS_EDU_MIN"].map(edu_dict.get)



# %%

def binary_aggregation(x: pd.DataFrame, column: str):
    x = x[x[column] > 0]
    prop_pop = x.population / x.population.sum()
    t_val = (x[column] * prop_pop).sum()
    f_val = 1 - t_val
    return t_val, f_val
    
    


def aggregate_columns(group: pd.DataFrame):
    # group = agency_lemas[agency_lemas.index.isin(group)]
    for col in binary_columns:
        t_val, f_val = binary_aggregation(group, col)
        group[col] = t_val
        group[f"NOT {col}"] = f_val
    return group
# %%

test = agency_lemas.groupby("state_name").apply(aggregate_columns)

# %%
