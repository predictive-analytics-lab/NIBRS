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
cont_columns = set(agency_lemas[lemas_columns].columns[[agency_lemas[lemas_columns].nunique() >= 10]]).union({"bwratio"})
categorical_columns = set(lemas_columns) - binary_columns - cont_columns
special_columns = {"incidents", "population", "selection_ratio"}
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

def continuous_aggregation(x: pd.DataFrame, column: str):
    x = x[x[column] > 0]
    # x = x[~x[column].isna()]
    prop_pop = x.population / x.population.sum()
    return (x[column] * prop_pop).sum()


def categorical_aggregation(x: pd.DataFrame, column: str):
    x = x[x[f"{column}_-1.0"].astype(int) != 1]
    matching_cols = [col for col in x if col.startswith(column)]
    # x = x[x[matching_cols].sum(axis=1) > 0]
    prop_pop = x.population / x.population.sum()
    return (x[matching_cols].mul(prop_pop.values, axis=0)).sum(axis=0)

def weighted_std(x: pd.DataFrame, column: str):
    prop_pop = x.population / x.population.sum()
    average = np.average(x[column], weights=prop_pop)
    return np.sqrt(np.average((x[column]-average)**2, weights=prop_pop))


def aggregate_columns(group: pd.DataFrame):
    output = pd.DataFrame()
    for col in cont_columns:
        output[col] = [continuous_aggregation(group, col)]
    for col in categorical_columns.union(binary_columns):
        cat_output = categorical_aggregation(group, col)
        output = pd.concat([output, pd.DataFrame([cat_output])], axis=1)
    for col in special_columns:
        if col == "incidents":
            output[col] = group[col].sum()
        if col == "population":
            output["population_covered"] = group[col].sum()
        if col == "selection_ratio":
            output[col] = continuous_aggregation(group, col)
            output["selection ratio std"] = weighted_std(group, col)
    return output


# %%
from functools import partial


def waverage(x: pd.DataFrame, column: str, weighting_column: str):
    weight = x[weighting_column] / x[weighting_column].sum()
    return (x[column] * weight).sum()
    
def wstd(x: pd.DataFrame, column: str, weighting_column: str):
    weight = x[weighting_column] / x[weighting_column].sum()
    average = np.average(x[column], weights=weight)
    return np.sqrt(np.average((x[column]-average)**2, weights=weight))

def selection_aggregation(group: pd.DataFrame):
    output = pd.DataFrame()
    output["nibrs_selection_pop"] = [waverage(group, column="selection_ratio", weighting_column="population")]
    output["nibrs_selection_inc"] = [waverage(group, column="selection_ratio", weighting_column="incidents")]
    output["nibrs_selection_pop_std"] = [wstd(group, column="selection_ratio", weighting_column="population")]
    output["nibrs_selection_inc_std"] = [wstd(group, column="selection_ratio", weighting_column="incidents")]
    return output

def agency_resolution(resolution: str):
    agency_lemas_filtered = agency_lemas[agency_lemas.incidents > 0]
    nibrs_incidents = agency_lemas_filtered.groupby(resolution)["incidents"].sum().to_frame("nibrs_incidents").reset_index()
    
    nibrs_selection_pop = agency_lemas_filtered.groupby(resolution).apply(selection_aggregation)
        
    agency_lemas_filtered = agency_lemas_filtered[agency_lemas_filtered["PERS_EDU_MIN"].notna()]
    agency_lemas_1hot = pd.get_dummies(agency_lemas_filtered, columns=list(categorical_columns.union(binary_columns)))
    agency_lemas_1hot = agency_lemas_1hot.groupby(resolution).apply(aggregate_columns)
    columns_to_drop = [column for column in agency_lemas_1hot.columns if column.endswith("-1.0")]
    agency_lemas_1hot = agency_lemas_1hot.drop(columns_to_drop, axis=1)
    agency_lemas_1hot["population_covered_proportion"] = (agency_lemas_1hot.groupby(resolution)["population_covered"].sum() / agency_lemas.groupby(resolution)["population"].sum()).dropna().values
    agency_lemas_1hot = agency_lemas_1hot.reset_index().drop(["level_1"], axis=1)
    agency_lemas_1hot = pd.merge(agency_lemas_1hot, nibrs_incidents, on=resolution)
    
    agency_lemas_1hot = pd.merge(agency_lemas_1hot, nibrs_selection_pop, on=resolution)

    agency_lemas_1hot.to_csv(data_path / "output" / f"aggregated_lemas_{resolution}.csv")

# %%

agency_resolution("State")
# agency_resolution("state_region")
# agency_resolution("FIPS")


# %%

# Coverage Percentage - DONE
# Drop missing stuff - DONE
# Create map(s) with STD
# Generate report


# NIBRS MAP - BIAS (+ STD), NIBRS Coverage


# Selection ratio weighted by Incidents
# Selection ratio without dropping lemas