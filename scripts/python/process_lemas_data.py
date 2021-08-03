"""This script loads and processes the LEMAS dataset."""
# %%
import pandas as pd

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent / "data"

# %%

ap_cols = [
    "ori",
    "nibrs_participated",
    "state_name",
    "county_name",
    "population",
    "suburban_area_flag",
    "male_officer",
    "female_officer",
    "data_year"
]
ap_df = pd.read_csv(data_dir / "misc" / "agency_participation.csv", usecols=ap_cols)

ap_df = ap_df[ap_df.data_year == 2019]

fips_ori_df = pd.read_csv(data_dir / "misc" / "LEAIC.tsv", delimiter="\t",
                          usecols=["ORI9", "FIPS"], dtype={'FIPS': object})

fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

ap_df = pd.merge(ap_df, fips_ori_df, on="ori")

# %%

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
    'EQ_AUTH_DIV',
    'EQ_AUTH_K9',
    'EQ_AUTH_FIREARM',
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
    "ORI9"
]

lemas_df = pd.read_csv(data_dir / "agency" / "lemas_2016.tsv", sep='\t', usecols=lemas_columns)
# %%
lemas_df = lemas_df.rename(columns={"ORI9": "ori"})


lemas_df = pd.merge(lemas_df, ap_df, on="ori")
# %%

lemas_df.to_csv(data_dir / "agency" / "lemas_processed.csv")
# %%
