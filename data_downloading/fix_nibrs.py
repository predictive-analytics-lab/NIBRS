"""Script which iterates over all NIBRS CSVs and removes duplicate rows"""

import pandas as pd
from pathlib import Path

id_columns = {
    "NIBRS_SUSPECTED_DRUG_ID",
    "RELATIONSHIP_ID",
    "DRUG_MEASURE_TYPE_ID",
    "SUSPECT_USING_ID",
    "PROP_LOSS_ID",
    "ETHNICITY_ID",
    "PROPERTY_ID",
    "CRIMINAL_ACT_ID",
    "OFFENSE_ID",
    "OFFENSE_TYPE_ID",
    "CRIMINAL_ACT_ID",
    "NIBRS_PROP_DESC_ID",
    "INCIDENT_ID",
    "BIAS_ID",
    "OFFENSE_ID",
    "SUSPECT_USING_ID",
    "OFFENSE_ID",
    "VICTIM_TYPE_ID",
    "INJURY_ID",
    "OFFENDER_ID",
    "CIRCUMSTANCES_ID",
    "ACTIVITY_TYPE_ID",
    "ARRESTEE_ID",
    "WEAPON_ID",
    "BIAS_ID",
    "ARREST_TYPE_ID",
    "VICTIM_ID",
    "INJURY_ID",
    "OFFENSE_ID",
    "NIBRS_VICTIM_OFFENDER_ID",
    "CLEARED_EXCEPT_ID",
    "VICTIM_ID",
    "NIBRS_MONTH_ID",
    "AGE_ID",
    "PROP_DESC_ID",
    "VICTIM_ID",
    "CIRCUMSTANCES_ID",
    "NIBRS_ARRESTEE_WEAPON_ID",
    "SUSPECTED_DRUG_TYPE_ID",
    "VICTIM_ID",
    "OFFENSE_ID",
    "LOCATION_ID",
    "ASSIGNMENT_TYPE_ID",
    "NIBRS_WEAPON_ID",
    "RACE_ID",
    "STATE_ID",
    "agency_id",
    "JUSTIFIABLE_FORCE_ID",
}


# def get_year(path: Path) -> int:
#     """
#     Returns the year of the NIBRS CSV
#     """
#     if "-" in path.parent.name:
#         return int(path.parent.name.split("-")[-1])
#     else:
#         return (path.parent.parent.name.split("-")[-1])

def dataframe_clean(df: pd.DataFrame, path: Path):
    ids_in_frame = set(df.columns) & id_columns
    if len(ids_in_frame) == 0:
        ids_in_frame = set(df.columns) & {id.lower() for id in id_columns}
    df = df.drop_duplicates(subset=ids_in_frame, keep="first")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    path = Path(__file__).parent / "downloads"
    for file in path.glob("*/*.csv"):
        print(f"cleaning csv: {str(file)}")
        df = pd.read_csv(file, encoding="ISO-8859-1", engine='python', encoding_errors="ignore", dtype=str)
        dataframe_clean(df, file)
    for file in path.glob("*/*/*.csv"):
        print(f"cleaning csv: {str(file)}")
        df = pd.read_csv(file, encoding="ISO-8859-1", engine='python', encoding_errors="ignore", dtype=str)
        dataframe_clean(df, file)
