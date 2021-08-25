from os import PathLike
from typing import Generator, List, Optional
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parents[3] / "data_downloading" / "downloads"

def search_filename(name: str, dir: Path) -> Path:
    for filename in dir.iterdir():
        if name in filename.name.lower():
            return filename

def query_one_year(year_dir: Path) -> pd.DataFrame:
    """Combine a single year-state combination into the desired df form."""
    if (subdir := year_dir / year_dir.stem.split("-")[0]).is_dir():
        # Sometimes we end up with /year-state/state/
        year_dir = subdir

    def read_csv(name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        filepath = search_filename(name, year_dir)
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
        if usecols is not None:
            df = df[usecols]
        return df


    offender_df = read_csv("offender", usecols=['offender_id', 'incident_id', 'offender_seq_num', 'age_num', 'sex_code', 'race_id', 'ethnicity_id'])
    incident_df = read_csv("incident", usecols=['agency_id', 'incident_id', 'nibrs_month_id', 'cleared_except_id'])
    main_df = offender_df.merge(incident_df, on="incident_id", how="left")
    arrestee_df = read_csv("arrestee", usecols=["incident_id", "arrestee_seq_num", "arrest_type_id"])
    arrestee_df = arrestee_df.rename(columns={'arrestee_seq_num': "offender_seq_num"})
    arrestee_df = arrestee_df.merge(read_csv("arrest_type", usecols=["arrest_type_id", "arrest_type_name"]), on="arrest_type_id", how="left")
    main_df = main_df.merge(arrestee_df, on=["incident_id", "offender_seq_num"], how="left")
    main_df["arrest_type_name"] = main_df["arrest_type_name"].fillna("No Arrest")
    main_df = main_df.drop(columns=["offender_seq_num"])
    # property_df = read_csv("property", usecols=["property_id", "incident_id"])\
    #     .merge(read_csv("property_desc", usecols=["property_id", "prop_desc_id", "property_value"]), on="property_id", how="left")
    # property_df["property_count"] = 1 # 0 if no desc?
    # property_df["drug_equipment_value"] = property_df["property_value"] * (property_df["prop_desc_id"] == 11).astype(float)
    # property_df = property_df.fillna(0.0)
    # property_df = property_df.groupby("property_id").agg({
    #     'drug_equipment_value': "sum",
    #     'property_count': "sum"
    # })
    drug_property_df = read_csv("property", usecols=["property_id", "incident_id"])\
         .merge(read_csv("suspected_drug", usecols=["property_id", "suspected_drug_type_id"]))
    drugs = ["crack", "cocaine", "heroin", "cannabis", "meth"]
    drug_ids = [1, 2, 4, 5, 12]
    for drug, id in zip(drugs, drug_ids):
        drug_property_df[f"{drug}_count"] = drug_property_df["suspected_drug_type_id"] == id
    drug_property_df["other_drugs_count"] = ~drug_property_df["suspected_drug_type_id"].isin(drug_ids)
    drug_property_df = drug_property_df.groupby("incident_id").sum()
    drug_property_df = drug_property_df.drop(columns=['property_id', 'suspected_drug_type_id'])
    main_df = main_df.merge(drug_property_df, on=["incident_id"], how="inner")


    criminal_act_df = read_csv("criminal_act", usecols=["offense_id", "criminal_act_id"])
    criminal_act_df["criminal_act"] = criminal_act_df["criminal_act_id"].map({
        8: 1,
        6: 2,
        1: 3
    }, na_action="ignore")
    criminal_act_df["criminal_act"] = criminal_act_df["criminal_act"].fillna(0)
    offense_df = read_csv("offense", usecols=["offense_id", "incident_id"])
    criminal_act_df = criminal_act_df.merge(offense_df, on="offense_id", how="left")
    criminal_act_df['criminal_act_count'] = 1
    criminal_act_df['other_criminal_act_count'] = ~criminal_act_df["criminal_act_id"].isin([1, 6, 8])
    criminal_act_df = criminal_act_df.drop(columns=["criminal_act_id", "offense_id"])
    criminal_act_df = criminal_act_df.groupby("incident_id").agg({
        'criminal_act_count': "sum",
        'other_criminal_act_count': "sum",
        'criminal_act': "max"
    })
    criminal_act_df['criminal_act'] = criminal_act_df['criminal_act'].map({
        1: "consuming",
        2: "possessing",
        3: "buying"
    })
    main_df = main_df.merge(criminal_act_df, on="incident_id", how="left")

    # Add number of offenders per incident
    main_df = main_df.merge(main_df.groupby("incident_id").size().to_frame("offender_count"), on="incident_id")

    # Add number of offenses per incident
    main_df = main_df.merge(offense_df.groupby("incident_id").size().to_frame("offense_count"), on="incident_id")

    # Add in agency ORI
    main_df = main_df.merge(read_csv("agencies", usecols=["agency_id", "ori"]), on="agency_id", how="left")

    # Map race
    main_df['race'] = main_df['race_id'].map({
        1: "white",
        2: "black"
    })

    return main_df

if __name__ == "__main__":
    df = query_one_year(data_dir / "TX-2019") # next(data_dir.iterdir()))

