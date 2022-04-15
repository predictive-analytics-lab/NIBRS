from os import PathLike
import re
from typing import Generator, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher

data_dir = Path(__file__).resolve(
).parents[3] / "data_downloading" / "downloads"
output_dir = Path(__file__).resolve().parents[3] / "data" / "NIBRS" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)


def search_filename(name: str, dir: Path) -> Path:
    for filename in dir.iterdir():
        if name == "agencies":
            if f"{name}.csv" in filename.name.lower():
                return filename
        else:
            if f"nibrs_{name}.csv" in filename.name.lower():
                return filename


def query_one_state_year(
    year_dir: Path,
    arrests: bool = False,
    # hispanics: bool = False,
    # all_incidents: bool = False,
) -> pd.DataFrame:
    """Combine a single year-state combination into the desired df form."""
    if (subdir := year_dir / next(year_dir.iterdir())).is_dir():
        # Sometimes we end up with /year-state/subdir_with_random_name/
        year_dir = subdir

    def read_csv(name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        filepath = search_filename(name, year_dir)
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.lower()
        if usecols is not None:
            df = df[usecols]
        return df

    offender_df = read_csv(
        "offender",
        usecols=[
            "offender_id",
            "incident_id",
            "offender_seq_num",
            "age_num",
            "sex_code",
            "race_id",
            "ethnicity_id",
        ],
    )
    incident_df = read_csv(
        "incident",
        usecols=[
            "agency_id",
            "incident_id",
            "nibrs_month_id",
            "cleared_except_id",
            "incident_date",
            "incident_hour",
        ],
    )
    main_df = offender_df.merge(incident_df, on="incident_id", how="left")
    arrestee_df = read_csv(
        "arrestee", usecols=["incident_id", "arrestee_seq_num", "arrest_type_id"]
    )
    arrestee_df = arrestee_df.rename(
        columns={"arrestee_seq_num": "offender_seq_num"})
    arrestee_df = arrestee_df.merge(
        read_csv("arrest_type", usecols=[
                 "arrest_type_id", "arrest_type_name"]),
        on="arrest_type_id",
        how="left",
    )
    main_df = main_df.merge(
        arrestee_df, on=["incident_id", "offender_seq_num"], how="left"
    )
    main_df["arrest_type_name"] = main_df["arrest_type_name"].fillna(
        "No Arrest")
    if arrests:
        main_df = main_df[main_df.arrest_type_name != "No Arrest"]
    main_df = main_df.drop(columns=["offender_seq_num"])

    # Add in agency ORI
    main_df = main_df.merge(
        read_csv("agencies", usecols=["agency_id", "ori"]), on="agency_id", how="left"
    )

    # Map race
    main_df["race"] = main_df["race_id"].map({1: "white", 2: "black"})

    main_df = main_df.merge(
        read_csv("month", usecols=[
                 "nibrs_month_id", "data_year", "month_num"]),
        on="nibrs_month_id",
    )
    fips_ori_df = pd.read_csv(
        Path(__file__).parents[3] / "data" / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "FIPS"],
        dtype={"FIPS": str},
    )
    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    main_df = main_df.merge(fips_ori_df, on="ori", how="left")

    return main_df[["data_year", "month_num", "ori", "FIPS", "race"]]


def query_all(downloads_dir: Path, arrests: bool) -> Tuple[Path, set]:
    temp_name = f"temp_csv.csv"
    years = set()
    for sy_dir in tqdm(list(downloads_dir.iterdir())):
        if (year_match := re.search(r'\d+', sy_dir.stem)) is not None:
            year = int(year_match.group())
            years.add(year)
        df = query_one_state_year(sy_dir, arrests=arrests)
        if df is not None:
            if (output_dir / temp_name).exists():
                df.to_csv(output_dir / temp_name, mode="a", header=False)
            else:
                df.to_csv(output_dir / temp_name)
    return output_dir / temp_name, years


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--arrests", action="store_true", default=False)

    args = parser.parse_args()

    df_path, years = query_all(data_dir, args.arrests)

    options = f"_{min(years)}-{max(years)}{'_arrests' if args.arrests else ''}"
    df_path.rename(output_dir / f"all_incident_count{options}.csv")
