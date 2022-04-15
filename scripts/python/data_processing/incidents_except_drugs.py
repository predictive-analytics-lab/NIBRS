"""Produce the data for NIBRS incidents with the number of drug incidents removed."""
from typing import List
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parents[3] / "data"


def drug_incidents(drug_list: List[str] = ["meth", "cocaine", "crack", "heroin", ""], arrests: bool = False):
    """Get the number of drug incidents per person."""
    if arrests:
        cols = ["black_arrests", "white_arrests"]
    else:
        cols = ["black_incidents", "white_incidents"]

    def _load_drug(drug: str):
        df = pd.read_csv(
            base_path / "output" /
            f"selection_ratio_county_2010-2020_wilson{'_' + drug if drug else ''}{'_arrests' if arrests else ''}.csv",
            dtype={"FIPS": str},
            usecols=["FIPS", "year", *cols]
        )
        df = df.rename(columns={"black_arrests": "black_incidents",
                                "white_arrests": "white_incidents"})
        return df[["FIPS", "year", "black_incidents", "white_incidents", ]]
    df = pd.concat([_load_drug(drug) for drug in drug_list]).groupby(
        ["FIPS", "year"]).sum().reset_index()
    pop_df = pd.read_csv(base_path / "output" / f"selection_ratio_county_2010-2020_wilson_{drug_list[0]}.csv", dtype={"FIPS": str}, usecols=[
        "FIPS", "year", "black_population", "white_population"])
    df = df.merge(pop_df, on=["FIPS", "year"])
    df = df.rename(columns={"black_incidents": "black_drug_incidents",
                   "white_incidents": "white_drug_incidents"})
    return df[["FIPS", "year", "white_drug_incidents", "black_drug_incidents", "black_population", "white_population"]]


def other_incidents(drug_df: pd.DataFrame, arrests: bool) -> pd.DataFrame:
    """Get the number of all incidents per county."""
    df = pd.read_csv(base_path / "NIBRS" /
                     f"all_incidents_per_county{'_arrests' if arrests else ''}.csv", dtype={"FIPS": str},)
    df = df.merge(drug_df, on=["FIPS", "year"])
    df["black_incidents"] = df["black_incidents"] - df["black_drug_incidents"]
    df["white_incidents"] = df["white_incidents"] - df["white_drug_incidents"]
    df["black_incidents_p100k"] = (
        df["black_incidents"] / df["black_population"]) * 100000
    df["white_incidents_p100k"] = (
        df["white_incidents"] / df["white_population"]) * 100000
    return df[["FIPS", "year", "black_incidents", "white_incidents", "black_incidents_p100k", "white_incidents_p100k", "black_population", "white_population"]]


def main(arrests: bool):
    """Produce the data for NIBRS incidents with the number of drug incidents removed."""
    drug_df = drug_incidents(arrests=arrests)
    other_df = other_incidents(drug_df, arrests=arrests)
    other_df.to_csv(base_path / "output" /
                    f"other_incidents_2010-2020{'_arrests' if arrests else ''}.csv", index=False)


if __name__ == "__main__":
    main(arrests=True)
