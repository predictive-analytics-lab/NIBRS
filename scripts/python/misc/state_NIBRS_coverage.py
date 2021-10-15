"""This script contains functionality to calculate the NIBRS coverage proportion (by agencies or population)."""

import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


def load_and_join_data() -> pd.DataFrame:
    data_path = Path(__file__).parents[3] / "data"

    ap_cols = [
        "ori",
        "nibrs_participated",
        "county_name",
        "population",
        "suburban_area_flag",
        "male_officer",
        "female_officer",
        "data_year",
    ]

    # LOAD AGENCY DATAFRAME

    agency_df = pd.read_csv(
        data_path / "misc" / "agency_participation.csv", usecols=ap_cols
    )
    agency_df = agency_df[agency_df.data_year == 2019]

    # LOAD FIPS <-> ORI DATAFRAME

    fips_ori_df = pd.read_csv(
        data_path / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "FIPS"],
        dtype={"FIPS": object},
    )

    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})

    # JOIN THE TWO DATAFRAMES

    agency_df = pd.merge(agency_df, fips_ori_df, on="ori")

    # LOAD SUBREGION DATAFRAME

    subregion_df = pd.read_csv(
        data_path / "misc" / "subregion_counties.csv",
        dtype={"FIPS": object},
        usecols=["State", "Region", "FIPS"],
    )

    # JOIN THE DATAFRAMES - Our agencies now have FIPS (county) and subregion information.

    agency_df = pd.merge(agency_df, subregion_df, on="FIPS")

    agency_df["state_region"] = agency_df["State"] + " : " + agency_df["Region"]

    # LOAD LEMAS DATAFRAME

    lemas_df = pd.read_csv(
        data_path / "agency" / "lemas_processed.csv", usecols=["FIPS", "PERS_EDU_MIN"]
    )

    lemas_df["FIPS"] = lemas_df.FIPS.apply(lambda x: str(x).rjust(5, "0"))

    # JOIN AND RETURN DATAFRAME THAT CONTAINS: Agency, FIPS, subregion, and LEMAS information.

    return pd.merge(agency_df, lemas_df, on="FIPS", how="left")


def coverage(
    df: pd.DataFrame, resolution: str, nibrs: bool = True, lemas: bool = False
):
    """
    Function which calculates the coverage of each agency at a particular resolution.

    :param df: DataFrame containing the data to be analyzed.
    :param resolution: The geographic resolution at which the data is to be aggregated.
    :param nibrs: Boolean indicating whether or not to only calculate coverage of NIBRS data.
    :param lemas: Boolean indicating whether or not to only calculate coverage of LEMAS data.
    """
    agencies = df.groupby(resolution).size()

    df_c = df.copy()

    if lemas:
        df_c = df_c[~df_c["PERS_EDU_MIN"].isnull()]

    if nibrs:
        df_c = df_c[df_c["nibrs_participated"] == "Y"]

    conditioned_agencies = df_c.groupby(resolution).size()

    population_covered = (
        df_c.groupby(resolution).population.sum()
        / df.groupby(resolution).population.sum()
    )
    population_covered = population_covered.fillna(0)
    reporting_proportion = conditioned_agencies / agencies
    reporting_proportion = reporting_proportion.fillna(0)

    return reporting_proportion, population_covered


def print_coverage(
    agency_proportion: pd.DataFrame, population_proportion: pd.DataFrame
) -> None:
    """Print the two dataframes in a clean manner.

    :param agency_proportion (pd.DataFrame): The Proportion of agency coverage.
    :param population_proportion (pd.DataFrame): The Proportion of agency coverage by population.
    """
    print("\n\n")
    print(f"{'Agency':<10} {'Coverage':>10}")
    print("-" * 30)
    for ag_name, ag_val in agency_proportion.items():
        print(f"{ag_name:<10} {round(ag_val*100, 3):>10}")
    print("-" * 30)
    print(f"{'Population':<10} {'Coverage':>10}")
    print("-" * 30)
    for pop_name, pop_val in population_proportion.items():
        print(f"{pop_name:<10} {round(pop_val*100, 3):>10}")
    print("-" * 30)


if __name__ == "__main__":
    agency_df = load_and_join_data()

    agency_prop, pop_prop = coverage(agency_df, "State", nibrs=True, lemas=True)
    agency_prop_lf, pop_prop_lf = coverage(agency_df, "State", nibrs=True, lemas=False)

    print_coverage(agency_prop, pop_prop)
    print_coverage(agency_prop_lf, pop_prop_lf)
