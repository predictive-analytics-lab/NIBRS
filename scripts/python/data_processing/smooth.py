from typing import Callable, List, Tuple
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
import numpy as np

from pathlib import Path

data_path = Path(__file__).parents[3] / "data"


def join_state_with_counties(
    df: pd.DataFrame, county_shp: gpd.GeoDataFrame, state: str
) -> gpd.GeoDataFrame:
    county_shp = county_shp[county_shp["state_name"] == state]
    county_shp.rename(columns={"geoid": "FIPS"}, inplace=True)
    county_shp = county_shp.merge(df, on="FIPS", how="left")
    return county_shp.reset_index()


def smooth_data(
    nibrs_df: pd.DataFrame,
    census_df: pd.DataFrame,
    metro: bool = False,
    poverty: bool = False,
    urban_filter: int = 2,
    smoothing_param: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    county_shp = gpd.read_file(data_path / "misc" / "us-county-boundaries.geojson")
    smoothed_nibrs = pd.DataFrame()
    smoothed_census = pd.DataFrame()
    dem_vars = ["age", "sex", "race"]
    dem_vars_c = dem_vars.copy()
    if poverty:
        dem_vars_c += ["poverty"]
    if metro:
        dem_vars_c += ["metrocounty"]
    for state in nibrs_df.state.unique():
        state_idf = nibrs_df[nibrs_df.state == state]
        state_cdf = census_df[census_df.state == state]
        state_gdf = county_shp[county_shp.state_name == state]

        state_sm_idf = smooth_state(
            state_idf,
            state_gdf,
            state,
            dem_vars,
            "incidents",
            urban_filter=urban_filter,
            smoothing_param=smoothing_param,
        )
        state_sm_cdf = smooth_state(
            state_cdf,
            state_gdf,
            state,
            dem_vars_c,
            "frequency",
            urban_filter=urban_filter,
            smoothing_param=smoothing_param,
        )

        smoothed_nibrs = smoothed_nibrs.append(state_sm_idf)
        smoothed_census = smoothed_census.append(state_sm_cdf)
    return smoothed_nibrs, smoothed_census


def reporting(state_df: pd.DataFrame, year: int) -> pd.DataFrame:
    agency_df = pd.read_csv(
        data_path / "misc" / "agency_participation.csv",
        usecols=["ori", "nibrs_participated", "data_year"],
    )
    agency_df = agency_df[agency_df.data_year == 2019]
    fips_ori_df = pd.read_csv(
        data_path / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "FIPS"],
        dtype={"FIPS": object},
    )
    fips_ori_df = fips_ori_df.rename(columns={"ORI9": "ori"})
    agency_df = pd.merge(agency_df, fips_ori_df, on="ori")
    reporting = (
        agency_df.groupby("FIPS")
        .nibrs_participated.apply(lambda x: "Y" if any(x == "Y") else "N")
        .to_frame("reporting")
        .reset_index()
    )
    return state_df.merge(reporting, how="left", on="FIPS")


def smooth_state(
    state_df: pd.DataFrame,
    county_gdf: gpd.GeoDataFrame,
    state: str,
    dem_vars: List[str],
    value_var: str,
    urban_filter: int = 2,
    smoothing_param: int = 1,
) -> pd.DataFrame:
    # If only one county in the state, just return (why smooth one value).
    if len(state_df.FIPS.unique()) == 1:
        return state_df

    state_df, urban_df = filter_urban(state_df, urban_filter)

    if len(state_df) == 0:
        return urban_df

    year = state_df.year.unique()[0]

    locations = state_df[["state", "state_region", "FIPS"]].drop_duplicates()

    state_df_p = state_df.pivot_table(
        index=["FIPS"], columns=dem_vars, values=value_var
    )

    # Left join s.t. all other counties are added back in.
    # So when we calculate adjacencies we use counties that are reporting (not just that have rows)
    state_gdf_p = join_state_with_counties(state_df_p, county_gdf, state).sort_values(
        by=["FIPS"]
    )

    # Calculate adjacencies
    qW = Queen.from_dataframe(state_gdf_p)
    amat, _ = qW.full()
    county_weights = get_county_weights(
        amat,
        distance_weighting=lambda x, y: 0 if x == 0 else 1 / (y + 1) ** smoothing_param,
    )

    # Get the indices of the counties that are reporting (minus those that are urban)
    state_gdf_p = reporting(state_gdf_p, year)
    state_gdf_p.loc[state_gdf_p.FIPS.isin(urban_df.FIPS.unique()), "reporting"] = "N"
    indicies = np.nonzero((state_gdf_p.reporting == "Y").values)[0]

    # Filter Urban DF
    urban_df = reporting(urban_df, year)
    urban_df = urban_df[urban_df["reporting"] == "Y"]

    # Filter out the counties that are urban or not reporting
    state_gdf_p = state_gdf_p.iloc[indicies, :]
    # Fill any N/As with 0 Incidents
    state_gdf_p = state_gdf_p.fillna(0)

    if len(state_gdf_p) == 0:
        return urban_df

    # Filter the adjacencies to only those counties that are reporting
    county_weights = county_weights[np.ix_(indicies, indicies)]
    # Matmul: the county weights matrix (C x C) with the demographic incident matrix (C x D) where D is the number of demographic combinations.
    state_gdf_p = state_gdf_p.drop(["reporting"], axis=1)
    state_gdf_p.iloc[:, 22:] = county_weights @ state_gdf_p.iloc[:, 22:].values

    # Melt the dataframe to get the demographic columns back. wide -> long
    state_gdf_p = state_gdf_p[["FIPS", *state_gdf_p.columns[22:].values]].melt(
        id_vars=["FIPS"], value_name=value_var
    )

    dem_df = pd.DataFrame(state_gdf_p["variable"].tolist(), columns=dem_vars)

    state_gdf_p = state_gdf_p.join(dem_df)

    # Clean and merge
    state_gdf_p.drop(["variable"], axis=1, inplace=True)
    state_gdf_p = state_gdf_p.merge(locations, on="FIPS", how="inner")

    return state_gdf_p.append(urban_df).reset_index()


def get_county_weights(
    state_amat: np.ndarray,
    max_path_length: int = 5,
    distance_weighting: Callable[[int], float] = lambda x, y: 0.0
    if x == 0
    else 1 / (y + 1),
) -> np.ndarray:
    vfunc = np.vectorize(distance_weighting)
    new_bool_amat = state_amat.copy()
    new_weighted_amat = vfunc(new_bool_amat, 1).astype(float)
    for path_length in range(2, max_path_length + 1):
        paths = (np.linalg.matrix_power(state_amat, path_length) > 0).astype(int)
        added_paths = ((paths - new_bool_amat) > 0).astype(int)
        new_bool_amat += added_paths
        new_weighted_amat += vfunc(added_paths, path_length)
    np.fill_diagonal(new_weighted_amat, 1)
    return new_weighted_amat


def filter_urban(
    df: pd.DataFrame,
    urban_level: int,
    coverage_required: float = 0.0,
    min_incidents: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    year = df.year.unique()[0]
    urban_codes = pd.read_csv(
        data_path / "misc" / "NCHSURCodes2013.csv", usecols=["FIPS code", "2013 code"]
    )
    urban_codes.rename(
        columns={"FIPS code": "FIPS", "2013 code": "urban_code"}, inplace=True
    )
    urban_codes["FIPS"] = urban_codes.FIPS.apply(lambda x: str(x).rjust(5, "0"))
    df = pd.merge(df, urban_codes, on="FIPS", how="left")
    coverage = pd.read_csv(
        data_path / "misc" / "county_coverage.csv",
        dtype=str,
        usecols=["FIPS", "coverage", "year"],
    )
    coverage = coverage[coverage.year == str(year)]
    coverage = coverage.drop(["year"], axis=1)
    df = pd.merge(df, coverage, on="FIPS", how="left")
    if "incidents" in df.columns:
        min_df = (
            (df.groupby("FIPS").incidents.sum() >= min_incidents)
            .to_frame("min_threshold")
            .reset_index()
        )
        df = pd.merge(df, min_df, on="FIPS", how="left")
    else:
        df["min_threshold"] = True
    condition = (
        (df.urban_code <= urban_level)
        & (df.coverage.astype(float) > coverage_required)
        & (df.min_threshold == True)
    )
    df = df.drop(columns=["urban_code"])
    return df[~condition], df[condition]
