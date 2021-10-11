"""
Script which loads the coverage data for each agency and checks that the coverage is valid.

Specifically:
- Determine expected population for each county
- Ensure that agency population coverage is less than the expected population.

In cases which this is not true, it is due to the agency covering an area larger than the county, despite what the data reports.

It should be true in few cases.
"""

import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent.parent / "data"

census = pd.read_csv(data_path / "census" / "census_processed_2019.csv", index_col=0)
agency = pd.read_csv(
    data_path / "output" / "agency_lemas.csv",
    index_col=0,
    usecols=["FIPS", "ori", "population"],
).reset_index()


county_population = (
    census.groupby("FIPS")["frequency"].sum().to_frame("population_c").reset_index()
)
populations = pd.merge(agency, county_population, how="left", on="FIPS")
populations["ratio"] = populations.population / populations.population_c

print(populations[populations.ratio > 1].sort_values("ratio", ascending=False))
