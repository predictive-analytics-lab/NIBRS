"""
Script 
"""
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "data"

census = pd.read_csv(data_path / "demographics" / "county_census.csv", index_col=0)
agency = pd.read_csv(data_path / "output" / "agency_lemas.csv", index_col=0, usecols=["FIPS", "ori", "population"]).reset_index()


county_population = census.groupby("FIPS")["frequency"].sum().to_frame("population_c").reset_index()
populations = pd.merge(agency, county_population, how="left", on="FIPS")
populations["ratio"] = populations.population / populations.population_c


