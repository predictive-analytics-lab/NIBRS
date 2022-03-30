"""Script which generates a state level selection ratio map with a year slider."""
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from pathlib import Path
from urllib.request import urlopen
import json

import geopandas as gpd

from matplotlib import pyplot as plt
import matplotlib as mpl



data_path = Path(__file__).parents[3] / "data" / "output"
map_path = Path(__file__).parents[3] / "choropleths"

def county_map(df: pd.DataFrame):
    counties = gpd.read_file(data_path.parent / "misc" / "us_county_hs_only" / "us_county_hs_only.shp")
    breakpoint()
    counties["FIPS"] = (counties["STATE"] + counties["COUNTY"])
    counties = counties.merge(df, how='left', on='FIPS')
    counties["selection_ratio_log10"] = np.log(counties["selection_ratio"])
    ax, fig = plt.subplots(figsize=(20, 20))
    counties.boundary.plot(color="black", linewidth=2, ax=ax)
    counties.plot(column='selection_ratio_log10',
            cmap='OrRd',
            linewidth=0,
            ax=ax,
            legend=False, missing_kwds={
            "color": "black",
            "label": "Missing values",},)

    plt.savefig(map_path / "county_map.png")




if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    df_all = pd.read_csv(data_path / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_all.csv", dtype={'FIPS': str})
    county_map(df_all)