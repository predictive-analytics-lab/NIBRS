"""Script which generates a state level selection ratio map with a year slider."""
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly as py
import plotly.express as px

from pathlib import Path
from urllib.request import urlopen
import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

data_path = Path(__file__).parents[3] / "data" / "output"
map_path = Path(__file__).parents[3] / "choropleths"

def confidence_categorization(df: pd.DataFrame, value_col: str, ci_col: str) -> pd.DataFrame:
    def _categorization(v, ci):
        if v - ci > 5:
            return "S>5"
        if v - ci > 2:
            return "S>2"
        if v - ci > 1:
            return "S>1"
        if v + ci < 1:
            return "S<1"
        if v + ci < 0.5:
            return "S<0.5"
        if v + ci < 0.2:
            return "S<0.2"
        return "Low confidence"
    df["cat"] = df.apply(lambda x: _categorization(x[value_col], x[ci_col]), axis=1)
    return df

def confidence_map(df: pd.DataFrame, time_col: str):

    #df = confidence_categorization(df, "selection_ratio", "ci")
    
    df["frequency"] = df["frequency"].apply(lambda x: f'{int(x):,}')
    df["bwratio"] = df["bwratio"].apply(lambda x: f'{x:.3f}')
    
    df = df.round({'ci': 3, "selection_ratio": 3})
    df["slci"] = df["selection_ratio"].astype(str) + " ± " + df["ci"].astype(str)
    
    color_map = {"S>5":"#E76258",
                 "S>2":"#EAB055",
                 "S>1":"#E0D987",
                 "S<1":"#5E925F",
                 "S<0.5":"#265F47",
                 "S<0.2":"#52675B",
                 "S~1":"#689891"}

    fig = px.choropleth_mapbox(
        df, 
        geojson=counties,
        locations='FIPS', 
        color="cat",
        color_discrete_map=color_map,
        mapbox_style="carto-positron",
        opacity=0.5,
        center = {'lat': 42.967243, 'lon': -101.271556},
        hover_data=["slci", "cat", "incidents", "frequency", "urban_code", "bwratio"],
        labels={
            "year": "Year",
            "cat": '95% Confidence S>X',
            "slci": 'Selection Ratios by Race (Black/White) with CIs',
            'incidents': "Number of recorded Incidents",
            "bwratio": "Black / White Ratio",
            "urban_code": "Urban Code",
            "frequency": "Population"},
        zoom=3.7,
        animation_frame=time_col
    )
    fig.update_geos(fitbounds="locations",visible=False, scope="usa")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},)
    fig.write_html(str(map_path / 'county_ratio_2019_no_poverty_conf.html'))

def map_with_slider(df: pd.DataFrame, time_col: str, drug: str):

    # if log:
    #     df[value_col] = np.log10(df[value_col])
    # else:
    #     df[value_col] = df[value_col] * 100_000

    df["selection_ratio_log10"] = np.log10(df["selection_ratio"])
    df = df.round({'ci': 3, "selection_ratio": 3, 'ub': 3, 'lb': 3})
    df["slci"] = df["lb"].astype(str) + " - " + df["ub"].astype(str)
    
    df["frequency"] = df["frequency"].apply(lambda x: f'{int(x):,}')
    df["bwratio"] = df["bwratio"].apply(lambda x: f'{x:.3f}')

    df["incidents"] = df["white_incidents"] + df["black_incidents"]
    df = df[df["incidents"] >= 30]

    fig = px.choropleth_mapbox(
        df, 
        geojson=counties,
        locations='FIPS', 
        color="selection_ratio_log10",
        color_continuous_scale="Viridis",
        range_color=(-2, 2),
        mapbox_style="carto-positron",
        opacity=0.5,
        center = {'lat': 42.967243, 'lon': -101.271556},
        hover_data=["slci", "incidents", "frequency", "urban_code", "bwratio"],
        labels={
            "year": "Year",
            "slci": 'Selection Ratios Range',
            'incidents': "Number of recorded Incidents",
            "bwratio": "Black / White Ratio",
            "urban_code": "Urban Code",
            "frequency": "Population"},
        zoom=3.7,
        animation_frame=time_col
    )
    fig.update_geos(fitbounds="locations",visible=False, scope="usa")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},)
    fig.write_html(str(map_path / f'county_ratio_2017_2019_{drug}.html'))

def usage_map(df: pd.DataFrame, time_col: str):

    
    df["usage_ratio"] = df["black_users"] + df["white_users"]
    df["usage_norm"] = df["usage_ratio"] / (df["black"] + df["white"])

    fig = px.choropleth_mapbox(
        df, 
        geojson=counties,
        locations='FIPS', 
        color="usage_norm",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        opacity=0.5,
        center = {'lat': 42.967243, 'lon': -101.271556},
        hover_data=["year", "usage_norm", "bwratio", "frequency"],
        labels={
            "year": "Year",
            "usage_norm": 'Usage Est.',
            "bwratio": "Black / White Ratio",
            "frequency": "Population"},
        zoom=3.7,
        animation_frame=time_col
    )
    fig.update_geos(fitbounds="locations",visible=False, scope="usa")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},)
    fig.write_html(str(map_path / 'crack_usage_ratio.html'))



if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    df = pd.read_csv(data_path / "selection_ratio_county_2017-2019_grouped_bootstraps_1000_poverty.csv", dtype={'FIPS': object})
    map_with_slider(df, "year", "cannabis")
    # usage_map(df, "year")
    # confidence_map(df, "year")