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

def map_with_slider(df: pd.DataFrame, time_col: str, value_col: str, log: bool = True,):

    if log:
        df[value_col] = np.log10(df[value_col])
    else:
        df[value_col] = df[value_col] * 100_000

    # data_slider = []
    # for year in df[time_col].unique():
    #     df_segmented =  df[(df[time_col]== year)]

    #     data_each_yr = dict(
    #                         type='Choroplethmapbox',
    #                         locations = df_segmented['FIPS'],
    #                         z=df_segmented[value_col].astype(float),
    #                         geojson=counties,
    #                         zmin=-1,
    #                         zmax=1,
    #                         colorscale = "RdBu",
    #                         customdata=df_segmented[["incidents", "bwratio", "FIPS", "frequency", "urban_code"]].values,
    #                         colorbar= {'title':'# Selection Ratio'},
    #                         hovertemplate="<br>".join([
    #                         "FIPS: %{customdata[2]}",
    #                         "Selection Ratio (log10): %{z:.3f}",
    #                         "Incidents: %{customdata[0]}",
    #                         "Black-White Population Ratio: %{customdata[1]:.3f}",
    #                         "Population: %{customdata[3]:.3f}",
    #                         "Urban Code: %{customdata[4]:.3f}"
    #                         ]))

    #     data_slider.append(data_each_yr)

    # steps = []
    # for i in range(len(data_slider)):
    #     step = dict(method='restyle',
    #                 args=['visible', [False] * len(data_slider)],
    #                 label='Year: {}'.format(i + np.min([int(x) for x in df[time_col].unique()])))
    #     step['args'][1][i] = True
    #     steps.append(step)

    # sliders = [dict(active=len(steps) - 1, pad={"t": 1}, steps=steps)]

    # layout = dict(title ='County level selection ratio', geo=dict(scope='usa',
    #                     projection={'type': 'albers usa'}),
    #             sliders=sliders)

    # fig = dict(data=data_slider, layout=layout)
    
    # max_count = df['count'].max()
    df["frequency"] = df["frequency"].apply(lambda x: f'{int(x):,}')
    df["bwratio"] = df["bwratio"].apply(lambda x: f'{x:.3f}')
    # df["selection_ratio"] = df["selection_ratio"].apply(lambda x: f'{x:.3f}')

    fig = px.choropleth_mapbox(
        df, 
        geojson=counties,
        locations='FIPS', 
        color=value_col,
        color_continuous_scale="Viridis",
        range_color=(-1, 1),
        mapbox_style="carto-positron",
        opacity=0.5,
        center = {'lat': 42.967243, 'lon': -101.271556},
        hover_data=["incidents", "frequency", "urban_code", "bwratio"],
        labels={
            "year": "Year",
            "selection_ratio": 'Incident Ratios  (log10) by Race (Black/White)',
            'incidents': "Number of recorded Incidents",
            "bwratio": "Black / White Ratio",
            "urban_code": "Urban Code",
            "frequency": "Population"},
        zoom=3.7,
        animation_frame=time_col
    )
    fig.update_geos(fitbounds="locations",visible=False, scope="usa")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},)
    fig.write_html(str(map_path / 'county_ratio.html'))


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    df = pd.read_csv(data_path / "selection_ratio_county_2016-2019_smoothed.csv", dtype={'FIPS': object})
    map_with_slider(df, "year", "selection_ratio", log=True)