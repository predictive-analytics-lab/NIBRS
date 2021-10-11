"""Script which generates a state level selection ratio map with a year slider."""
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly as py
from pathlib import Path
from urllib.request import urlopen
import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

data_path = Path(__file__).parents[3] / "data" / "output"
map_path = Path(__file__).parents[3] / "choropleths"

def map_with_slider(df: pd.DataFrame, time_col: str, value_col: str, log: bool = True,):
    subregion_counties = pd.read_csv(data_path.parent / "misc" / "subregion_counties.csv", index_col=0,  dtype={'FIPS': object})
    subregion_counties["state_region"] = subregion_counties["State"] + "-" + subregion_counties["Region"]

    df = pd.merge(df, subregion_counties, how='left', on="state_region")
    
    if log:
        df[value_col] = np.log10(df[value_col])

    data_slider = []
    for year in df[time_col].unique():
        df_segmented =  df[(df[time_col]== year)]

        data_each_yr = dict(
                            type='choropleth',
                            locations = df_segmented['FIPS'],
                            z=df_segmented[value_col].astype(float),
                            geojson=counties,
                            zmin=-1,
                            zmax=1,
                            colorscale = "RdBu",
                            customdata=df_segmented[["incidents", "bwratio", "state_region"]].values,
                            colorbar= {'title':'# Selection Ratio'},
                            hovertemplate="<br>".join([
                            "State sub-region: %{customdata[2]}",
                            "Selection Ratio (log10): %{z:.3f}",
                            "Incidents: %{customdata[0]}",
                            "Black-White Population Ratio: %{customdata[1]:.3f}",
                            ]),
                            marker_line_width=0)

        data_slider.append(data_each_yr)

    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='Year: {}'.format(i + np.min([int(x) for x in df[time_col].unique()])))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=len(steps) - 1, pad={"t": 1}, steps=steps)]

    layout = dict(title ='State sub-region level selection ratio', geo=dict(scope='usa',
                        projection={'type': 'albers usa'}),
                sliders=sliders)

    fig = dict(data=data_slider, layout=layout)


    fig = py.offline.plot(fig, filename=str(map_path / 'stateregion_map_with_slider.html'))


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    df = pd.read_csv(data_path / "selection_ratio_state_region_2016-2019.csv", dtype={'FIPS': object})
    map_with_slider(df, "year", "selection_ratio")