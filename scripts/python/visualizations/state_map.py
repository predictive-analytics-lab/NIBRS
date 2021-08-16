"""Script which generates a state level selection ratio map with a year slider."""
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly as py
from pathlib import Path

data_path = Path(__file__).parents[3] / "data" / "output"
map_path = Path(__file__).parents[3] / "choropleths"

def map_with_slider(df: pd.DataFrame, time_col: str, value_col: str, log: bool = True,):

    state_abbr = pd.read_csv(data_path.parent / "misc" / "FIPS_ABBRV.csv", usecols=["STATE", "ABBRV"])
    state_abbr = state_abbr.rename(columns={"STATE":"state"})

    df = pd.merge(df, state_abbr, how='left', on="state")

    if log:
        df[value_col] = np.log10(df[value_col])

    data_slider = []
    for year in df[time_col].unique():
        df_segmented =  df[(df[time_col]== year)]

        data_each_yr = dict(
                            type='choropleth',
                            locations = df_segmented['ABBRV'],
                            z=df_segmented[value_col].astype(float),
                            locationmode='USA-states',
                            zmin=-1,
                            zmax=1,
                            colorscale = "RdBu",
                            colorbar= {'title':'# Selection Ratio'})

        data_slider.append(data_each_yr)

    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='Year: {}'.format(i + np.min([int(x) for x in df[time_col].unique()])))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

    layout = dict(title ='State level selection ratio', geo=dict(scope='usa',
                        projection={'type': 'albers usa'}),
                sliders=sliders)

    fig = dict(data=data_slider, layout=layout)

    py.offline.plot(fig, filename=str(map_path / 'state_map_with_slider.html'))


if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    df = pd.read_csv(data_path / "selection_ratio_state_2017-2019.csv")
    map_with_slider(df, "year", "selection_ratio")