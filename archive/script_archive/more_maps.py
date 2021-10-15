# %%
import pandas as pd
import numpy as np

import plotly.express as px
from pathlib import Path

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

data_path = Path(__file__).parent.parent.parent.parent / "data"
plot_path = Path(__file__).parent.parent.parent.parent / "choropleths"

df = pd.read_csv(data_path / "output" / "selection_ratio_county_2019.csv", dtype={'FIPS': object}, index_col=0)
# %%

df["selection_ratio_log10"] = np.log10(df["selection_ratio"])
df = df.round({'ci': 3, "selection_ratio": 3})

df["slci"] = df["selection_ratio"].astype(str) + " ± " + df["ci"].astype(str)

fig = px.choropleth(df, geojson=counties, locations='FIPS', color="selection_ratio_log10",
                           scope="usa",
                           range_color=(-1, 1),
                           hover_data=["slci", "incidents", "bwratio"],
                           labels={"slci": 'Selection Ratios by Race (Black/White) with CIs',
                            'incidents': "Number of recorded Incidents",
                            "bwratio": "Black / White Ratio"},
                           color_continuous_scale = "RdBu"
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "county_selection_ratio_cis.html")
# %%

df = pd.read_csv(data_path / "output" / "selection_ratio_state_2019.csv", index_col=0)
df["selection_ratio"] = np.log10(df["selection_ratio"])

state_abbr = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["STATE", "ABBRV"])
state_abbr = state_abbr.rename(columns={"STATE":"state"})

df = pd.merge(df, state_abbr, how='left', on="state")
# %%
df = df.round({'bwratio': 4, "selection_ratio": 4})

fig = px.choropleth(df, locationmode="USA-states", locations='ABBRV', color="selection_ratio",
                           scope="usa",
                            hover_data=["incidents", "bwratio"],
                           labels={"selection_ratio": 'Incident Ratios (log10) by Race (Black/White)',
                            'incidents': "Number of recorded Incidents",
                            "bwratio": "Black / White Ratio"},
                            range_color=(-1, 1),
                            color_continuous_scale = "RdBu"

                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.show()
fig.write_html(plot_path / "state_selection_ratio.html")# %%
# %%
df = pd.read_csv(data_path / "output" / "selection_ratio_state_region_2019.csv", index_col=0)
df["selection_ratio"] = np.log10(df["selection_ratio"])

subregion_counties = pd.read_csv(data_path / "misc" / "subregion_counties.csv", index_col=0,  dtype={'FIPS': object})
subregion_counties["state_region"] = subregion_counties["State"] + "-" + subregion_counties["Region"]

df = pd.merge(df, subregion_counties, how='left', on="state_region")

# %%
df = df.round({'bwratio': 4, "selection_ratio": 4})

fig = px.choropleth(df, geojson=counties, locations='FIPS', color="selection_ratio",
                           scope="usa",
                           range_color=(-1, 1),
                           hover_data=["incidents", "bwratio"],
                           labels={"selection_ratio": 'Incident Ratios  (log10) by Race (Black/White)',
                            'incidents': "Number of recorded Incidents",
                            "bwratio": "Black / White Ratio"},
                           color_continuous_scale = "RdBu"
                          )
fig.update_traces(marker_line_width=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
fig['layout']['geo']['subunitcolor']='rgba(0,0,0,0)'

fig.show()
fig.write_html(plot_path / "stateregion_selection_ratio.html")
# %%
