# %%
import pandas as pd

import plotly.express as px
from pathlib import Path

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

data_path = Path(__file__).parent.parent.parent / "data"
plot_path = Path(__file__).parent.parent.parent / "choropleths"

df = pd.read_csv(data_path / "output" / "agency_output_FIPS.csv", dtype={'FIPS': object}, index_col=0)
df = df[df.incidents >= 10]
# %%
df = df.round({'bwratio': 4, "selection_ratio": 4})


fig = px.choropleth(df, geojson=counties, locations='FIPS', color="selection_ratio",
                           color_continuous_scale="Viridis",
                           scope="usa",
                           range_color=(0, 10),
                           hover_data=["incidents", "bwratio"],
                           labels={"selection_ratio": 'Incident Ratios by Race (Black/White)',
                            'incidents': "Number of recorded Incidents",
                            "bwratio": "Black / White Ratio"},
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "county_selection_ratio.html")
# %%

df = pd.read_csv(data_path / "output" / "agency_output_State.csv", index_col=0)

state_abbr = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["STATE", "ABBRV"])
state_abbr = state_abbr.rename(columns={"STATE":"State"})

df = pd.merge(df, state_abbr, how='left', on="State")
# %%
df = df.round({'bwratio': 4, "selection_ratio": 4})

fig = px.choropleth(df, locationmode="USA-states", locations='ABBRV', color="selection_ratio",
                           color_continuous_scale="Viridis",
                           scope="usa",
                            hover_data=["incidents", "bwratio"],
                           labels={"selection_ratio": 'Incident Ratios by Race (Black/White)',
                            'incidents': "Number of recorded Incidents",
                            "bwratio": "Black / White Ratio"},
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.show()
fig.write_html(plot_path / "state_selection_ratio.html")
# %%
