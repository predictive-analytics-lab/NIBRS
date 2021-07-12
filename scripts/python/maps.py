# %%

import pandas as pd
from pathlib import Path
import numpy as np

data_path = Path(__file__).parent.parent.parent / "data"
plot_path = Path(__file__).parent.parent.parent / "choropleths"
agency_df = pd.read_csv(data_path / "misc" / "agency_participation.csv")

# %%
def agency_name(x):
    return f"{x.pub_agency_name}{x.pub_agency_unit} - {x.state_name}"

def countystate(x):
    return f"{x.state_name}:{x.county_name}"

agency_df["agency_name"] = agency_df.apply(agency_name, axis=1)
agency_df["countystate"] = agency_df.apply(countystate, axis=1)

# %%

columns = ["ori", "agency_name", "countystate", "data_year", "state_name", "pub_agency_name", "agency_status", "population", "county_name", "nibrs_participated", "participated"]

agency_df = agency_df[columns]
# %%
fips_ori = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t")[["ORI9", "FIPS"]]
fips_ori = fips_ori.rename(columns={"ORI9": "ori"})

agency_df = pd.merge(agency_df, fips_ori, on="ori")

def fips_fix(x):
    if int(x["FIPS"]) < 10_000:
        return f"0{x['FIPS']}"
    else:
        return str(x['FIPS'])

agency_df.FIPS = agency_df.apply(fips_fix, axis=1)

filtered_agencies = agency_df[agency_df.data_year == 2019]

p_agencies = filtered_agencies[filtered_agencies.nibrs_participated == "Y"]


# %%

# PROPORTION OF PARTICIPATION

county_participation = p_agencies.groupby("countystate").size() / filtered_agencies.groupby("countystate").size()
county_participation.iloc[:] = np.nan_to_num(county_participation.values)
county_participation = county_participation.reset_index()
county_participation = pd.merge(county_participation, agency_df, on="countystate", how="left")
county_participation = county_participation.groupby("countystate").first().reset_index()

# %%

county_participation_pop = p_agencies.groupby("countystate")["population"].sum() / filtered_agencies.groupby("countystate")["population"].sum()
county_participation_pop.iloc[:] = np.nan_to_num(county_participation_pop.values)
county_participation_pop = pd.merge(county_participation_pop, agency_df, on="countystate", how="left")
county_participation_pop = county_participation_pop.groupby("countystate").first().reset_index()

# %%

import plotly.express as px

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(county_participation, geojson=counties, locations='FIPS', color=0,
                           color_continuous_scale="Viridis",
                           range_color=(0, 1),
                           scope="usa",
                           labels={"0":'Proportion NIBRS'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "agency_participation.html")
# %%

fig = px.choropleth(county_participation_pop, geojson=counties, locations='FIPS', color="population_x",
                           color_continuous_scale="Viridis",
                           range_color=(0, 1),
                           scope="usa",
                           labels={"population_x":'Proportion by population NIBRS'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "agency_participation_pop.html")

# %%

