# %%
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


data_path = Path(__file__).parent.parent.parent / "data"
plot_path = Path(__file__).parent.parent.parent / "choropleths"
nibrs_agency = pd.read_csv(data_path / "NIBRS" / "cannabis_agency_2019_20210608.csv")
census = pd.read_csv(data_path / "demographics" / "census-2019-fips.csv", usecols=["FIPS", "POPESTIMATE2019"])

# %%
fips_ori = pd.read_csv(data_path / "misc" / "LEAIC.tsv", delimiter="\t", usecols=["ORI9", "FIPS"])
fips_ori = fips_ori.rename(columns={"ORI9": "ori"})

nibrs_agency = pd.merge(nibrs_agency, fips_ori, on="ori")

def fips_fix(x):
    if int(x["FIPS"]) < 10_000:
        return f"0{x['FIPS']}"
    else:
        return str(x['FIPS'])

nibrs_agency.FIPS = nibrs_agency.apply(fips_fix, axis=1)
census.FIPS = census.apply(fips_fix, axis=1)
# %%

fips_incident_count = nibrs_agency.groupby("FIPS").size().reset_index()
fips_incident_count = fips_incident_count.rename(columns={0: "Count"})
fips_incident_count = pd.merge(fips_incident_count, census, on="FIPS", how="left")
fips_incident_count["IncidentPer100K"] = (fips_incident_count["Count"] * 100_000) / fips_incident_count["POPESTIMATE2019"]
# fips_incident_count = fips_incident_count[fips_incident_count["POPESTIMATE2019"] >= 1000]
# %%

fig = px.choropleth(fips_incident_count, geojson=counties, locations='FIPS', color="Count",
                           color_continuous_scale="Viridis",
                           range_color=(0, fips_incident_count.Count.max()),
                           scope="usa",
                           labels={"count":'Incident Count'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "nibrs_incident_count.html")
# %%

fig = px.choropleth(fips_incident_count, geojson=counties, locations='FIPS', color="IncidentPer100K",
                           color_continuous_scale="Viridis",
                           range_color=(0, fips_incident_count.IncidentPer100K.max()),
                           scope="usa",
                           labels={"IncidentPer100K":'Incidents Per 100K'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

fig.write_html(plot_path / "nibrs_incident_per100k.html")
# %%

### HISTOGRAMS

fips_incident_count.plot(x="POPESTIMATE2019", y="IncidentPer100K",kind="scatter")
# %%

