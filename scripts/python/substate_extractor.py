import numpy as np
from Levenshtein import jaro_winkler
from bs4 import BeautifulSoup
from pathlib import Path
import re
import unidecode
import pandas as pd

misc_path = Path(__file__).parent.parent.parent / "data" / "misc"

with open(misc_path / 'substates.htm', 'r') as f:
    content = f.read()
    soup = BeautifulSoup(content)

data = []

current_state = None

for note in soup.find("div", {"class": "endnotes"}).findAll("p", {"class": "f90"}):
    if note.find("span"):
        text = note.getText()
        current_state = re.search(
            'map of (.+?) showing substate', text).group(1)
    if note.find("a"):
        continue
    else:
        text = unidecode.unidecode(note.getText())
        search = re.search(
            'referred to as (.+?) is made up of the following', text)
        if search:
            counties = text.split(":")[-1].split(",")
            if len(counties) == 1:
                counties = counties[0].split(" and ")
            for county in counties:
                county = county.strip()
                if county[-1] == ".":
                    county = county.replace("and ", "").replace(".", "")
                data.append(
                    [current_state, search.group(1).split(":")[0], county])

df = pd.DataFrame(data, columns=["State", "Region", "County"])
df = pd.concat([df, pd.read_csv(misc_path / "manual_counties.csv")])


fips = pd.read_csv(misc_path / "county_fips_master.csv", encoding="ISO-8859-1")

found_fips = []
closest_counties = []


def fips_fix(x):
    if x < 10_000:
        return f"0{x}"
    else:
        return str(x)


def clean_name(x):
    x = x.lower()
    x = x.replace("census area", "").replace("borough", "").replace(
        "county", "").replace(" ", "").strip()
    return x


for i, row in df.iterrows():
    fips_state = fips[fips.state_name == row["State"]]
    counties = fips_state.county_name.values
    closest = counties[np.argmax(
        [jaro_winkler(clean_name(str(row["County"])), clean_name(c)) for c in counties])]
    matched_fips = fips_state[fips_state.county_name ==
                              closest]["fips"].values[0]
    found_fips.append(fips_fix(matched_fips))
    closest_counties.append(closest)

df["Matched County"] = closest_counties
df["FIPS"] = found_fips

df.to_csv(misc_path / "subregion_counties.csv")
