from typing import List, Tuple
import requests
import json
from itertools import chain
import asyncio
import time

from tqdm import tqdm
import pandas as pd
import numpy as np
import aiohttp
from pathlib import Path

API_KEYS = [
    "zZdfM3eL0UVPHE2ms4aXZMKNQbwVn96C5WMgWgOQ",
    "6nWZwnXHhdYYuzW01FhkshXWMbnqgkVh0YJJe177",
    "ymwJehx63qw8TlVu8Q1yYwcVQCdncEPyCaSyGgY0",
    "qRvIoIDMzlKQttI8ok2LSGidWeRqzwyyWOz3eC69",
    "pyqdzYnh1NbSO1OmbOQL9IZ4XY0uNiuE8TIsfpuN"
]


class ApiKeys:
    def __init__(self, api_keys: List[str], start_count: int = 0):
        self.api_keys = api_keys
        self.first_use_timestamp = time.time()
        self.current_key = 0
        self.num_keys = len(self.api_keys)
        self.counter = np.zeros(self.num_keys) + start_count

    def get_api_key(self, increment: int = 1) -> str:
        key_idx = self.current_key
        if self.counter[key_idx] + increment > 1000 or self.first_use_timestamp + 3600 < time.time():
            self.wait()
        self.current_key += 1
        if self.current_key == self.num_keys:
            self.current_key = 0
        self.counter[key_idx] += increment
        return self.api_keys[key_idx]

    def wait(self):
        next_use_timestamp = self.first_use_timestamp + 3900
        wait_time = next_use_timestamp-time.time()
        print(f"Waiting {wait_time} to request again")
        print(self.counter)
        if wait_time > 0:
            for i in tqdm(range(int(wait_time))):
                time.sleep(1)
        else:
            self.next_use_timestamp
        self.first_use_timestamp = next_use_timestamp
        self.counter[:] = 0
        return


api_keys = ApiKeys(API_KEYS, start_count=100)

data_dir = Path(__file__).parents[3] / "data"
data_path = data_dir / "UCR"
data_path.mkdir(exist_ok=True)

get_states_url = f"https://api.usa.gov/crime/fbi/master/api/states?api_key={api_keys.get_api_key()}&size=100"


def get_states() -> List[str]:
    r = requests.get(get_states_url)
    assert r.status_code == 200
    return [state["state_abbr"] for state in json.loads(r.content)["results"]]


def get_all_agencies(nibrs_only: bool) -> List[str]:
    if nibrs_only:
        nibrs_df = pd.read_csv(data_dir / "NIBRS" /
                               "raw" / "cannabis_allyears.csv")
        return list(nibrs_df.ori.dropna().unique())
    return list(chain(*[get_agencies(state_code) for state_code in get_states()]))


def get_agencies(state: str) -> str:
    url = f"https://api.usa.gov/crime/fbi/master/api/agencies/byStateAbbr/{state}?api_key={api_keys.get_api_key()}"
    r = requests.get(url)
    assert r.status_code == 200
    return [state["ori"] for state in json.loads(r.content)["results"]]


def get_agency_counts_url(ori: str, years: Tuple[int, int], offense: str, api_key: str):
    return f"https://api.usa.gov/crime/fbi/master/api/arrest/agencies/{ori}/{offense}/race/{years[0]}/{years[1]}?api_key={api_key}"


async def get_agency_counts(session, ori: str, api_key: str, years: Tuple[int, int] = (2017, 2020)) -> str:
    data = []
    async with session.get(get_agency_counts_url(ori, years, "dui", api_key)) as r:
        dui_json = await r.json()
    async with session.get(get_agency_counts_url(ori, years, "drug-possession-marijuana", api_key)) as r:
        mj_json = await r.json()
    async with session.get(get_agency_counts_url(ori, years, "drunkenness", api_key)) as r:
        drunk_json = await r.json()
    try:
        dui_df = pd.DataFrame(dui_json["data"])
        mj_df = pd.DataFrame(mj_json["data"])
        drunk_df = pd.DataFrame(drunk_json["data"])
    except KeyError as e:
        print(f"KeyError: {e}, on agency: {ori}, response body: {dui_json}")
        return data

    for year in range(years[0], years[1]+1):
        white_dui = get_count(dui_df, year, "White")
        black_dui = get_count(dui_df, year, "Black or African American")

        white_mj = get_count(mj_df, year, "White")
        black_mj = get_count(mj_df, year, "Black or African American")

        white_drunk = get_count(drunk_df, year, "White")
        black_drunk = get_count(drunk_df, year, "Black or African American")

        data.append((ori, year, white_dui, black_dui, white_mj,
                    black_mj, white_drunk, black_drunk))
    return data


def get_count(df, year, race):
    try:
        return int(df[(df["data_year"] == year) & (df["key"] == race)]["value"])
    except (TypeError, KeyError):
        return np.NaN


async def request_all(years: Tuple[int, int] = (2010, 2020)):
    agencies = get_all_agencies(nibrs_only=False)
    # agencies = ["AK0010100", "WY0150200"]
    all_agency_data = []
    batch_size = 30
    for i in tqdm(range(0, len(agencies), batch_size)):
        agency_subset = agencies[i:i+batch_size]
        api_key = api_keys.get_api_key(batch_size*3)  # 3 requests per agency
        try:
            async with aiohttp.ClientSession() as session:
                agency_subset_data = await asyncio.gather(*[get_agency_counts(session, agency, api_key, years=years) for agency in agency_subset])
            all_agency_data.extend(agency_subset_data)
        except Exception as e:
            print(e)
            break
        # async for agency in tqdm(agencies[:50]):
        #     data = await get_agency_counts(session, agency, years=years)
        #     agency_data.extend(data)
    df = pd.DataFrame(chain.from_iterable(all_agency_data), columns=[
        "ori",
        "year",
        "white_dui_arrests",
        "black_dui_arrests",
        "white_cannabis_arrests",
        "black_cannabis_arrests",
        "white_drunkenness_arrests",
        "black_drunkenness_arrests"
    ])
    return df

# enumerate states https://api.usa.gov/crime/fbi/master/api/states
# enumerate agencies per state https://api.usa.gov/crime/fbi/master/api/agencies/byStateAbbr/AL?API_KEY=iiHnOKfno2Mgkt5AynpvPpUQTEyxE77jo1RU8PIv
# DUI race data for ORI https://api.usa.gov/crime/fbi/master/api/arrest/agencies/AK0015200/dui/race/2010/2020?API_KEY=iiHnOKfno2Mgkt5AynpvPpUQTEyxE77jo1RU8PIv


async def test_request_all():
    async with aiohttp.ClientSession() as session:
        r = await get_agency_counts(session, "WY0150200", api_keys.get_api_key())
    return r

if __name__ == "__main__":
    df = asyncio.run(request_all((2010, 2020)))
    df.to_csv(data_path / "ucr_arrests.csv")
    # r = asyncio.run(test_request_all())
    # a = get_all_agencies(False)
