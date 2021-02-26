import io
import requests
import zipfile
from pathlib import Path
from time import sleep

from download_and_extract import state_abbreviations


base_url = "https://crime-data-explorer.app.cloud.gov/proxy/api/participation/states/"

def download_data():
    states = state_abbreviations
    download_location = Path(__file__).parent / 'downloads' / 'participation'
    download_location.mkdir(parents=True, exist_ok=True)
    for state in states:
        if (download_location / f"{state}.json").exists():
            pass
        r = requests.get(base_url + state)
        if r.status_code != 200:
            continue
        file_path = download_location / f"{state}.json"
        file_path.touch()
        file_path.write_bytes(r.content)

if __name__ == "__main__":
    download_data()
