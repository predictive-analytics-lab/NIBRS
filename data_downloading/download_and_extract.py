import io
import requests
import zipfile
from pathlib import Path
from time import sleep

aws_url_pre_2016 = 'http://s3-us-gov-west-1.amazonaws.com/cg-d3f0433b-a53e-4934-8b94-c678aa2cbaf3'
aws_url_from_2016 = 'http://s3-us-gov-west-1.amazonaws.com/cg-d4b776d0-d898-4153-90c8-8336f86bdfec'

state_abbreviations = [s.split(',')[1].strip() for s in """Alabama, AL
    Alaska, AK
    Arizona, AZ
    Arkansas, AR
    California, CA
    Colorado, CO
    Connecticut, CT
    Delaware, DE
    Florida, FL
    Georgia, GA
    Hawaii, HI
    Idaho, ID
    Illinois, IL
    Indiana, IN
    Iowa, IA
    Kansas, KS
    Kentucky, KY
    Louisiana, LA
    Maine, ME
    Maryland, MD
    Massachusetts, MA
    Michigan, MI
    Minnesota, MN
    Mississippi, MS
    Missouri, MO
    Montana, MT
    Nebraska, NE
    Nevada, NV
    New Hampshire, NH
    New Jersey, NJ
    New Mexico, NM
    New York, NY
    North Carolina, NC
    North Dakota, ND
    Ohio, OH
    Oklahoma, OK
    Oregon, OR
    Pennsylvania, PA
    Rhode Island, RI
    South Carolina, SC
    South Dakota, SD
    Tennessee, TN
    Texas, TX
    Utah, UT
    Vermont, VT
    Virginia, VA
    Washington, WA
    West Virginia, WV
    Wisconsin, WI
    Wyoming, WY""".split("\n")]


def download_data():
    years = [2019]
    states = state_abbreviations
    download_location = Path(__file__).parent / 'downloads'
    for year in years:
        for state in states:
            if (download_location / f"{state}-{year}").exists():
                pass
            r = requests.get(get_download_url(state, year))
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(download_location / f"{state}-{year}")
            sleep(10)


def get_download_url(state: str, year: int) -> str:
    if year < 2016:
        url_base = aws_url_pre_2016
    else:
        url_base = aws_url_from_2016
    return f"{url_base}/{year}/{state}-{year}.zip"


if __name__ == "__main__":
    download_data()
