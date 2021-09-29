# %%
from geopandas import GeoDataFrame
from pathlib import Path
from timezonefinder import TimezoneFinder

data_path = Path(__file__).parents[3] / "data" / "misc"

gdf = GeoDataFrame.from_file(data_path / "us-county-boundaries.geojson")

gdf = gdf[["geoid", "intptlon", "intptlat"]]
# %%

gdf = gdf.rename(columns={"geoid": "FIPS", "intptlon": "lon", "intptlat": "lat"})
gdf["lon"] = gdf.lon.astype(float)
gdf["lat"] = gdf.lat.astype(float)


def get_timezone(latitude: float, longitude: float):
    tzf = TimezoneFinder()
    timezone = tzf.timezone_at(lat=latitude, lng=longitude)
    return timezone


gdf["timezone"] = gdf.apply(lambda row: get_timezone(row.lat, row.lon), axis=1)

gdf.to_csv(data_path / "us-county-boundaries-latlong.csv", index=False)
# %%
