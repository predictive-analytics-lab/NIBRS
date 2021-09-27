# %%
from geopandas import GeoDataFrame

from pathlib import Path

data_path = Path(__file__).parents[3] / "data" / "misc"

gdf = GeoDataFrame.from_file(data_path / "us-county-boundaries.geojson")

gdf = gdf[["geoid", "intptlon", "intptlat"]]
# %%

gdf = gdf.rename(columns={"geoid": "FIPS", "intptlon": "lon", "intptlat": "lat"})
gdf["lon"] = gdf.lon.astype(float)
gdf["lat"] = gdf.lat.astype(float)

gdf.to_csv(data_path / "us-county-boundaries-latlong.csv", index=False)
# %%
