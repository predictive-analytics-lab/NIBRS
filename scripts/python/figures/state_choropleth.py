"""Script which generates a state level selection ratio map with a year slider."""
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from pathlib import Path
from urllib.request import urlopen
import json
import geopandas as gpd

from matplotlib import pyplot as plt
import matplotlib as mpl



data_path = Path(__file__).parents[3] / "data" / "output"
map_path = Path(__file__).parents[3] / "choropleths"

def state_map(dfs: List[pd.DataFrame], names: List[str]):


    states = gpd.read_file(data_path.parent / "misc" / "usa-states-census-2014.shp", dtype={"NAME": str})
    states["state"] = states["NAME"]
    fig, axs = plt.subplots(2, 3, figsize=(80, 25))
    axs[1][2].set_visible(False)
    vmin = 100
    vmax = -100
    for df in dfs:
        df["selection_ratio_log10"] = np.log(df["selection_ratio"])
        if vmin > df["selection_ratio_log10"].min():
            vmin = df["selection_ratio_log10"].min()
        if vmax < df["selection_ratio_log10"].max():
            vmax = df["selection_ratio_log10"].max()
    for df, ax, name in zip(dfs, axs.flatten(), names):
        new_states = states.merge(df, how='left', on='state')
        new_states["selection_ratio_log10"] = np.log(new_states["selection_ratio"])
        new_states["selection_ratio_log10"] = new_states["selection_ratio_log10"].round(3)
        new_states.loc[new_states["black_incidents"] + new_states["white_incidents"] < 100, "selection_ratio_log10"] = np.nan
        #new_states.apply(lambda x: ax.annotate(text=x.selection_ratio_log10, xy=x.geometry.centroid.coords[0], ha='center', fontsize=14),axis=1)
        # outline
        # new_states.boundary.plot(color="black", linewidth=2, ax=ax)
        polygon = new_states.geometry.unary_union
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=new_states.crs)
        gdf.boundary.plot(color="black", linewidth=2, ax=ax)
        new_states.plot(column='selection_ratio_log10',
                cmap='OrRd',
                linewidth=0,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                legend=False, missing_kwds={
                "color": "black",
                "label": "Missing values",},)
        ax.set_title(name, fontsize=40)
        # remove the axis
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # tight 
    cmap = mpl.cm.get_cmap('OrRd')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.subplots_adjust(right=0.95)

    # add colorbar with labels fontsize 20
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical', label='Some Units')
    cbar_ax.tick_params(labelsize=40)
    # change cbar label fontsize
    cbar_ax.set_ylabel('Log Enforcement Ratio', fontsize=40)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(map_path / "state_map.png")




if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"

    df_meth = pd.read_csv(data_path / "selection_ratio_state_2017-2019_grouped_bootstraps_1000_meth.csv", dtype={'state': str})
    df_cannabis = pd.read_csv(data_path / "selection_ratio_state_2017-2019_grouped_bootstraps_1000.csv", dtype={'state': str})
    df_heroin = pd.read_csv(data_path / "selection_ratio_state_2017-2019_grouped_bootstraps_1000_heroin.csv", dtype={'state': str})
    df_crack = pd.read_csv(data_path / "selection_ratio_state_2017-2019_grouped_bootstraps_1000_crack.csv", dtype={'state': str})
    df_cocaine = pd.read_csv(data_path / "selection_ratio_state_2017-2019_grouped_bootstraps_1000_cocaine.csv", dtype={'state': str})

    dfs = [df_meth, df_cannabis, df_heroin, df_crack, df_cocaine]
    df_names = ["Methamphetamine", "Cannabis", "Heroin", "Crack", "Cocaine"]

    state_map(dfs, df_names)