# %%
import pandas as pd
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[3] / "data"

df = pd.read_csv(data_path / "correlates" / "processed_correlates.csv", index_col=0)
df = df.drop(columns=[x for x in df.columns if x.startswith("var_log")])
value_cols = [x for x in df.columns if x.startswith("selection_ratio")]
id_cols = [x for x in df.columns if not x.startswith("selection_ratio")]

df = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="Model",
    value_name="selection_ratio",
)

mapping = {
    "selection_ratio_log10_poverty": "Dems + Poverty",
    "selection_ratio_log10_dem_only": "Dems Only",
    "selection_ratio_log10_urban": "Urban Only",
    "selection_ratio_log10_buying": "Buying",
    "selection_ratio_log10_buying_outside": "Buying in Public",
}

df["Model"] = df.Model.map(mapping)

id_cols = ["FIPS", "Model", "selection_ratio"]

value_cols = list(set(df.columns) - set(id_cols))

df = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="Property",
    value_name="Correlate_Value",
)

correlate_map = {
    "employment_county_ratio": "Employment rate at 35",
    "hsgrad_bw_ratio": "High school graduation rate",
    "employment_bw_ratio": "High school graduation rate B/W ratio",
    "collegegrad_county_ratio": "College graduation rate",
    "census_county_ratio": "Census Response rate",
    "density_county_ratio": "Population density",
    "income_county_ratio": "Income",
    "incarceration_bw_ratio": "Incarceration B/W ratio",
    "birthrate_county_ratio": "Teenage birth rate",
    "incarceration_county_ratio": "Incarceration",
    "hsgrad_county_ratio": "High school graduation rate",
    "perc_republican_votes": "\% Republican Vote Share",
    "birthrate_bw_ratio": "Teenage birth rate B/W ratio",
    "collegegrad_bw_ratio": "College graduation rate bw ratio",
    "income_bw_ratio": "Income bw ratio",
}

df = df[df["Property"].isin(correlate_map.keys())]
df["Property"] = df["Property"].map(correlate_map)
df.Correlate_Value = df.Correlate_Value.astype(float)

# %%

g = sns.FacetGrid(df, col="Property", col_wrap=4, hue="Model", sharex=False)

# g.map(
#     sns.scatterplot,
#     "Correlate_Value",
#     "selection_ratio",
#     palette="colorblind",
#     alpha=0.5,
#     size=0.1
# )

g.map(
    sns.kdeplot,
    "Correlate_Value",
    "selection_ratio",
    palette="colorblind",
    alpha=0.5,
    fill=True,
    levels=10,
)

g.axes[5].set_xlim([0.75, 1.25])
g.axes[9].set_xlim([0, 25])
g.axes[12].set_xlim([0.8, 1.2])

g.add_legend()

plt.show()
# %%

df_poverty = df[df.Model == "Dems + Poverty"]

g = sns.FacetGrid(df_poverty, col="Property", col_wrap=4)

g.map(
    sns.scatterplot,
    "Correlate_Value",
    "selection_ratio",
    palette="colorblind",
    alpha=0.5,
    size=0.1,
)


g.add_legend()

plt.show()
# %%
