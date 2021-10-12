import pandas as pd
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[3] / "data"
plot_path = Path(__file__).parents[3] / "plots"

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
    "selection_ratio_log_poverty": "Dems + Poverty",
    "selection_ratio_log_dem_only": "Dems Only",
    "selection_ratio_log_urban": "Urban Only",
    "selection_ratio_log_buying": "Buying",
    "selection_ratio_log_buying_outside": "Buying in Public",
}

df["Model"] = df.Model.map(mapping)

id_cols = ["FIPS", "Model", "selection_ratio"]

value_cols = list(set(df.columns) - set(id_cols))

df = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="Property",
    value_name="Correlate Value",
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
df["Correlate Value"] = df["Correlate Value"].astype(float)
df = df.rename(columns={"selection_ratio": "Selection Ratio"})

sns.set(
    style="white", font_scale=1.2, rc={"text.usetex": True, "legend.loc": "upper left"}
)


df_poverty = df[df.Model == "Dems + Poverty"]


sns.lmplot(
    x="Correlate Value",
    y="Selection Ratio",
    col="Property",
    col_wrap=3,
    data=df_poverty,
    scatter_kws={"color": "black"},
    line_kws={"color": "red"},
)

plt.savefig(plot_path / "OA_regression_plots.pdf")
