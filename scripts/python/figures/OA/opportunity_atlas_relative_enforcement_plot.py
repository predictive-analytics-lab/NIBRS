import pandas as pd
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

data_path = Path(__file__).parents[4] / "data"
plot_path = Path(__file__).parents[4] / "plots"

df = pd.read_csv(data_path / "correlates" / "OA_RE_other_drugs.csv", index_col=0)

id_cols = ["FIPS", "drug", "relative_enforcement"]
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
df = df.rename(columns={"relative_enforcement": "Relative Enforcement"})

sns.set(
    style="white", font_scale=1.2, rc={ "legend.loc": "upper left"}
)


for drug in ["cannabis", "cocaine", "heroin", "crack", "meth"]:
    df_drug = df[df.drug == drug]


    sns.lmplot(
        x="Correlate Value",
        y="Relative Enforcement",
        col="Property",
        col_wrap=3,
        data=df_drug,
        scatter_kws={"color": "black"},
        line_kws={"color": "red"},
    )

    plt.savefig(plot_path / f"RE_plots_other_drugs_{drug}.pdf")
    plt.clf()
