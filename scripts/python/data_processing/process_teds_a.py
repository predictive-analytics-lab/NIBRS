"""Process the TEDS-A dataset to get admission ratios - as a proxy for usage."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = Path(__file__).parents[3]
output_path = Path(__file__).parents[3] / "plots"


def load_admission_data():
    """Load the data, and process."""
    df = pd.read_csv(
        base_dir / "data" / "misc" / "tedsa_puf_2000_2019.csv",
        usecols=["RACE", "ADMYR", "COKEFLG", "HERFLG", "MTHAMFLG", "MARFLG"],
        dtype={"ADMYR": int}
    )
    df = df[df.ADMYR >= 2010]
    df["RACE"] = df.RACE.map({4: "black", 5: "white"})
    df = df.groupby(["RACE", "ADMYR"]).sum().reset_index()
    df = df.rename(columns={"RACE": "race", "ADMYR": "year"})
    df = df.melt(id_vars=["race", "year"], value_vars=[
                 "COKEFLG", "HERFLG", "MTHAMFLG", "MARFLG"], var_name="drug", value_name="admissions").reset_index()
    df["drug"] = df.drug.map({"COKEFLG": "crack/cocaine", "HERFLG": "heroin",
                             "MTHAMFLG": "methamphetamine", "MARFLG": "marijuana"})
    return df


def load_census():
    df = pd.read_csv(
        base_dir / "data" / "census" / "census-2020-county.csv",
        usecols=["YEAR", "WA_MALE", "WA_FEMALE",
                 "BA_MALE", "BA_FEMALE", "AGEGRP"],
        dtype={"YEAR": int},
        engine="python",
        encoding="ISO-8859-1",
    )
    df = df[df.YEAR >= 3]
    df["YEAR"] += 2007
    df = df[df.AGEGRP == 0]
    df = df.replace("X", np.nan)
    df["black_population"] = df.BA_MALE.astype(
        float) + df.BA_FEMALE.astype(float)
    df["white_population"] = df.WA_MALE.astype(
        float) + df.WA_FEMALE.astype(float)
    df = df.groupby(["YEAR"]).agg(
        {"black_population": "sum", "white_population": "sum"}).reset_index()
    df = df.rename(columns={"YEAR": "year",
                   "white_population": "white", "black_population": "black"})
    df = df.melt(id_vars=["year"], value_vars=["black", "white"],
                 var_name="race", value_name="population").reset_index()
    return df[["year", "race", "population"]]


def load_data():
    admin = load_admission_data()
    census = load_census()
    df = admin.merge(census, on=["year", "race"], how="left")
    df["admissions_p100k"] = (df.admissions / df.population) * 100_000
    return df


def admission_plot(df: pd.DataFrame, filename: str):
    # sns.set(font_scale=1.5, rc={'text.usetex': True}, style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 7))

    ax = sns.lineplot(data=df, x="year", y="admissions_p100k", hue="drug",
                      style="race", markers=True, ax=ax)
    ax.set(yscale="log")

    # get legend handles lables
    handles, labels = ax.get_legend_handles_labels()

    # disable legend
    ax.legend_.remove()

    ax.set_ylabel("Admissions per 100,000")
    ax.set_xlabel("Year")
    plt.tight_layout()
    plt.legend(handles, labels, loc="lower right", ncol=3,)
    plt.savefig(output_path / f"{filename}.pdf")
    plt.clf()


def admission_ratio_plot(df: pd.DataFrame, filename: str):
    df = df.pivot(index=["year", "drug"], columns="race",
                  values="admissions_p100k")
    df["admission_ratio"] = df.black / df.white
    # sns.set(font_scale=1.5, rc={'text.usetex': True}, style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 7))

    ax = sns.lineplot(data=df, x="year", y="admission_ratio",
                      hue="drug", style="drug", markers=True, ax=ax)
    ax.set(yscale="log")
    ax.set_ylim(0.1, 10)

    # get legend handles lables
    handles, labels = ax.get_legend_handles_labels()

    # disable legend
    ax.legend_.remove()

    ax.set_ylabel("Admissions Black/White Ratio")
    ax.set_xlabel("Year")
    plt.tight_layout()
    plt.legend(handles, labels, loc="lower right", ncol=3,)
    plt.savefig(output_path / f"{filename}.pdf")
    plt.clf()


def main():
    df = load_data()
    admission_plot(df, "admissions_p100k")
    admission_ratio_plot(df, "admission_ratio")


if __name__ == "__main__":
    main()
