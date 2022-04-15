import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


nsduh_path = Path(__file__).parents[3] / "data" / "NSDUH"
output_path = Path(__file__).parents[3] / "plots"

groups = {
    "drugs": ["cocaine", "heroin", "meth", "crack", "cannabis"],
    "ucr": ["dui", "drunkeness"]
}


def usage_ratio_plot(df: pd.DataFrame):
    sns.lineplot(x="year", y="usage_ratio", hue="drug",
                 data=df, markers=True, style="drug")
    plt.legend(loc="lower center",  ncol=5,
               fancybox=True, bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.yscale("log")
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel("Usage Ratio")
    plt.ylim(0.1, 10)
    plt.xlabel("Year")
    plt.savefig(output_path / "usage_ratio.pdf")


def usage_plot(df: pd.DataFrame, filename: str):
    sns.set(font_scale=1.5, rc={'text.usetex': True}, style="whitegrid")

    for name, li in groups.items():
        fig, ax = plt.subplots(figsize=(8, 7))

        sdf = df[df.drug.isin(li)]
        sdf = sdf.melt(id_vars=["year", "drug"], value_vars=[
                       "white", "black"], value_name="usage", var_name="race").reset_index()
        ax = sns.lineplot(data=sdf, x="year", y="usage", hue="drug",
                          style="race", markers=True, ax=ax)
        ax.set(yscale="log")

        # get legend handles lables
        handles, labels = ax.get_legend_handles_labels()

        # disable legend
        ax.legend_.remove()

        if "dou" in filename:
            ax.set_ylabel(
                r"Average days of use")
        else:
            ax.set_ylabel("User proportion")
        ax.set_xlabel("Year")
        plt.tight_layout()
        plt.legend(handles, labels, loc="lower right", ncol=3,)
        plt.savefig(output_path / f"{filename}_{name}.pdf")
        plt.clf()


def smooth(df):
    return df.groupby(["drug"]).rolling(3, on="year", center=True).mean().reset_index()


def main(smoothing: bool = True):
    df_dou = pd.read_csv(nsduh_path / "nsduh_usage_ratio_dou.csv")
    df = pd.read_csv(nsduh_path / "nsduh_usage_ratio.csv")
    if smoothing:
        df_dou = smooth(df_dou)
        df = smooth(df)
    usage_plot(df_dou, "nsduh_usage_ratio_dou")
    usage_plot(df, "nsduh_usage_ratio")


if __name__ == "__main__":
    main()
