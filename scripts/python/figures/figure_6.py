"""Correlation Matrix Between Drug Rates + DUI/Drunkeness/Other Incidents Rates."""
from pathlib import Path
from typing import Tuple
from matplotlib.pyplot import legend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

data_dir = Path(__file__).parents[3] / "data" / "output"

selected_years = [2017, 2018, 2019]

dataframe_paths = {
    "crack": data_dir / "selection_ratio_county_2010-2020_wilson_crack.csv",
    "meth": data_dir / "selection_ratio_county_2010-2020_wilson_meth.csv",
    "cocaine": data_dir / "selection_ratio_county_2010-2020_wilson_cocaine.csv",
    "heroin": data_dir / "selection_ratio_county_2010-2020_wilson_heroin.csv",
    "cannabis": data_dir / "selection_ratio_county_2010-2020_wilson.csv",
    "non-drug_offenses": data_dir / "other_incidents_2010-2020.csv",
}


def load_data(data_path: Path, name: str):
    df = pd.read_csv(data_path)
    df = df[df.year.isin(selected_years)]
    df = df.groupby(["FIPS"]).agg(
        {"black_population": "sum", "white_population": "sum", "black_incidents": "sum", "white_incidents": "sum"}).reset_index()
    df[f"black_{name}_rate"] = (
        df[f"black_incidents"] / df["black_population"]) * 100_000
    df[f"white_{name}_rate"] = (
        df[f"white_incidents"] / df["white_population"]) * 100_000
    df[f"black_{name}_error"] = df.apply(lambda row: wilson_error(
        row[f"black_incidents"], row["black_population"])[1], axis=1) * 100_000
    df[f"white_{name}_error"] = df.apply(lambda row: wilson_error(
        row[f"white_incidents"], row["white_population"])[1], axis=1) * 100_000
    return df[["FIPS", f"black_{name}_rate", f"white_{name}_rate", f"black_{name}_error", f"white_{name}_error"]].reset_index(drop=True)


def load_all_data():
    from functools import reduce
    """Load all data into a single dataframe."""
    dataframes = [load_data(path, name)
                  for name, path in dataframe_paths.items()]
    # merge all dataframes on fips
    df = reduce(lambda left, right: pd.merge(
        left, right, on="FIPS", how="inner"), dataframes)
    df = df.drop(columns=["FIPS"])
    df = df.replace(np.nan, 0)
    return df


def wilson_error(n_s: int, n: int, z=1.96):
    """
    Wilson score interval

    param n_s: number of successes
    param n: total number of events
    param z: The z-value

    return: The lower and upper bound of the Wilson score interval
    """
    n_f = np.max([1, n - n_s])
    denom = n + z ** 2
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2 / 4))
    return adjusted_p, ci


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def heatmap(data, labels, mask, focus_mask, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the heatmap
    data = np.ma.masked_where(mask == False, data)
    focus_data = np.ma.masked_where(focus_mask == False, data)
    im = ax.imshow(focus_data, **kwargs,)
    im2 = ax.imshow(data, alpha=0.7, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im2, ax=ax, shrink=0.7, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1] - 1))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(labels[:-1])
    ax.set_yticklabels(["", *labels[1:]])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, focus_mask, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    data = np.ma.masked_where(focus_mask == False, data)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data[i, j] > threshold:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts


def main():
    df = load_all_data()
    corr_cols = [col for col in df.columns if "rate" in col]
    corrs = np.zeros((len(corr_cols), len(corr_cols)))
    for i, x in enumerate(corr_cols):
        for j, y in enumerate(corr_cols):
            if i == j:
                corrs[i, j] = 1
            else:
                error = df[x[:-5] + "_error"] + df[y[:-5] + "_error"]
                corrs[i, j] = corr(df[x], df[y], 1 / error)
    # get lower triangle
    lower_triangle = np.tril(corrs)
    # plot heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.set(style="white", )
    # fig, ax = plt.subplots(figsize=(12, 10))
    formatted_cols = [x.replace("_", " ").title() for x in corr_cols]
    # sns.set(rc={'text.usetex': True})
    # sns.heatmap(lower_triangle, annot=True, fmt=".2f",
    #             xticklabels=formatted_cols[1:], yticklabels=formatted_cols, mask=np.array(np.array(lower_triangle < 0.5).astype(int) + np.array(lower_triangle >= 1).astype(int)).astype(bool), ax=ax)
    # sns.heatmap(lower_triangle, annot=False,
    #             xticklabels=formatted_cols[1:], yticklabels=formatted_cols, mask=np.array(np.array(lower_triangle > 0.5).astype(int) + np.tril(corrs)).astype(bool), ax=ax, legend=False)
    # # rotate x ticks - shift back left
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    im, cbar = heatmap(data=lower_triangle, labels=formatted_cols, mask=np.tril(
        corrs, -1).astype(bool), focus_mask=(corrs > 0.5).astype(bool), cbarlabel="Correlation")
    annotate_heatmap(im, data=corrs, focus_mask=(
        corrs > 0.5).astype(bool), threshold=0.5)
    plt.tight_layout()
    plt.savefig(data_dir.parents[1] / "plots" / "corr_heatmap.pdf")


if __name__ == "__main__":
    main()
