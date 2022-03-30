"""Plot Incident / Usage in 2020 with Lockdown Markers"""
from matplotlib import use
import pandas as pd
from pathlib import Path
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_path = Path(__file__).parents[3] / "data"

def load_lockdown_data() -> pd.DataFrame:
    df = pd.read_csv(data_path / "misc" / "interventions.csv", index_col=False)
    df = df.drop(columns=["AREA_NAME", "FIPS"])
    df = df.rename(columns={"stay at home": "lockdown", "STATE": "state", "stay at home rollback": "lockdown_rollback"})

    intervention_cols = list(set(df.columns) - {col for col in df.columns if "rollback" in col} - {"state"})
    rollback_cols = [col for col in df.columns if "rollback" in col]

    def earliest_intervention(row):
        # get todays date
        earliest = datetime.datetime.today().date()
        for col in intervention_cols:
            if not isinstance(row[col], float) and row[col] < earliest:
                earliest = row[col]
        if earliest == datetime.datetime.today().date():
            return None
        return earliest
    
    def latest_rollback(row):
        latest = datetime.datetime.min.date()
        for col in rollback_cols:
            if not isinstance(row[col], float) and row[col] > latest:
                latest = row[col]
        if latest == datetime.datetime.min.date():
            return None
        return latest

    def apply_datetime(x):
        return datetime.date.fromordinal(int(x)) if not np.isnan(x) else np.nan
    def int_or_nan(x):
        return int(np.mean(x)) if not np.isnan(np.mean(x)) else np.nan

    df = df.groupby(["state"]).agg({k: int_or_nan for k in intervention_cols + rollback_cols}).reset_index()
    for col in intervention_cols + rollback_cols:
        df[col] = df[col].apply(apply_datetime)
    df["earliest_intervention"] = df.apply(earliest_intervention, axis=1)
    df["latest_rollback"] = df.apply(latest_rollback, axis=1)
    return df[["state", "earliest_intervention", "latest_rollback", "lockdown", "lockdown_rollback"]]

def load_covid_stats() -> pd.DataFrame:
    df = pd.read_csv(data_path / "misc" / "covid-stats.csv", index_col=False, usecols=["date", "cases", "state"])
    df["state"] = df.state.str.lower()
    abbv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["STATE", "ABBRV"])
    abbv_df["STATE"] = abbv_df.STATE.str.lower()
    df = df.merge(abbv_df, left_on="state", right_on="STATE")[["date", "cases", "ABBRV"]]
    df = df.rename(columns={"ABBRV": "state"})
    df["date"] = pd.to_datetime(df.date)
    df["week"] = df.date.dt.isocalendar().week
    # filter to 2020
    df = df[df.date.dt.year == 2020]
    return df


def load_usage_data() -> pd.DataFrame:
    df = pd.read_csv(data_path / "output" / "selection_ratio_county_2020_bootstraps_1000.csv", dtype={"FIPS": str}, usecols=["FIPS", "black_users", "white_users"])
    fips_abbrv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["ABBRV", "FIPS"], dtype={"FIPS": str})
    df["state_fips"] = df.FIPS.str[:2]
    df = df.merge(fips_abbrv_df, left_on="state_fips", right_on="FIPS")
    df = df.rename(columns={"ABBRV": "state"})
    df = df.groupby(["state"]).agg({"black_users": "sum", "white_users": "sum"}).reset_index()
    # wide to long
    # df = df.melt(id_vars=["state"], value_vars=["black_users", "white_users"], var_name="race", value_name="uses")
    # df["race"] = df.race.apply(lambda x: x.split("_")[0])
    return df

def load_pop_data(split_by_race: bool = False) -> pd.DataFrame:
    df = pd.read_csv(data_path / "output" / "selection_ratio_county_2020_bootstraps_1000.csv", dtype={"FIPS": str}, usecols=["FIPS", "black", "white"])
    fips_abbrv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["ABBRV", "FIPS"], dtype={"FIPS": str})
    df["state_fips"] = df.FIPS.str[:2]
    df = df.merge(fips_abbrv_df, left_on="state_fips", right_on="FIPS")
    df = df.rename(columns={"ABBRV": "state"})
    df = df.groupby(["state"]).agg({"black": "sum", "white": "sum"}).reset_index()
    if split_by_race:
        return df.melt(id_vars="state", value_vars=["black", "white"], var_name="race", value_name="population")
    else:
        df["population"] = df.black + df.white
        return df[["state", "population"]]

def compute_er(incident_df: pd.DataFrame, usage_df: pd.DataFrame) -> pd.DataFrame:
    incident_df = incident_df.pivot(index=["state", "incident_date"], columns="race", values="incident_count").reset_index().fillna(0)
    incident_df["week"] = incident_df.incident_date.dt.week
    # incident_df = incident_df.groupby(["month", "state"]).agg({"black": sum, "white": sum}).reset_index()
    # incident_df["black"] += 0.0001
    # incident_df["white"] += 0.0001
    df = usage_df.merge(incident_df, on=["state"], how="left")
    df["er"] = np.log((df.black / df.black_users) / (df.white / df.white_users))
    return df[["week", "state", "er"]]

def load_incident_data(split_by_race: bool) -> pd.DataFrame:
    df = pd.read_csv(data_path / "NIBRS" / "raw" / "drug_incidents_2010-2020.csv", index_col=False, usecols=["ori", "incident_date", "data_year", "race", "cannabis_count"])
    df = df[df.data_year == 2020]
    df = df[df.cannabis_count > 0]
    ori_df = pd.read_csv(
        data_path / "misc" / "LEAIC.tsv",
        delimiter="\t",
        usecols=["ORI9", "STATENAME"],
    )
    ori_df["STATENAME"] = ori_df.STATENAME.str.lower()
    abbv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["STATE", "ABBRV"])
    abbv_df["STATE"] = abbv_df.STATE.str.lower()
    ori_df = ori_df.merge(abbv_df, left_on="STATENAME", right_on="STATE")[["ORI9", "ABBRV"]]
    ori_df = ori_df.rename(columns={"ORI9": "ori", "ABBRV": "state"})
    df = df.merge(ori_df, on="ori")
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    gb_list = ["state", "incident_date"]
    if split_by_race:
        gb_list += ["race"]
    df = df.groupby(gb_list).size().reset_index()
    df = df.rename(columns={0: "incident_count"})
    return df

def state_incident_plot():
    misc_dir = Path(__file__).parents[3] / "data" / "misc"


    lockdown_df = load_lockdown_data()
    incident_df = load_incident_data(split_by_race=True)
    pop_df = load_pop_data(split_by_race=True)

    pop_df = pop_df.groupby(["state"]).filter(lambda x: all(x["population"] > 10000))

    incident_df["week"] = incident_df.incident_date.dt.week
    incident_df = incident_df[incident_df.state.isin(incident_df.state.unique()[incident_df.groupby("state")["incident_count"].max() > 10])]

    cov_df = pd.read_csv(misc_dir / "county_coverage.csv", dtype={"FIPS": str})
    cov_df = cov_df[cov_df.year == 2020]
    fips_abbrv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["ABBRV", "FIPS"], dtype={"FIPS": str})
    cov_df["state_fips"] = cov_df.FIPS.str[:2]
    cov_df = cov_df.merge(fips_abbrv_df, left_on="state_fips", right_on="FIPS")
    cov_df = cov_df.rename(columns={"ABBRV": "state"})
    cov_df = cov_df.groupby(["state"])["coverage"].mean().reset_index()

    incident_df = incident_df.merge(cov_df, on="state", how="left")
    incident_df = incident_df[incident_df.coverage > 0.8]

    incident_df = incident_df.merge(pop_df, on=["state", "race"], how="inner")
    # calc incident per 100k population
    incident_df["incident_100k"] = np.log((incident_df.incident_count / incident_df.population) * 100000)

    g = sns.relplot(
        data=incident_df,
        x="week", y="incident_100k", color="blue", col="state", hue="race", col_wrap=8, kind="line", facet_kws={'sharey': True, 'sharex': True},
    )
    # set ylabel
    g.set(ylabel="Log Incidents Per 100K")
    # add a vertical line to each plot
    for ax in plt.gcf().axes:
        state = ax.title.get_text().replace("state = ", "")
        import matplotlib.patches as patches
        lockdown_val = lockdown_df[lockdown_df.state == state]["lockdown"].values[0]
        unlock_val = lockdown_df[lockdown_df.state == state]["lockdown_rollback"].values[0]
        earliest_intervention_val = lockdown_df[lockdown_df.state == state]["earliest_intervention"].values[0]
        latest_rollback_val = lockdown_df[lockdown_df.state == state]["latest_rollback"].values[0]
        # get y axis range
        y_min, y_max = ax.get_ylim()
        if isinstance(earliest_intervention_val, datetime.date) and isinstance(latest_rollback_val, datetime.date):
            rect = patches.Rectangle((earliest_intervention_val.isocalendar().week - 0.5, y_min), latest_rollback_val.isocalendar().week - earliest_intervention_val.isocalendar().week, y_max - y_min, linewidth=0, facecolor='green', alpha=0.1)
            ax.add_patch(rect)
        if isinstance(lockdown_val, datetime.date) and isinstance(unlock_val, datetime.date):
            rect = patches.Rectangle((lockdown_val.isocalendar().week - 0.5, y_min), unlock_val.isocalendar().week - lockdown_val.isocalendar().week, y_max - y_min, linewidth=0, facecolor='red', alpha=0.3)
            ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(data_path.parent / "plots" / "covid_state_incidents.pdf")

def tilegrid_plot(lockdown_df: pd.DataFrame, er_df: pd.DataFrame, covid_df:pd.DataFrame):
    misc_dir = Path(__file__).parents[3] / "data" / "misc"

    tile_grid = pd.read_csv(misc_dir / "us_tile_grid_alt.csv")

    # datetime object midway through from the start of the month
    # er_df["month_date"] = er_df.month.apply(lambda x: datetime.date(2020, x, 15).timetuple().tm_yday)

    # load coverage
    cov_df = pd.read_csv(misc_dir / "county_coverage.csv", dtype={"FIPS": str})
    cov_df = cov_df[cov_df.year == 2020]
    fips_abbrv_df = pd.read_csv(data_path / "misc" / "FIPS_ABBRV.csv", usecols=["ABBRV", "FIPS"], dtype={"FIPS": str})
    cov_df["state_fips"] = cov_df.FIPS.str[:2]
    cov_df = cov_df.merge(fips_abbrv_df, left_on="state_fips", right_on="FIPS")
    cov_df = cov_df.rename(columns={"ABBRV": "state"})
    cov_df = cov_df.groupby(["state"])["coverage"].mean().reset_index()

    er_df = er_df.merge(cov_df, on="state")
    er_df = er_df[er_df.coverage > 0.8]
    
    tile_grid = tile_grid.rename(columns={"code": "state"})
    tile_grid_2 = tile_grid.merge(er_df, on="state", how="right")
    

    row_max = tile_grid.row.max()
    col_max = tile_grid.column.max()

    # tile_grid = tile_grid.set_index(['row', 'column'])

    fig, axs = plt.subplots(row_max + 1, tile_grid.column.max() + 1, figsize=(20, 12))


    for i in range(row_max + 1):
        for j in range(col_max + 1):
            df = tile_grid[(tile_grid.row == i) & (tile_grid.column == j)]
            df2 = tile_grid_2[(tile_grid_2.row == i) & (tile_grid_2.column == j)]
            if len(df) > 0:
                ax = axs[i, j]
                state = df.state.values[0]
                if len(df2) > 0:
                    # another plot same axis twin
                    # ax2 = ax.twinx()
                    sns.lineplot(data=df2, x="week", y="er", color="black", ax=ax)
                    # sns.lineplot(data=covid_df[covid_df.state == state], x="week", y="cases", color="red", ax=ax2)
                    # ax2.xaxis.set_major_locator(plt.NullLocator())
                    # ax2.yaxis.set_major_locator(plt.NullLocator())
                    
                    # get ceiling
                    y_max = np.ceil(df2.replace([np.inf, -np.inf], np.nan).dropna().groupby("week").mean().er.max())
                    y_min = np.floor(df2.replace([np.inf, -np.inf], np.nan).dropna().groupby("week").mean().er.min())

                    ax.set_yticks([y_min, y_max])


                    ax.set_ylim(y_min, y_max)
                    # sns.lineplot(data=df2, x="week", y="incident_count", color="blue", ax=ax2)
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    # text in the top center, fontsize 20
                    ax.text(0.5, 0.85, state, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
                    # add a vertical line to each plot
                    import matplotlib.patches as patches
                    lockdown_val = lockdown_df[lockdown_df.state == state]["lockdown"].values[0]
                    unlock_val = lockdown_df[lockdown_df.state == state]["unlock"].values[0]
                    if isinstance(lockdown_val, datetime.date) and isinstance(unlock_val, datetime.date):
                        rect = patches.Rectangle((lockdown_val.isocalendar().week, y_min), unlock_val.isocalendar().week - lockdown_val.isocalendar().week, y_max - y_min, linewidth=0, facecolor='red', alpha=0.3)
                        ax.add_patch(rect)
                    # if isinstance(val, datetime.date):
                    #     ax.axvline(val.isocalendar().week, color="black", linestyle="--")
                    # val = lockdown_df[lockdown_df.state == state]["unlock"].values[0]
                    # if isinstance(val, datetime.date):
                    #     ax.axvline(val.isocalendar().week, color="blue", linestyle="--")
                    # ax.axhline(y=0, color="red", linestyle="-.")
                else:
                    ax.set_facecolor("black")
                    ax.text(0.5, 0.85, state, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, color="white")
                    ax.yaxis.set_major_locator(plt.NullLocator())
                ax.xaxis.set_major_locator(plt.NullLocator())
                # ax.yaxis.set_major_locator(plt.NullLocator())
            else:
                axs[i, j].axis('off')
                
    fig.tight_layout()

    plt.savefig(data_path.parent / "plots" / "covid_state_tilemap_cannabis.pdf")


if __name__ == "__main__":
    # lockdown_df = load_lockdown_data()
    # incident_df_split = load_incident_data(split_by_race=True)
    # use_df = load_usage_data()
    # er_df = compute_er(incident_df_split, use_df)
    # covid_df=load_covid_stats()
    # tilegrid_plot(lockdown_df, er_df, covid_df)
    state_incident_plot()