"""Script which combines the incidents and usage for all current drug types: cannabis, cocaine, heroin, crack, meth."""
import pandas as pd
from pathlib import Path

def get_usage(data_path: Path, base_name: str, dem_split: bool) -> pd.DataFrame:
    drugs = ["_cocaine", "_heroin", "_crack", "_meth", ""]
    if not dem_split:
        df = pd.DataFrame(columns=["FIPS", "uses"])
    else:
        df = pd.DataFrame(columns=["FIPS", "black_uses", "white_uses"])
    for drug in drugs:
        drug_df = pd.read_csv(data_path / f"{base_name}{drug}.csv", dtype={"FIPS": str})
        drug_df.fillna(0, inplace=True)
        if not len(df) > 0:
            df[["FIPS", "black_uses", "white_uses"]] = drug_df[["FIPS", "black_users", "white_users"]]
            if not dem_split:
                df["uses"] = df["black_uses"] + df["white_uses"]
                df.drop(columns=["black_uses", "white_uses"], inplace=True)
        else:
            df = df.merge(drug_df[["FIPS", "black_users", "white_users"]], on="FIPS", how="outer")
            df.fillna(0, inplace=True)
            if not dem_split:
                df["uses"] += df["black_users"] + df["white_users"]
            else:
                df["black_uses"] += df["black_users"]
                df["white_uses"] += df["white_users"]
            df.drop(columns=["black_users", "white_users"], inplace=True)
    return df

def get_incidents(rawdata_path: Path, dem_split: bool) -> pd.DataFrame:
    # only read FIPS and race
    df = pd.read_csv(rawdata_path, dtype={"FIPS": str}, usecols=["FIPS", "race", "incidents"])
    if dem_split:
        return df.groupby(["FIPS", "race"]).incidents.sum().reset_index().pivot(index="FIPS", columns="race", values="incidents").fillna(0).reset_index()
    else:
        return df.groupby("FIPS").incidents.sum().to_frame("incidents").reset_index()


def enforcement(with_racial_bias: bool) -> pd.DataFrame:
    data_path = Path(__file__).parents[3] / "data"
    incident_path = data_path / "NIBRS" / "drugs:all_resolution:county_2017-2019.csv"
    incident_df = get_incidents(incident_path, with_racial_bias)
    usage_df = get_usage(data_path / "output", "selection_ratio_county_2017-2019_grouped_bootstraps_1000", with_racial_bias)
    enforcement_df = incident_df.merge(usage_df, on="FIPS", how="left")
    if with_racial_bias:
        enforcement_df["enforcement_ratio"] = (enforcement_df["black"] / enforcement_df["black_uses"]) / (enforcement_df["white"] / enforcement_df["white_uses"])
    else:
        enforcement_df["enforcement_rate"] = enforcement_df["incidents"] / enforcement_df["uses"]
    return enforcement_df


if __name__ == "__main__":
    bias_df = enforcement(True)
    rate_df = enforcement(False)
    bias_df.to_csv(Path(__file__).parents[3] / "data" / "output" / "enforcement_bias_all_drugs.csv", index=False)
    rate_df.to_csv(Path(__file__).parents[3] / "data" / "output" / "enforcement_rate_all_drugs.csv", index=False)
