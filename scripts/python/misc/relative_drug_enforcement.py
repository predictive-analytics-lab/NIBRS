"""Calculate the relative drug enforcement given drug A and drug B over counties."""
from typing import List
import pandas as pd
from pathlib import Path

def process_df(df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
    df[f"{drug_name}_incidents"] = df["black_incidents"] + df["white_incidents"]
    df[f"{drug_name}_uses"] = df["black_users"] + df["white_users"]
    df[f"{drug_name}_enforcement"] = df[f"{drug_name}_incidents"] / df[f"{drug_name}_uses"]
    df[f"{drug_name}_bw_incidents"] = df["black_incidents"] / df["white_incidents"]
    df[f"{drug_name}_SR"] = df["selection_ratio"]
    return df[["FIPS", f"{drug_name}_incidents", f"{drug_name}_uses", f"{drug_name}_enforcement", f"{drug_name}_bw_incidents", f"{drug_name}_SR", "year"]]


def relative_drug_enforcement_AB(
    drug_A: pd.DataFrame,
    drug_B: pd.DataFrame,
    drug_A_name: str,
    drug_B_name: str,
    min_incidents : int = 30
) -> pd.DataFrame:
    """Calculate the relative drug enforcement given drug A and drug B over counties."""
    drug_A[f"{drug_A_name}_incidents"] = drug_A["black_incidents"] + drug_A["white_incidents"]
    drug_B[f"{drug_B_name}_incidents"] = drug_B["black_incidents"] + drug_B["white_incidents"]
    drug_A = drug_A.loc[drug_A[f"{drug_A_name}_incidents"] >= min_incidents]
    drug_B = drug_B.loc[drug_B[f"{drug_B_name}_incidents"] >= min_incidents]
    drug_A[f"{drug_A_name}_uses"] = drug_A["black_users"] + drug_A["white_users"]
    drug_B[f"{drug_B_name}_uses"] = drug_B["black_users"] + drug_B["white_users"]
    drug_A[f"{drug_A_name}_bw_ratio"] = drug_A["black_users"] / drug_A["white_users"]
    drug_B[f"{drug_B_name}_bw_ratio"] = drug_B["black_users"] / drug_B["white_users"]
    drug_B.drop(columns=["bwratio"], inplace=True)
    cols = ["FIPS", f"{drug_A_name}_incidents", f"{drug_A_name}_uses", f"{drug_B_name}_incidents", f"{drug_B_name}_uses", f"{drug_A_name}_bw_ratio", f"{drug_B_name}_bw_ratio", "bwratio"]
    comb_drug = drug_A.merge(drug_B, on="FIPS")[cols]
    comb_drug[f"{drug_A_name}_enforcement"] = comb_drug[f"{drug_A_name}_incidents"] / comb_drug[f"{drug_A_name}_uses"]
    comb_drug[f"{drug_B_name}_enforcement"] = comb_drug[f"{drug_B_name}_incidents"] / comb_drug[f"{drug_B_name}_uses"]
    comb_drug["relative_enforcement"] = comb_drug[f"{drug_A_name}_enforcement"] / comb_drug[f"{drug_B_name}_enforcement"]
    comb_drug["relative_uses"] = comb_drug[f"{drug_A_name}_uses"] / comb_drug[f"{drug_B_name}_uses"]
    comb_drug["relative_incidents"] = comb_drug[f"{drug_A_name}_incidents"] / comb_drug[f"{drug_B_name}_incidents"]
    comb_drug["relative_bw_ratio"] = comb_drug[f"{drug_A_name}_bw_ratio"] / comb_drug[f"{drug_B_name}_bw_ratio"]
    return comb_drug


def main(data_loc: Path, year_range: str, grouped: bool) -> pd.DataFrame:
    drugs = ["crack", "cocaine", "heroin", "meth", "poverty"]
    drug_csvs = [data_loc / f"selection_ratio_county_{year_range}{'_grouped' if grouped else ''}_bootstraps_1000_{d}.csv" for d in drugs]
    df = None
    drugs.remove("poverty")
    drugs.append("cannabis")
    for drug_name, drug_csv in zip(drugs, drug_csvs):
        drug_df = pd.read_csv(drug_csv)
        drug_df = process_df(drug_df, drug_name)
        if df is None:
            df = drug_df
        else:
            df = df.merge(drug_df, on=["FIPS", "year"])
    for drug_name in drugs:
        other_incident_cols = [col for col in df.columns if "incidents" in col]
        other_use_cols = [col for col in df.columns if "uses" in col]
        df["other_incidents"] = df[other_incident_cols].sum(axis=1)
        df["other_uses"] = df[other_use_cols].sum(axis=1)
        df["other_enforcement"] = df["other_incidents"] / df["other_uses"]
        df[f"{drug_name}_relative_enforcement"] = df[f"{drug_name}_enforcement"] / df["other_enforcement"]
    # convert wide to long with FIPS as index and drug as column
    return df

def to_long(df):
    sr_df = pd.melt(df, id_vars=["FIPS"], value_vars=["crack_SR", "meth_SR", "heroin_SR", "cannabis_SR", "cocaine_SR"], var_name="drug", value_name="SR")
    sr_df.drug = sr_df.drug.str.replace("_SR", "")

    re_df = pd.melt(df, id_vars=["FIPS"], value_vars=["crack_relative_enforcement", "meth_relative_enforcement", "heroin_relative_enforcement", "cannabis_relative_enforcement", "cocaine_relative_enforcement"], var_name="drug", value_name="relative_enforcement")
    re_df.drug = re_df.drug.str.replace("_relative_enforcement", "")

    bw_inc_df = pd.melt(df, id_vars=["FIPS"], value_vars=["crack_bw_incidents", "meth_bw_incidents", "heroin_bw_incidents", "cannabis_bw_incidents", "cocaine_bw_incidents"], var_name="drug", value_name="bw_incidents")
    bw_inc_df.drug = bw_inc_df.drug.str.replace("_bw_incidents", "")

    df = sr_df.merge(re_df, on=["FIPS", "drug"])
    df = df.merge(bw_inc_df, on=["FIPS", "drug"])
    return df

if __name__ == "__main__":
    data_path = Path(__file__).parents[3] / "data" / "output"
    years = "2017-2019"
    df = main(data_path, years, grouped=True)
    df = to_long(df)
    save_loc = data_path / f"relative_drug_enforcement_{years}.csv"
    df.to_csv(save_loc)