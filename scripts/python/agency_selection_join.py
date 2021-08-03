# %%
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "data"

## LOAD DATASETS

agency_df = pd.read_csv(data_path / "output" / "agency_output.csv", index_col=0)
lemas_df = pd.read_csv(data_path / "agency" / "lemas_processed.csv", index_col=0)

# %%

agency_df = pd.merge(agency_df, lemas_df, on="ori", how="left")
agency_df.to_csv(data_path / "output" / "agency_lemas.csv")
# %%
