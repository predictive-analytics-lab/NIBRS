# %%
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from selection_bias import wilson_error
# %%

prob = 1e-5
counts = np.linspace(1, 100, 100)

y = np.array([wilson_error(prob, count)[1] for count in counts])

sns.lineplot(x = counts, y = y, markers=True)
plt.ylabel("Wilson Interval Error")
plt.xlabel("Count")
plt.title(f"Wilson Interval Error / Counts when P = {prob}")
plt.show()
# %%

df = pd.read_csv("../../../data/output/selection_ratio_county_2019.csv")
# %%



sns.scatterplot(x=df.incidents, y=df.ci)
plt.show()
# %%
