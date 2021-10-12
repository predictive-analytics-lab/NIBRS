import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


plot_path = Path(__file__).parents[3] / "plots"

locations = ["Hotel", "Parking", "Road", "Other", "Home", "School"]
ERs = [3.93, 3.62, 3.51, 2.91, 2.50, 2.03]
sns.barplot(locations, ERs, color="#37474F")
# plt.title("Differential Enforcement Ratio By Location of Incident")
plt.ylabel("Enforcement Ratio")
plt.xlabel("Location")
plt.savefig(plot_path / "location_ER_bar.pdf")
