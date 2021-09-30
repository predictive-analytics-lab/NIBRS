import matplotlib.pyplot as plt
import seaborn as sns

locations = ["Hotel", "Parking", "Road", "Other", "Home", "School"]
ERs = [3.93, 3.62, 3.51, 2.91, 2.50, 2.03]
sns.barplot(locations, ERs, color="#37474F")
# plt.title("Differential Enforcement Ratio By Location of Incident")
plt.ylabel("Enforcement Ratio")
plt.xlabel("Location")
plt.savefig("location_ER_bar.pdf")
