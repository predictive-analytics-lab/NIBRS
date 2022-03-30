"""Plots to visualize relative drug enforcement."""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

def kde_plot(enforcement_df: pd.DataFrame,):
    """Plot KDE of relative drug enforcement in log scale"""
    
    sns.kdeplot(np.log(enforcement_df["relative_enforcement"]), shade=True, label="Relative drug enforcement")

    plt.xlabel("Log Relative drug enforcement")
    plt.ylabel("Density")
    plt.title("Relative drug enforcement Crack / Cocaine")
    plt.show()

def histogram_plot(enforcement_df: pd.DataFrame,):
    """Plot histogram of relative drug enforcement."""
    sns.distplot(np.log(enforcement_df["relative_enforcement"]),
        bins=100, kde=True, label="Relative drug enforcement")
    plt.xlabel("Log Relative drug enforcement")
    plt.ylabel("Frequency")
    plt.title("Relative drug enforcement Crack / Cocaine")
    plt.show()

def scatter_plot(enforcement_df: pd.DataFrame,):
    """Plot scatter plot of relative drug enforcement."""
    sns.scatterplot(x=np.log(enforcement_df["bwratio"]), y=np.log(enforcement_df["relative_enforcement"]))
    #fit and plot regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(enforcement_df["bwratio"]), np.log(enforcement_df["relative_enforcement"]))
    plt.plot(np.log(enforcement_df["bwratio"]), slope*np.log(enforcement_df["bwratio"]) + intercept, 'r', label='Regression Line')
    # plot line on y=1
    plt.axhline(y=0, color='black', linestyle='--')
    



    plt.xlabel("Log Black / White Population Ratio")
    plt.ylabel("Log Relative drug enforcement Crack/Cocaine")
    plt.title(f"Relative drug enforcement, reg. coef: {slope:.2f}, p-value: {p_value:.2f}")
    plt.show()

def rel_enf_box_plots(enforcement_df: pd.DataFrame,):
    """Plot box plots of relative drug enforcement. Remove outliers."""
    enforcement_df["drug"] = enforcement_df.drug.str.title()
    enforcement_df["relative_enforcement"] = np.log(enforcement_df["relative_enforcement"])
    sns.boxplot(x="drug", y="relative_enforcement", data=enforcement_df)
    plt.xlabel("Drug")
    plt.ylabel("Log Relative drug enforcement")
    plt.title("Log Relative drug enforcement for 5 common drugs.")
    # Add padding above x label
    plt.tight_layout(pad=0.5)
    # capatilize x axis labels
    plt.show()

def bwratio_inc_box_plots(enforcement_df: pd.DataFrame,):
    """Plot box plots of relative drug enforcement. Remove outliers."""
    enforcement_df["drug"] = enforcement_df.drug.str.title()
    enforcement_df["bw_incidents"] = np.log(enforcement_df["bw_incidents"])
    sns.boxplot(x="drug", y="bw_incidents", data=enforcement_df)
    plt.xlabel("Drug")
    plt.ylabel("Log B/W Incidents")
    plt.title("B/W Incidents for 5 common drugs.")
    # Add padding above x label
    plt.tight_layout(pad=0.5)
    plt.show()
    # capatilize x axis labels

def SR_box_plot(enforcement_df: pd.DataFrame,):
    enforcement_df["drug"] = enforcement_df.drug.str.title()
    enforcement_df["SR"] = np.log(enforcement_df["SR"])
    sns.boxplot(x="drug", y="SR", data=enforcement_df)
    plt.xlabel("Drug")
    plt.ylabel("Log Enforcement Ratio")
    plt.title("Log Enforcement Ratio for 5 common drugs.")
    # Add padding above x label
    plt.tight_layout(pad=0.5)
    plt.show()

    # capatilize x axis labels

    plt.show()
if __name__ == "__main__":
    enforcement_df = pd.read_csv(Path(__file__).parents[3] / "data" / "output" / "relative_drug_enforcement.csv")
    # rel_enf_box_plots(enforcement_df)
    # bwratio_inc_box_plots(enforcement_df)
    SR_box_plot(enforcement_df)
    # kde_plot(enforcement_df)
    # histogram_plot(enforcement_df)
    # scatter_plot(enforcement_df) 