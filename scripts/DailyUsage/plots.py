# %%
dag.get_node("incident")["CPD"] = baynet.parameters.ConditionalProbabilityTable(dag.get_node("incident"))
dag.get_node("incident")["CPD"].parents = ["pop_" + col.lower() for col in cols] + ["uses_cannabis"]

# %%

dag.get_node("incident")["CPD"].array = inc_array
dag.get_node("incident")["CPD"].rescale_probabilities()



# %%
p = ["#00685c", "#00695c"]

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)


incidents["TYPE"] = "Incident"
arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([incidents, arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Frequency")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Frequency of Incident / Arrests given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")
        # if patch._facecolor[0] == 0.0514705882352941:
        #     pass
        # else:
        #     patch.set_hatch("\\")



# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
census_data = df.groupby(["AGE", "SEX", "RACE"])["POPESTIMATE2019"].sum().reset_index()

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
incidents[0] = (incidents[0] / census_data["POPESTIMATE2019"]) * 100_000

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
arrests[0] = (arrests[0] / census_data["POPESTIMATE2019"]) * 100_000

incidents["TYPE"] = "Incident"
arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([incidents, arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Rate per 100K")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Number of Incidents/Arrests per 100,000 population by demographic")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")


# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)

arrests[0] /= incidents[0]

g = sns.FacetGrid(arrests, row = 'SEX',  col = 'AGE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Ratio")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Ratio of arrests to incidents given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%

census_data = df.groupby(["AGE", "SEX", "RACE"])["POPESTIMATE2019"].sum().reset_index()

census_data["POPESTIMATE2019"] /= sum(census_data["POPESTIMATE2019"])

g = sns.FacetGrid(census_data, row = 'SEX',  col = 'AGE', sharex=False, palette=p, height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', "POPESTIMATE2019", ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Proportion")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Demographic distribution as a proportion of total population")
for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%
nibrs_data.groupby(["AGE", "SEX", "RACE"]).count() / len(nibrs_data)
# %%
nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()

# %%

drug_use_df
# %%


cross = pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df[col] for col in ["AGE", "SEX", "RACE"]], normalize="columns")
cross_mult = cross.multiply(cross.index, axis="rows")
cross_sum = (cross_mult.sum(axis="rows") / 30).reset_index()

g = sns.FacetGrid(cross_sum, row = 'SEX',  col = 'AGE', sharex=False, palette=p, height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("P(U|D = d)")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Probability of drug use (in a given month) by demographics according to the NSDUH survey")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")
# %%
pd.crosstab(drug_use_df.MJDAY30A, [drug_use_df[col] for col in ["AGE", "SEX", "RACE"]], normalize="columns")
# %%

# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

incidents = nibrs_df.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# incidents[0] /= len(nibrs_df)

arrests = nibrs_arrests.groupby(["AGE", "SEX", "RACE"]).size().reset_index()
# arrests[0] /= len(nibrs_arrests)


g = sns.FacetGrid(arrests, row = 'SEX',  col = 'AGE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', 0, ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Ratio")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Ratio of arrests to incidents given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")

# %%

nibrs_df.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})
nibrs_arrests.rename({"dm_offender_race_ethnicity": "RACE", "dm_offender_age": "AGE", "dm_offender_sex":"SEX"})

nibrs_df["TYPE"] = "Incident"
nibrs_arrests["TYPE"] = "Arrest"

nibrs_data = pd.concat([nibrs_df, nibrs_arrests])

g = sns.FacetGrid(nibrs_data, row = 'SEX',  col = 'AGE', hue = 'TYPE', sharex=False, palette=p,height=5, aspect=5/5, gridspec_kws={"hspace":0.4, "wspace":0})
g = (g.map(sns.barplot, 'RACE', "cannabis_mass", ci = None).add_legend())
g.despine(left=True)
g.set_ylabels("Frequency")

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Frequency of Incident / Arrests given demographics in the NIBRS dataset.")

for ax in g.axes.flat:
    for patch in ax.patches:
        patch.set_edgecolor("black")


# %%

nibrs_data.groupby(["TYPE", "RACE", "SEX", "AGE"]).mean().reset_index()

# %%
