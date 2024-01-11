import pandas as pd 
import matplotlib.pyplot as plt
import uproot 
from matplotlib import use
use('Agg')

pd.set_option('display.max_columns', None)  # to see all columns of df.head()

filename = "vlq_nn\sig400.root"
file = uproot.open(filename)

print(file.classnames())
print(file.keys())

tree = file["sixtop"] 
tree.show()  # show all the branches inside the TTree
dfall = tree.arrays(library="pd")  # convert uproot TTree into pandas dataframe
print("============================================")
print("File loaded with ", len(dfall), " events ")


filenamebkg=r"vlq_nn\bkg.root"
filebkg = uproot.open(filenamebkg)
treebkg = filebkg["sixtop"]
dfallbkg = treebkg.arrays(library="pd")

dfall["label"]=0
# dfall["event_index"] = range(0, len(dfall))

dfallbkg["label"]=1
# dfallbkg["event_index"] = range(0, len(dfall))

print(dfallbkg.head())

# ----------------- #
print("Number of Events Before Selections:", len(dfall))
selected_events = dfall[
    (dfall.Pt_bj1 > 30.0 ) & (dfall.Pt_bj2 > 30.0 ) & (dfall.Pt_bj3 > 30.0 ) & (dfall.Pt_bj4 > 30.0 ) & (dfall.Pt_bj5 > 30.0 ) & (dfall.Pt_bj6 > 30.0 )
    & (abs(dfall.Eta_bj1) < 4.5 ) & (abs(dfall.Eta_bj2) < 4.5 ) & (abs(dfall.Eta_bj3) < 4.5 ) & (abs(dfall.Eta_bj4) < 4.5 ) & (abs(dfall.Eta_bj5) < 4.5 ) & (abs(dfall.Eta_bj6) < 4.5 )
    & (dfall.met_NonInt > 15.0) & (dfall.met_NonInt<400)& (dfall.num_bjets == 6) & (dfall != -999).all(axis=1) & (dfall.isnull().any(axis=1)==False)
]

# Remove later, this makes it so there is less signal than background.
# selected_events = selected_events.sample(frac=0.5, random_state=42)

# Print the results
print("Number of Signal:", len(selected_events))
print("Number of Events Cut:", len(dfall)-len(selected_events))
selected_events.head()

print(dfallbkg.head())
# ----------------- #
print("Number of Events Before Selections:", len(dfallbkg))
selected_eventsbkg = dfallbkg[
(dfallbkg.met_NonInt > 15.0) & (dfallbkg.met_NonInt<400) & (dfallbkg != -999).all(axis=1) ]

# Print the results
print("Number of Background:", len(selected_eventsbkg))
print("Number of Events Cut:", len(dfallbkg) - len(selected_eventsbkg))

print(selected_eventsbkg.head())


print(selected_events.head())

merged_df=selected_events.combine_first(selected_eventsbkg)

print(merged_df.head())
signal = merged_df[merged_df['label']==0]
background = merged_df[merged_df['label']==1]
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))  # create subplots

variables = ["Pt_bj1", "Pt_bj2", "Pt_bj3", "Pt_bj4", "Pt_bj5", "Pt_bj6", "Eta_bj1", "Eta_bj2", "Eta_bj3", "Eta_bj4", "Eta_bj5", "Eta_bj6", "met_NonInt", "num_bjets", "HTb"]
for i, var in enumerate(variables):
    ax = axs[i//5, i%5]  # calculate appropriate subplot index
    ax.hist(signal[var], bins=100, alpha=0.5, density=True, label="6b + 6bjet", color="blue")
    ax.hist(background[var], bins=100, alpha=0.5, density=True, label="ttbar", color="red")
    ax.set_title(var)
    ax.legend()

plt.tight_layout()
plt.savefig("sig_bac.pdf")

