import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


df = pd.read_csv('processed_sig.csv')
df = df[df['num_bjets'] == 6]
df = df.reset_index(drop=True)

print(df.head())
print('Number of Events = ', len(df))

results_list = []

# Iterate through each row (event) in the DataFrame
for index, row in df.iterrows():
    
    pt_values = [row[f'Pt_bj{i}'] for i in range(1, 7)]
    eta_values = [row[f'Eta_bj{i}'] for i in range(1, 7)]
    phi_values = [row[f'Phi_bj{i}'] for i in range(1, 7)]
    
    # Generate combinations of two bjets for Pt, Eta, and Phi
    pt_combinations = list(itertools.combinations(pt_values, 2))
    eta_combinations = list(itertools.combinations(eta_values, 2))
    phi_combinations = list(itertools.combinations(phi_values, 2))
    combdf = pd.DataFrame(zip(pt_combinations, eta_combinations, phi_combinations), columns=['Pt', 'Eta', 'Phi'])
    combdf.to_csv('mycombinations.csv', index=False)
    
    # Calculate masses for each combination and store them in a list
    masses = []
    for pt, eta, phi in zip(pt_combinations, eta_combinations, phi_combinations):
        mass = np.sqrt(2 * pt[0] * pt[1] * (np.cosh(eta[0] - eta[1]) - np.cos(phi[0] - phi[1])))
        masses.append(mass)
    
    # Append the masses to the results list for this event
    results_list.append(masses)
    

masses_df = pd.DataFrame(results_list, columns=[f'Mass_{i+1}' for i in range(15)])
print(masses_df.describe())
masses_df.to_csv('masses.csv', index=False)
print(masses_df.max(), masses_df.min())

# Concatenate the original DataFrame with the new masses DataFrame
result_df = pd.concat([df, masses_df], axis=1)
# result_df.to_csv('concatted.csv', index=False)


# Change this!!!!
target_mass = 1500  # Target invariant mass in GeV
mass_column_names = ['Mass_{}'.format(i) for i in range(1, 16)]

# Lists to store the results
combinations_1 = []
combinations_2 = []
masses_1 = []
masses_2 = []
pt_values_1 = []
pt_values_2 = []
eta_values_1 = []
eta_values_2 = []
phi_values_1 = []
phi_values_2 = []

# Iterate through each row (event) in the DataFrame
for index, row in result_df.iterrows():
    # Get the calculated masses for this event
    masses = row[mass_column_names]
    
    # Find the indices where the mass is closest to the target_mass (top 2 closest masses)
    closest_mass_indices = np.argsort(np.abs(masses - target_mass))[:2]
    
    # Get the indices of the bjets combinations corresponding to the closest masses
    bjet_indices_1, bjet_indices_2 = list(itertools.combinations(range(1, 7), 2))[closest_mass_indices[0]], list(itertools.combinations(range(1, 7), 2))[closest_mass_indices[1]]
    
    # Check if any jet indices are common between the two combinations
    if set(bjet_indices_1) & set(bjet_indices_2):
        # If there are common jet indices, find an alternative combination for bjet_indices_2
        available_jet_indices = list(set(range(1, 7)) - set(bjet_indices_1))
        bjet_indices_2 = list(itertools.combinations(available_jet_indices, 2))[0]
    
    # Extract the Pt, Eta, and Phi values for the bjet combinations for the first closest mass
    pt_values_combination_1 = [row[f'Pt_bj{i}'] for i in bjet_indices_1]
    eta_values_combination_1 = [row[f'Eta_bj{i}'] for i in bjet_indices_1]
    phi_values_combination_1 = [row[f'Phi_bj{i}'] for i in bjet_indices_1]
    
    # Extract the Pt, Eta, and Phi values for the bjet combinations for the second closest mass
    pt_values_combination_2 = [row[f'Pt_bj{i}'] for i in bjet_indices_2]
    eta_values_combination_2 = [row[f'Eta_bj{i}'] for i in bjet_indices_2]
    phi_values_combination_2 = [row[f'Phi_bj{i}'] for i in bjet_indices_2]
    
    # Append the combinations, masses, and values to the respective lists
    combinations_1.append(bjet_indices_1)
    combinations_2.append(bjet_indices_2)
    masses_1.append(masses[closest_mass_indices[0]])
    masses_2.append(masses[closest_mass_indices[1]])
    pt_values_1.append(pt_values_combination_1)
    pt_values_2.append(pt_values_combination_2)
    eta_values_1.append(eta_values_combination_1)
    eta_values_2.append(eta_values_combination_2)
    phi_values_1.append(phi_values_combination_1)
    phi_values_2.append(phi_values_combination_2)


results_df = pd.DataFrame({
    'Combination_1': combinations_1,
    'Mass_1': masses_1,
    'Pt_1': pt_values_1,
    'Eta_1': eta_values_1,
    'Phi_1': phi_values_1,
    'Combination_2': combinations_2,
    'Mass_2': masses_2,
    'Pt_2': pt_values_2,
    'Eta_2': eta_values_2,
    'Phi_2': phi_values_2
})

# Now, results_df contains the desired output with unique jet combinations for the closest masses.

result_df.to_csv('resulting_masses.csv', index=False)





lower_bound = 0  # Lower bound for the invariant mass
upper_bound = 2500  # Upper bound for the invariant mass
bin_width = 25  # Width of each bin

num_bins = int((upper_bound - lower_bound) / bin_width)
masses_flat = masses_df.values.flatten()

# Plotting
plt.figure(figsize=(8, 6))
plt.hist(masses_flat, bins=num_bins, range=(lower_bound, upper_bound), color='black', edgecolor='white')

mean = np.mean(masses_flat)
median = np.median(masses_flat)
std_dev = np.std(masses_flat)
min_val = np.min(masses_flat)
max_val = np.max(masses_flat)
stats_str = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'

plt.legend([stats_str])
plt.xlabel('Invariant Mass')
plt.ylabel('Number of Occurences')
plt.title('Histogram of Invariant Masses for Potential Heavy Higgs (VLQ 2000 GeV)')
plt.grid(True)

plt.xlim(lower_bound, upper_bound)
plt.xticks(np.arange(0, 2500, 500))

plt.show()

closest_masses_1 = []
closest_masses_2 = []
for index, row in result_df.iterrows():
    # Get the calculated masses for this event
    masses = row[mass_column_names]
    
    # Find the indices where the mass is closest to the target_mass (top 2 closest masses)
    closest_mass_indices = np.argsort(np.abs(masses - target_mass))[:2]
    
    # Get the closest mass values
    closest_mass_1 = masses.iloc[closest_mass_indices[0]]
    closest_mass_2 = masses.iloc[closest_mass_indices[1]]
    
    # Append the closest mass values to the lists
    closest_masses_1.append(closest_mass_1)
    closest_masses_2.append(closest_mass_2)

# Append the closest mass values to the original DataFrame
df['H1_M'] = closest_masses_1
df['H2_M'] = closest_masses_2
df.to_csv('processed_sig_with_closest_masses.csv', index=False)
