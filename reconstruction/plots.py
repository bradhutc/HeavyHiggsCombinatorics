import pandas as pd
import matplotlib.pyplot as plt
import ast
import pandas as pd
import ast
import numpy as np

def unpack_nested_list(series, prefix):
    """ Unpacks a nested list in a series and returns a DataFrame with separate columns """
    # Convert string representation of list to actual list using ast.literal_eval
    unpacked = series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    unpacked_df = pd.DataFrame(unpacked.tolist(), columns=[f'{prefix}_{i+1}' for i in range(len(unpacked[0]))])
    
    return unpacked_df

def plot_deltas(df):
    """ Plot the Delta_Pt, Delta_Eta, and Delta_Phi distributions using LaTeX for delta symbols """
    plt.figure(figsize=(15, 5))

    # Plot Delta_Pt
    plt.subplot(1, 3, 1)
    plt.hist(df['Delta_Pt'], bins=100, color='blue', alpha=0.7, edgecolor='black', density=True)
    plt.title(r'$\Delta p_t$ Distribution')  # LaTeX notation for Delta
    plt.xlabel(r'$\Delta p_t$')
    plt.ylabel('Frequency')

    # Plot Delta_Eta
    plt.subplot(1, 3, 2)
    plt.hist(df['Delta_Eta'], bins=100, color='green', alpha=0.7, edgecolor='black', density=True)
    plt.title(r'$\Delta \eta$ Distribution')  # LaTeX notation for Delta
    plt.xlabel(r'$\Delta \eta$')

    # Plot Delta_Phi
    plt.subplot(1, 3, 3)
    plt.hist(df['Delta_Phi'], bins=100, color='red', alpha=0.7, edgecolor='black', density=True)
    plt.title(r'$\Delta \Phi$ Distribution')  # LaTeX notation for Delta
    plt.xlabel(r'$\Delta \Phi$')
    plt.suptitle('Distributions of Differences in Pt, Eta, and Phi for Bjets that Yield Closest Heavy Higgs Mass')
    plt.tight_layout()
    plt.show()
    
def plot_combination_histogram(df, combination_column):

    combination_counts = df[combination_column].value_counts()

    plt.figure(figsize=(10, 6))
    combination_counts.plot(kind='bar', color='#8c564b')
    plt.title(f'Frequency of Each Bjet Combination in {combination_column}')
    plt.xlabel('Bjet Combinations')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_bjet_parameters(df):
    plt.figure(figsize=(18, 6))

    # Histogram for Pt
    plt.subplot(1, 3, 1)
    for i in range(1, 7):
        plt.hist(df[f'Pt_bj{i}'], bins=range(0, int(max(df[f'Pt_bj{i}'])) + 11, 10), alpha=0.7, label=f'Pt_bj{i}')
    plt.title('Pt Histogram for all Bjets')
    plt.xlabel('Pt GeV')
    plt.ylabel('Frequency')
    plt.legend()
    bin_width = 0.05
    plt.subplot(1, 3, 2)
    for i in range(1, 7):
        data = df[f'Eta_bj{i}']
        eta_max = max(data)
        eta_min = min(data)
        num_bins = int((eta_max - eta_min) / bin_width)
        plt.hist(data, bins=num_bins, alpha=0.4, label=f'Eta_bj{i}', density=True)
    plt.title('Eta Histogram for all Bjets')
    plt.xlabel('Eta')
    plt.ylabel('Frequency')
    plt.legend()

    # Histogram for Phi
    plt.subplot(1, 3, 3)
    for i in range(1, 7):
        data = df[f'Phi_bj{i}']
        phi_max = max(data)
        phi_min = min(data)
        num_bins = int((phi_max - phi_min) / bin_width)
        plt.hist(data, bins=num_bins, alpha=0.7, label=f'Phi_bj{i}', density=True)
    plt.title('Phi Histogram for all Bjets')
    plt.xlabel('Phi')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_mass_histogram_excluding_closest(df):
    all_masses = []

    for index, row in df.iterrows():
        differences = np.abs(row - 200)
        closest_idx = differences.nsmallest(2).index
        other_masses = row.drop(closest_idx)
        all_masses.extend(other_masses)

    plt.figure(figsize=(10, 6))
    plt.hist(all_masses, bins=100, color='purple', edgecolor='black', range=(0,2500), alpha=0.8)
    plt.title('Histogram of Masses Excluding Two Closest to 200')
    plt.xlabel('Mass')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 2500, 200))

    # Statistical analysis
    mean = np.mean(all_masses)
    median = np.median(all_masses)
    std_dev = np.std(all_masses)
    plt.figtext(0.15, 0.8, f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}', fontsize=10)

    plt.show()

    
    
if __name__ == "__main__":
    df = pd.read_csv('combination_results.csv')
    
    plot_combination_histogram(df, 'Combination_1')

    plot_combination_histogram(df, 'Combination_2')
    
    # # Apply the function to each nested list column
    pt_cols = unpack_nested_list(df['Pt_1'], 'Pt')
    eta_cols = unpack_nested_list(df['Eta_1'], 'Eta')
    phi_cols = unpack_nested_list(df['Phi_1'], 'Phi')

    # Concatenate these new columns with the original DataFrame
    df = pd.concat([ pt_cols, eta_cols, phi_cols], axis=1)

    # Calculate differences
    df['Delta_Pt'] = df['Pt_1'] - df['Pt_2']
    df['Delta_Eta'] = df['Eta_1'] - df['Eta_2']
    df['Delta_Phi'] = abs(df['Phi_1'] - df['Phi_2'])

    # Plotting the deltas
    plot_deltas(df)
    
    bjets_df = pd.read_csv('processed_sig_with_closest_masses.csv')

    # Plot the data
    plot_bjet_parameters(bjets_df)
    print(bjets_df['Pt_bj6'].describe())
    plt.show()
    
    masses_df = pd.read_csv('masses.csv')
    plot_mass_histogram_excluding_closest(masses_df)