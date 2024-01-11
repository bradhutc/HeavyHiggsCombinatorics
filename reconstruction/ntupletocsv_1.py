import os
import numpy as np
import pandas as pd
import uproot

def load_data(signal_file, tree_name):
    file_sig = uproot.open(signal_file)
    tree = file_sig[tree_name]
    signal_df = tree.arrays(library="pd") 
    print(signal_df.head())
    print("============================================")
    print("File loaded with ", len(signal_df), " events ")
    
    return signal_df

def preprocess_dataframe(signal_df):
    signal_df["label"] = 0
    print("Number of Signal Events Before Selections:", len(signal_df))
    selected_events = signal_df[['M_H1', 'Pt_H1', 'Eta_H1', 'Phi_H1','M_H2', 'Pt_H2', 'Eta_H2', 'Phi_H2',
        'Pt_bj1','Pt_bj2','Pt_bj3','Pt_bj4','Pt_bj5','Pt_bj6','Eta_bj1','Eta_bj2','Eta_bj3','Eta_bj4','Eta_bj5','Eta_bj6','Phi_bj1','Phi_bj2','Phi_bj3','Phi_bj4','Phi_bj5','Phi_bj6', 'num_bjets']].replace(-999.000000, np.nan).dropna()

    processed_sig_df = 'processed_sig.csv'
    selected_events.to_csv(processed_sig_df, index=False) 
    print("Processed data saved to:", processed_sig_df)

    print("Number of Signal:", len(selected_events))
    print("Number of Events Cut:", len(signal_df)-len(selected_events))
    
    return print(selected_events.head())

def main():
    signal_file = "C:\\Users\\Bradl\\OneDrive\\BH_CV\\ATLAS\\jetfindalg\\vlqupdated1000"
    tree_name = "sixtop"
    return preprocess_dataframe(load_data(signal_file, tree_name))

if __name__ == "__main__":
    main()

