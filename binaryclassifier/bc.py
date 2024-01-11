import pandas as pd
import matplotlib.pyplot as plt
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.metrics import AUC
from matplotlib import use
use('Agg')
import time
from extra_functions import compare_train_test


def load_data(signal_file, background_file, tree_name):
    file_sig = uproot.open(signal_file)
    tree = file_sig[tree_name]
    signal_df = tree.arrays(library="pd") 
    print("============================================")
    print("File loaded with ", len(signal_df), " events ")
    
    file_bkg = uproot.open(background_file)
    tree_bkg = file_bkg[tree_name]
    background_df = tree_bkg.arrays(library="pd") 
    print("============================================")
    print("File loaded with ", len(background_df), " events ")
    
    return signal_df, background_df

def preprocess_dataframe(signal_df, background_df):
    signal_df["label"] = 0
    background_df["label"] = 1
    print("Number of Signal Events Before Selections:", len(signal_df))
    
    selected_events = signal_df[
    (signal_df.Pt_bj1 > 30.0 ) & (signal_df.Pt_bj2 > 30.0 ) & (signal_df.Pt_bj3 > 30.0 ) & (signal_df.Pt_bj4 > 30.0 ) & (signal_df.Pt_bj5 > 30.0 ) & (signal_df.Pt_bj6 > 30.0 )
    & (abs(signal_df.Eta_bj1) < 4.5 ) & (abs(signal_df.Eta_bj2) < 4.5 ) & (abs(signal_df.Eta_bj3) < 4.5 ) & (abs(signal_df.Eta_bj4) < 4.5 ) & (abs(signal_df.Eta_bj5) < 4.5 ) & (abs(signal_df.Eta_bj6) < 4.5 )
    & (signal_df.met_NonInt > 15.0) & (signal_df.met_NonInt<400)& (signal_df.num_bjets == 6) & (signal_df != -999).any(axis=1) & (signal_df.isnull().any(axis=1)==False)
    ]
    
    # Remove later, this makes it so there is less signal than background.
    # selected_events = selected_events.sample(frac=0.5, random_state=42)


    # Getting a Clean Signal
   
    # Print the results
    print("Number of Signal:", len(selected_events))
    print("Number of Events Cut:", len(signal_df)-len(selected_events))
    selected_events.head()
    
    # ----------------- #
    print("Number of Events Before Selections:", len(background_df))
    selected_eventsbkg = background_df[
    (background_df.Pt_bj1 > 30.0 ) & (background_df.Pt_bj2 > 30.0 ) & (abs(background_df.Eta_bj1) < 4.5 ) & (abs(background_df.Eta_bj2) < 4.5 ) &
    (background_df.met_NonInt > 15.0) & (background_df.met_NonInt<400)& (background_df.num_bjets == 2) & (background_df != -999).any(axis=1) & (background_df.isnull().any(axis=1)==False)
    ]

    print("Number of Background:", len(selected_eventsbkg))
    print("Number of Events Cut:", len(background_df) - len(selected_eventsbkg))

    selected_eventsbkg.head()

    # ----------------- #
    
    # We must balance the impact of each class during training so that the NN gives more weight to the underrepresented class.

    selected_events["weight"] = 1
    selected_events.loc[:, "weight"] = (len(selected_events)+len(selected_eventsbkg)) / (2 * len(selected_events))
    selected_eventsbkg["weight"] = 1
    selected_eventsbkg.loc[:, "weight"] = (len(selected_events)+len(selected_eventsbkg)) / (2 * len(selected_eventsbkg))


    # ----------------- #


    merged_df=selected_events.combine_first(selected_eventsbkg)
    # merged_df=merged_df.drop("event_index", axis=1)
    merged_df = merged_df.fillna(-10)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)


    data=pd.DataFrame(merged_df, columns=[
    "Pt_bj1", "Pt_bj2", "Pt_bj3", "Pt_bj4", "Pt_bj5", "Pt_bj6",
    "Eta_bj1", "Eta_bj2", "Eta_bj3", "Eta_bj4",'Eta_bj5', 'Eta_bj6',
    "met_NonInt", "num_bjets"
    ])
    
    target = merged_df["label"]
    weights = merged_df["weight"]
    
    return data, target, weights

def split_data(data, target, weights, train_size):
    
    true_class_weights = [weights[target == 0].sum(), weights[target == 1].sum()]
    
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    data, target, weights, train_size=train_size
    )

    y_train, y_test, weights_train, weights_test = (
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        weights_train.reset_index(drop=True),
        weights_test.reset_index(drop=True),
    )

    print("X_train Shape: ", X_train.shape)
    print("y_train Shape: ", y_train.shape)
    print("Training Weights shape: ", weights_train.shape, "\n")
    print("X_test Shape: ", X_test.shape)
    print("y_test Shape: ", y_test.shape)
    print("Test Weights shape: ", weights_test.shape)

    (
        X_test,
        X_val,
        y_test,
        y_val,
        weights_test,
        weights_val,
    ) = train_test_split(X_test, y_test, weights_test, train_size=0.5, shuffle=False)

    print("Xtrain Shape: ", X_train.shape)
    print("ytrain Shape: ", y_train.shape)
    print("Training Weights: ", weights_train.shape, "\n")
    print("Xval Shape: ", X_val.shape)
    print("yval Shape: ", y_val.shape)
    print("Validation Weights: ", weights_val.shape, "\n")
    print("Xtest Shape: ", X_test.shape)
    print("ytest Shape: ", y_test.shape)
    print("Test Weights: ", weights_test.shape)

    prec = 2

    print("Before standardization:")
    print("X_train mean, std: ", X_train.to_numpy().mean(axis=0).round(prec), X_train.to_numpy().std(axis=0).round(prec))
    print("===")
    print("X_val mean, std: ", X_val.to_numpy().mean(axis=0).round(prec), X_val.to_numpy().std(axis=0).round(prec))
    print("===")
    print("X_test mean, std: ", X_test.to_numpy().mean(axis=0).round(prec), X_test.to_numpy().std(axis=0).round(prec))
    print("===")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    print("After standardization:")

    print("X_train mean, std: ", X_train.mean(axis=0).round(prec), X_train.std(axis=0).round(prec))
    print("===")
    print("X_val mean, std: ", X_val.mean(axis=0).round(prec), X_val.std(axis=0).round(prec))
    print("===")
    print("X_test mean, std: ", X_test.mean(axis=0).round(prec), X_test.std(axis=0).round(prec))


    class_weights_train = (weights_train[y_train == 0].sum(), weights_train[y_train == 1].sum())

    print("true_class_weights for (bkg, sig):", true_class_weights)

    for i in range(len(class_weights_train)):
        weights_train[y_train == i] *= (
        max(class_weights_train) / class_weights_train[i]
        )  # equalize number of background and signal event

    weights_val[y_val == i] *= max(class_weights_train) / class_weights_train[i]  # likewise for validation and test sets

    weights_test[y_test == i] *= (
        true_class_weights[i] / weights_test[y_test == i].sum()
    )  # increase test weight to compensate for sampling

    print("class_weights_train for (bkg, sig):", class_weights_train)
    print("Train : total weight sig", weights_train[y_train == 1].sum())
    print("Train : total weight bkg", weights_train[y_train == 0].sum())
    print("Validation : total weight sig", weights_val[y_val == 1].sum())
    print("Validation : total weight bkg", weights_val[y_val == 0].sum())
    print("Test : total weight sig", weights_test[y_test == 1].sum())
    print("Test : total weight bkg", weights_test[y_test == 0].sum())

    return X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test

def model_and_train(X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test, num_epochs, sel_batch_size, lr):
    model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    
        tf.keras.layers.Dense(16, activation='relu'),  # layer 1
        tf.keras.layers.Dense(8, activation='relu'),  # layer 2

        tf.keras.layers.Dense(1, activation='sigmoid'),  # output layer
        
    ]
    )
    
    model.compile(
    loss="binary_crossentropy",  # use loss="mean_squared_error" for MSE loss
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # tf.keras.optimizers.experimental.SGD for the SGD optimizer
    weighted_metrics=[AUC(name="auc")],  # monitor AUC for each epoch during the training
    )

    starting_time = time.time()
    print("Start training!")

    # model.fit() performs the training

    the_fit = model.fit(
        X_train,
        y_train.values,
        sample_weight=weights_train,
        epochs=num_epochs,
        batch_size=sel_batch_size,
        validation_data=(X_val, y_val, weights_val),
        # callbacks=[K.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')], # uncomment to use early stopping (should also increase the epochs)
        )


    y_pred_test = model.predict(X_test).ravel()
    y_pred_train = model.predict(X_train).ravel()

    compare_train_test(
    y_pred_train,
    y_train,
    y_pred_test,
    y_test,
    xlabel="NN Score",
    title="NN",
    weights_train=weights_train.values,
    weights_test=weights_test.values,
    density=True # set to True to normalize distributions
    )

    plt.savefig("NN.pdf")





    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(the_fit.history['loss'], color="tab:blue")
    ax.plot(the_fit.history['val_loss'], color="tab:blue", ls="--")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.tick_params(width=2, grid_alpha=0.5, labelsize=12)
    # ax.legend(fontsize=14, frameon=False, loc="upper right")

    ax.grid(True, axis='y')

    ax2 = ax.twinx()
    ax2.set_ylabel("ROC AUC", color='tab:orange', fontsize=14)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=12, axis='y', labelcolor='tab:orange')
    ax2.plot(the_fit.history['auc'], color='tab:orange', lw=2)
    ax2.plot(the_fit.history['val_auc'], color='tab:orange', ls='--', lw=2)
    # ax2.set_ylim([0, 1])

    ax2.plot([], [], color="black", label="training set")
    ax2.plot([], [], color="black", ls='--', label="validation set")
    # ax2.legend(fontsize=14, frameon=False, loc="right")
    fig.tight_layout()

    plt.savefig("ROCAUC.pdf")

    return "Done!"

def main():
    
    signal_file = "vlq_nn/sig400.root"
    background_file = "vlq_nn/bkg.root"
    tree_name = "sixtop"
    train_size = 0.75 # fraction of events used for training
    num_epochs = 100 # number of epochs
    sel_batch_size = 128 # number of events per batch fed to the NN
    lr = 0.0001 # learning rate
    
    dataframes = load_data(signal_file, background_file, tree_name)
    combined_df = preprocess_dataframe(dataframes[0], dataframes[1])
    data_split = split_data(combined_df[0], combined_df[1], combined_df[2], train_size)
    return model_and_train(data_split[0], data_split[1], data_split[2], data_split[3], data_split[4], data_split[5], data_split[6], data_split[7], data_split[8], num_epochs, sel_batch_size, lr) 

if __name__ == "__main__":
    main()

