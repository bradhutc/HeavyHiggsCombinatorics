import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout, LeakyReLU
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import os
import numpy as np

class HiggsPredictor(keras.Model):
    def __init__(self):
        super(HiggsPredictor, self).__init__()
        self.dense1 = Dense(64, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.001))
        self.dropout1 = Dropout(0.001)
        self.dense2 = Dense(64, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.001))
        self.dropout2 = Dropout(0.001)
        self.output_layer = Dense(2, activation='linear')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

class PlotHiggsMassesCallback(Callback):
    def __init__(self, X_test, Y_test, model, scaler_Y, epoch_interval=10, output_dir='plots'):
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = model
        self.scaler_Y = scaler_Y
        self.epoch_interval = epoch_interval
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        predicted_scaled = self.model.predict(self.X_test)
        predicted = self.scaler_Y.inverse_transform(predicted_scaled)
        original = self.scaler_Y.inverse_transform(self.Y_test)

        # Determine common bin edges
        all_values = np.concatenate([predicted.flatten(), original.flatten()])
        bins = np.linspace(all_values.min(), all_values.max(), 250)
        mean_pred_H1, std_pred_H1 = np.mean(predicted[:, 0]), np.std(predicted[:, 0])
        mean_orig_H1, std_orig_H1 = np.mean(original[:, 0]), np.std(original[:, 0])
        plt.figure(figsize=(10, 6))
        plt.hist(predicted[:, 0], bins=bins, alpha=0.5,color = 'red', label=f'Predicted H1_M (mean: {mean_pred_H1:.2f}, std: {std_pred_H1:.2f})')
        plt.hist(original[:, 0], bins=bins, alpha=0.5,color='blue', label=f'Original H1_M (mean: {mean_orig_H1:.2f}, std: {std_orig_H1:.2f})')
        # plt.hist(predicted[:, 1], bins=bins, alpha=0.5, label='Predicted H2_M')
        # plt.hist(original[:, 1], bins=bins, alpha=0.5, label='Original H2_M')
        plt.xlabel('Mass (GeV)')
        plt.ylabel('Frequency')
        plt.title(f'Heavy Higgs Mass Prediction from B-Jet Observables (Epoch {epoch + 1})')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'epoch_{epoch + 1}.png'))
        plt.close()



if __name__ == '__main__':
    data1500 = pd.read_csv('1500.csv')
    data200 = pd.read_csv('200.csv')

    data = pd.concat([data1500, data200], ignore_index=True)
    # X, Y = data.drop(['M_H1', 'Pt_H1', 'Eta_H1', 'Phi_H1','M_H2', 'Pt_H2', 'Eta_H2', 'Phi_H2'], axis=1), data[['M_H1', 'Pt_H1', 'Eta_H1', 'Phi_H1','M_H2', 'Pt_H2', 'Eta_H2', 'Phi_H2']]
    X, Y = data.drop(['M_H1', 'Pt_H1', 'Eta_H1', 'Phi_H1','M_H2', 'Pt_H2', 'Eta_H2', 'Phi_H2', 'num_bjets'], axis=1), data[['M_H1', 'M_H2']]
    # X, Y = data.drop(['H1_M', 'H2_M', 'num_bjets'], axis=1), data[['H1_M', 'H2_M']]
    
    # Scale inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale outputs
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
    
    batch_size = 8
    
    model = HiggsPredictor()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])
    plot_callback = PlotHiggsMassesCallback(X_test, Y_test, model, scaler_Y, epoch_interval=5)

    history = model.fit(
        X_train, Y_train, 
        batch_size=batch_size, epochs=40,
        validation_split=0.2, callbacks=[plot_callback]
        )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    test_loss, test_mse = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mse}")
    # Predictions
    predictions_scaled = model.predict(X_test)

    # Inverse transform predictions to original scale
    predictions = scaler_Y.inverse_transform(predictions_scaled)

    # Now predictions are in the original scale (GeV)
    print(predictions)