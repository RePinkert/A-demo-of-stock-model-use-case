import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv('stock_data.csv')
data.drop('DATE', axis=1, inplace=True)

# Split into training and testing
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]

scaler = MinMaxScaler(feature_range=(-1, 1))
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Split features and labels
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Define and train feedforward neural network
input_dim = X_train.shape[1]
hidden_1, hidden_2, hidden_3, hidden_4 = 1024, 512, 256, 128
output_dim = 1

# Build FFNN model
inputs = Input(shape=(input_dim,))
x = Dense(hidden_1, activation='relu')(inputs)
x = Dense(hidden_2, activation='relu')(x)
x = Dense(hidden_3, activation='relu')(x)
x = Dense(hidden_4, activation='relu')(x)
outputs = Dense(output_dim)(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

# Train and evaluate FFNN model
max_epochs = 100
mse_threshold = 0.003
epochs = 0
while epochs < max_epochs:
    model.fit(X_train, y_train, epochs=1, batch_size=256, shuffle=True, verbose=0)
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    
    print(f"Iteration {epochs+1}: Train MSE: {train_loss}, Test MSE: {test_loss}")
    
    if test_loss < mse_threshold:
        print("Test MSE is below threshold. Stopping training.")
        break
    epochs += 1

plt.plot(y_test, label='Test')
plt.plot(y_pred, label='Prediction')
plt.legend()
plt.show()

# Prepare data for LSTM
seq_len = 5
hidden_size = 128

X_train_seq = np.array([data_train[i:i + seq_len, 0] for i in range(len(data_train) - seq_len)])[:, :, np.newaxis]
y_train_seq = np.array([data_train[i + seq_len, 0] for i in range(len(data_train) - seq_len)])
X_test_seq = np.array([data_test[i:i + seq_len, 0] for i in range(len(data_test) - seq_len)])[:, :, np.newaxis]
y_test_seq = np.array([data_test[i + seq_len, 0] for i in range(len(data_test) - seq_len)])

# Build LSTM model
inputs = Input(shape=(seq_len, 1))
x = LSTM(hidden_size, activation='relu')(inputs)
outputs = Dense(1)(x)

model_lstm = Model(inputs, outputs)
model_lstm.compile(optimizer='adam', loss='mse')

# Train and evaluate LSTM model
epochs = 0
while epochs < max_epochs:
    model_lstm.fit(X_train_seq, y_train_seq, epochs=1, batch_size=256, shuffle=True, verbose=0)
    train_loss_lstm = model_lstm.evaluate(X_train_seq, y_train_seq, verbose=0)
    test_loss_lstm = model_lstm.evaluate(X_test_seq, y_test_seq, verbose=0)
    y_pred_lstm = model_lstm.predict(X_test_seq)
    
    print(f"Iteration {epochs+1}: Train MSE (LSTM): {train_loss_lstm}, Test MSE (LSTM): {test_loss_lstm}")
    
    if test_loss_lstm < mse_threshold:
        print("Test MSE is below threshold. Stopping training.")
        break
    epochs += 1

plt.plot(y_test_seq, label='Test')
plt.plot(y_pred_lstm, label='Prediction')
plt.legend()
plt.show()
