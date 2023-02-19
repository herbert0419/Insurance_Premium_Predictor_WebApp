# import yfinance as yf
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Load historical data for ADANIENT.NS from Yahoo Finance
# ticker = "ADANIENT.NS"
# df = yf.download(ticker, start="2020-01-01", end="2022-02-18")

# # Create features and target variables
# df["SMA_20"] = df["Close"].rolling(window=20).mean()
# df["SMA_50"] = df["Close"].rolling(window=50).mean()
# df["SMA_200"] = df["Close"].rolling(window=200).mean()
# df["Return"] = df["Close"].pct_change()
# df.dropna(inplace=True)

# X = df[["SMA_20", "SMA_50", "SMA_200", "Return"]].values
# y = df["Close"].values

# # Split data into training and testing sets
# n = len(df)
# train_X, train_y = X[:int(n * 0.8)], y[:int(n * 0.8)]
# test_X, test_y = X[int(n * 0.8):], y[int(n * 0.8):]

# # Train linear regression model
# model = LinearRegression()
# model.fit(train_X, train_y)

# # Make predictions on test set
# predictions = model.predict(test_X)

# # Calculate mean absolute error (MAE) on test set
# mae = np.mean(np.abs(predictions - test_y))
# print("Mean absolute error:", mae)


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Download data
df = yf.download('ADANIENT.NS', start='2010-01-01', end='2023-02-19')

# Prepare data
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * 0.8))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Test model
test_data = scaled_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Visualize predictions
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

