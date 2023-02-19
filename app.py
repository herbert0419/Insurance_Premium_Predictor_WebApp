import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# Load historical data for ADANIENT.NS from Yahoo Finance
ticker = "ADANIENT.NS"
df = yf.download(ticker, start="2020-01-01", end="2022-02-18")

# Create features and target variables
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

X = df[["SMA_20", "SMA_50", "SMA_200", "Return"]].values
y = df["Close"].values

# Split data into training and testing sets
n = len(df)
train_X, train_y = X[:int(n * 0.8)], y[:int(n * 0.8)]
test_X, test_y = X[int(n * 0.8):], y[int(n * 0.8):]

# Train linear regression model
model = LinearRegression()
model.fit(train_X, train_y)

# Make predictions on test set
predictions = model.predict(test_X)

# Calculate mean absolute error (MAE) on test set
mae = np.mean(np.abs(predictions - test_y))
print("Mean absolute error:", mae)
