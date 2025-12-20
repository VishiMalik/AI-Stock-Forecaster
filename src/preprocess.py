import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_windows(prices, window_size = 60):
  X, y = [], []
  for i in range(window_size, len(prices)):
    X.append(prices[i-window_size:i])
    y.append(prices[i])
  return np.array(X), np.array(y)

def scale_series(series):
  scaler = MinMaxScaler(feature_range = (0,1))
  scaled = scaler.fit_transform(series.reshape(-1,1))
  return scaled, scaler
  
