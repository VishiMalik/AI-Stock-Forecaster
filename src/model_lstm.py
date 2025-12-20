import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape):
  model = models.Sequential([
    layers.LSTM(64, return_sequences = True, input_shape = input_shape),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences = False),
    layers.Dropout(0.2),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(1)
  ])
  model.compile(optimizer = 'adam', loss = 'mse')
  return model
                
