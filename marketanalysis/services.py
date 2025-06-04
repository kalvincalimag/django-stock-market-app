import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import time
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from tensorflow.keras.models import Sequential, load_model


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'keras_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'keras_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')


def create_lstm_model():
    """
    Create the LSTM model with the exact architecture specified by the user
    """
    
    layers = [
        (LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(100, 1)), 0.2),
        (LSTM(units=60, activation='tanh', return_sequences=True), 0.3),
        (LSTM(units=80, activation='tanh', return_sequences=True), 0.4),
        (LSTM(units=120, activation='tanh'), 0.5),
        (Dense(units=1), None)
    ]

    model = Sequential()

    for layer, dropout_rate in layers:
        if dropout_rate is not None:
            model.add(layer)
            model.add(Dropout(dropout_rate))
        else:
            model.add(layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

