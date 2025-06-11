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


def fetch_stock_data(ticker_symbol, start_date='2010-01-01', end_date=None, max_retries=3):
    """
    Fetch stock data from yfinance with retry logic and error handling
    """
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    stockdataframe = None
    error_message = None
    
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            stockdataframe = ticker_obj.history(start=start_date, end=end_date, timeout=10)
            
            if not stockdataframe.empty and 'Close' in stockdataframe.columns:
                break
            else:
                if attempt == max_retries - 1:
                    error_message = f"No data available for ticker {ticker_symbol}. Please verify the ticker symbol is correct."
                    
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker_symbol}: {str(e)}")
            if attempt == max_retries - 1:
                error_message = f"Unable to fetch data for {ticker_symbol}. This could be due to network issues or the ticker symbol may be invalid."
            else:
                time.sleep(1)
    
    return stockdataframe, error_message


