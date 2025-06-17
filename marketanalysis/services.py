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


def get_company_info(ticker_symbol):
    """
    Get company information and stock exchange details
    """
    stock_info_mapping = {
        "NMS": "NASDAQ",
        "NYQ": "NYSE",
        "NGM": "NASDAQ Global Market",
        "NIM": "NASDAQ Capital Market",
        "ASE": "NYSE American",
    }
    
    ticker_info = yf.Ticker(ticker_symbol)
    company_name = "N/A"
    stock_exchange = "N/A"
    
    try:
        info = ticker_info.info
        company_name = info.get('longName', info.get('shortName', ticker_symbol.upper()))
        if not company_name:
            company_name = ticker_symbol.upper()
    except:
        company_name = ticker_symbol.upper()
        
    try:
        exchange_code = ticker_info.fast_info.get('exchange', 'Unknown')
        stock_exchange = stock_info_mapping.get(exchange_code, exchange_code)
    except:
        stock_exchange = "Unknown"
    
    return company_name, stock_exchange


def calculate_sma_dataframe(stockdataframe):
    """
    Calculate Simple Moving Averages (SMA) for crossover analysis
    """
    close_prices = stockdataframe['Close'].values.flatten()
    
    if len(close_prices) < 200:
        raise ValueError(f"Insufficient data for crossover analysis. Need at least 200 days of data, but only have {len(close_prices)} days.")
    
    simple_moving_avg_100 = pd.Series(close_prices).rolling(100).mean()
    simple_moving_avg_200 = pd.Series(close_prices).rolling(200).mean()

    sma_dataframe = pd.DataFrame({
        'Date': stockdataframe.index,
        'Close': close_prices,
        'SMA100': simple_moving_avg_100,
        'SMA200': simple_moving_avg_200
    })
    
    return sma_dataframe


def prepare_lstm_data(stockdataframe):
    """
    Prepare data for LSTM model training and testing
    """
    if len(stockdataframe) < 150:
        raise ValueError(f"Insufficient data for prediction analysis. Need at least 150 days of data, but only have {len(stockdataframe)} days. Please try a stock with more historical data.")
    
    training_70 = pd.DataFrame(stockdataframe['Close'][0:int(len(stockdataframe) * 0.70)])
    testing_30 = pd.DataFrame(stockdataframe['Close'][int(len(stockdataframe) * 0.70): int(len(stockdataframe))])
    
    if len(testing_30) < 30:
        raise ValueError(f"Insufficient testing data for prediction. Need more historical data for this ticker.")
    
    return training_70, testing_30


def train_lstm_model(training_data):
    """
    Train LSTM model with the training data
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(training_data)

    model = create_lstm_model()
    
    x_train = []
    y_train = []
    
    for i in range(100, len(data_training_array)):
        x_train.append(data_training_array[i-100:i, 0])
        y_train.append(data_training_array[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print(f"Training LSTM model with {len(x_train)} samples...")
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    print("✓ Model training completed")
    
    return model, scaler


def make_predictions(model, scaler, training_data, testing_data):
    """
    Make predictions using the trained LSTM model
    """
    past_100_days = training_data.tail(100)
    final_dataframe = pd.concat([past_100_days, testing_data], ignore_index=True)
    input_data = scaler.fit_transform(final_dataframe)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    if len(x_test) == 0:
        raise ValueError(f"Not enough data to create test samples for prediction. Need more historical data.")
    
    y_predicted = model.predict(x_test)
    scaler_scale = scaler.scale_
    
    scale_factor = 1 / scaler_scale[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    y_test_flat = y_test.flatten()
    y_predicted_flat = y_predicted.flatten()
    
    return y_test_flat, y_predicted_flat, input_data, scale_factor


