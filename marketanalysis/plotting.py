import threading
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


closing_prices_plot_lock = threading.Lock()
crossover_plot_lock = threading.Lock()
prediction_plot_lock = threading.Lock()
weekly_forecast_plot_lock = threading.Lock()


def determine_cross_signal(sma_dataframe):
    try:
        if sma_dataframe.empty or len(sma_dataframe) < 200:
            return 'Insufficient data for cross analysis (need at least 200 days)'
        
        required_cols = ['SMA100', 'SMA200', 'Close']
        for col in required_cols:
            if col not in sma_dataframe.columns:
                return f'Missing required column: {col}'
            if sma_dataframe[col].isna().all():
                return f'No valid data in {col} column'
        
        sma100_last = sma_dataframe['SMA100'].dropna().iloc[-1] if not sma_dataframe['SMA100'].dropna().empty else None
        sma200_last = sma_dataframe['SMA200'].dropna().iloc[-1] if not sma_dataframe['SMA200'].dropna().empty else None
        close_last = sma_dataframe['Close'].dropna().iloc[-1] if not sma_dataframe['Close'].dropna().empty else None
        
        if any(val is None for val in [sma100_last, sma200_last, close_last]):
            return 'Insufficient valid data for cross analysis'
        
        golden_cross = sma100_last > sma200_last
        death_cross = close_last < sma100_last and close_last < sma200_last

        if golden_cross:
            return 'A Golden Cross has been detected'
        elif death_cross:
            return 'A Death Cross has been detected'
        else: 
            return 'No significant cross has been detected'
            
    except Exception as e:
        return f'Error in cross signal analysis: {str(e)}'


def generate_closing_prices_plot(stockdataframe):
    with closing_prices_plot_lock:
        if not isinstance(stockdataframe.index, pd.DatetimeIndex):
            stockdataframe.index = pd.to_datetime(stockdataframe.index)
        
        close_series = stockdataframe['Close']
        if hasattr(close_series, 'values'):
            close_values = close_series.values
            if close_values.ndim > 1:
                close_values = close_values.flatten()
            close_prices = close_values.tolist()
        else:
            close_prices = list(close_series)
        
        dates = stockdataframe.index.tolist()
        
        fig = px.line(x=dates, y=close_prices, title='Stock Closing Prices Over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Closing Price ($)')
        plot_div = fig.to_html(full_html=False)
        return plot_div


def generate_crossover_plot(sma_dataframe):
    with crossover_plot_lock:
        dates = sma_dataframe['Date'].tolist() if 'Date' in sma_dataframe.columns else sma_dataframe.index.tolist()
        
        def safe_extract_values(series):
            """Safely extract values from a pandas Series"""
            if hasattr(series, 'values'):
                values = series.values
                if values.ndim > 1:
                    values = values.flatten()
                return values.tolist()
            else:
                return list(series)
        
        close_prices = safe_extract_values(sma_dataframe['Close'])
        sma100_values = safe_extract_values(sma_dataframe['SMA100'])
        sma200_values = safe_extract_values(sma_dataframe['SMA200'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=close_prices, mode='lines', name='Closing Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=sma100_values, mode='lines', name='SMA 100', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=dates, y=sma200_values, mode='lines', name='SMA 200', line=dict(color='green')))
        fig.update_layout(
            title='Stock Price with Moving Averages',
            xaxis_title='Date', 
            yaxis_title='Price ($)', 
            legend_title='Indicators'
        )
        
        cross_signal = determine_cross_signal(sma_dataframe)
        fig.add_annotation(
            text=f"{cross_signal}", 
            xref="paper", yref="paper", 
            x=0.5, y=0.95, 
            showarrow=False, 
            font=dict(color="white", size=12), 
            bgcolor="red", 
            opacity=0.8,
            bordercolor="white",
            borderwidth=1
        )
        plot_div = fig.to_html(full_html=False)
        return plot_div


def generate_prediction_vs_actual_plot(dates, y_test_flat, y_predicted_flat, future_prediction=False):
    with prediction_plot_lock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_test_flat, mode='lines', name='Actual Prices' , line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=y_predicted_flat, mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Prices')
        title = "Weekly Forecast" if future_prediction else "Prediction vs. Actual"
        return fig


def generate_weekly_forecast_plot(dates, y_predicted_flat):
    with weekly_forecast_plot_lock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_predicted_flat, mode='lines', name='Weekly Forecast', line=dict(color='green')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Prices')

        trend_direction = 'Bearish' if y_predicted_flat[0] > y_predicted_flat[-1] else 'Bullish'
        
        fig.add_annotation(text=f'{trend_direction} Trend', xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(size=12, color='black'), bgcolor="red", opacity=0.5)
        return fig 