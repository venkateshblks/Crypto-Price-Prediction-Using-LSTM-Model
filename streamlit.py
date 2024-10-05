import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from prophet import Prophet

# @st.cache_data()
def fetch_data(symbol):
    # Fetch last 5 days of data at 5-minute intervals
    data = yf.download(symbol, period='3mo', interval='1h')
    data.reset_index(inplace=True)
    # Convert to IST
    data['Datetime'] = data['Datetime'] + pd.Timedelta(hours=5, minutes=30)
    df = data.drop([ 'Adj Close', 'Volume'], axis=1)
    return df
# @st.cache_data()
def create_features(data):
    for lag in range(1, 6):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    

    data['Volatility'] = data['Close'].rolling(window=5).std()

    data['Momentum_5'] = data['Close'] - data['Close'].shift(5)

    data['Pct_Change'] = data['Close'].pct_change()

    # Bollinger Bands
    data['Bollinger_Upper'] = data['SMA_5'] + (data['Volatility'] * 2)
    data['Bollinger_Lower'] = data['SMA_5'] - (data['Volatility'] * 2)

    data.dropna(inplace=True)
    return data

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
    return np.array(X)
# @st.cache_data()
def predict_prophet(data):
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=120, freq='1h')
    forecast = model.predict(future)
    return forecast

# @st.cache_data()
def training_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data[['Close']])
    for lag in range(1, 6):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    
    data.dropna(inplace=True)

    dataset = data['Close'].values.reshape(-1, 1)
    X = create_dataset(dataset, time_step=10)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, dataset[10:], epochs=10, batch_size=16, verbose=0)

    return model, scaler, dataset

  
def predict_next_prices(model, scaler, dataset):
    last_sequence = dataset[-10:].reshape(1, 10, 1)
    next_predictions = []

    for _ in range(120):
        next_price = model.predict(last_sequence,verbose=0)
        next_predictions.append(next_price[0, 0])
        next_sequence = np.array([[next_price[0, 0]]])
        last_sequence = np.append(last_sequence[:, 1:, :], next_sequence.reshape(1, 1, 1), axis=1)

    next_predictions = scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1))
    return next_predictions

st.title("Live Stock Price Prediction")

symbol = st.text_input("Enter stock symbol (e.g., BTC-USD):", value='BTC-USD')

if st.button("Fetch Data"):
    data = fetch_data(symbol)
    data = create_features(data)
    df=data.copy()
    st.write(data)

    model, scaler, dataset = training_model(data)
    next_predictions = predict_next_prices(model, scaler, data['Close'].values)##.reshape(-1, 1))

    plt.figure(figsize=(14, 5))
    plt.plot(data['Datetime'], data['Close'], color='blue', label='Historical Prices')
    plt.title(f'Historical Prices for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(14, 5))
    # future_dates = pd.date_range(start=data['Datetime'].iloc[-1], periods=31, freq='5T')[1:]
    future_dates = pd.date_range(start=data['Datetime'].iloc[-1] + pd.Timedelta(minutes=5), periods=120, freq='1h')  # Next 30 timestamps
    plt.plot(future_dates, next_predictions, color='red', label='Next 5 days Predicted Prices')
    plt.title(f'Predicted Prices for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    df_prophet = data[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None) 
    prophet_forecast = predict_prophet(df_prophet)

    df_prophet = data[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Candlestick(x=df['Datetime'] ,
                open=df['Open'],
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'] , 
                name='Historical Prices',))
    future_dates = pd.date_range(start=data['Datetime'].iloc[-1], periods=120, freq='1h')[1:]
    fig_lstm.add_trace(go.Scatter(x=future_dates, y=next_predictions.flatten(), mode='lines', name='Long Short-Term Memory Predictions'))
    fig_lstm.update_layout(
       
        title='Long Short-Term Memory Predictions',
        xaxis_title='Time (IST)',
        yaxis_title='Close Price',
        xaxis_rangeslider_visible=True,

    )   
    st.plotly_chart(fig_lstm)

    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Prices'))
    fig_prophet.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Prophet Predictions'))
    fig_prophet.update_layout(xaxis_rangeslider_visible=True,
    title='Facebook Prophet Predictions', xaxis_title='Time (IST)', yaxis_title='Close Price')
    st.plotly_chart(fig_prophet)
