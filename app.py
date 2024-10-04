from flask import Flask, render_template, request,jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import json
import plotly.graph_objs as go
from prophet import Prophet
import json
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly
import requests
from concurrent.futures import ThreadPoolExecutor
import os
executor = ThreadPoolExecutor(max_workers=2)
app = Flask(__name__)

def fetch_data(symbol):
    data = yf.download(symbol, period='5d', interval='5m')
    if data.empty:
        return 'error'
    data.reset_index(inplace=True)
    data['Datetime'] = data['Datetime'] + pd.Timedelta(hours=5, minutes=30)
    return data.drop(['Adj Close', 'Volume'], axis=1)

def create_features(data):
    for lag in range(1, 6):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['Volatility'] = data['Close'].rolling(window=5).std()
    data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
    data['Pct_Change'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

def predict_prophet(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=37, freq='5min')
    return model.predict(future)
def create_prophet_plot(symbol):
    data = fetch_data(symbol)
    if 'error' in data:
        return None
    df_prophet = data[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    prophet_forecast = predict_prophet(df_prophet)
    df_prophet = df_prophet[df_prophet['ds'] >= (df_prophet['ds'].max() - timedelta(days=1))]
    prophet_forecast = prophet_forecast[prophet_forecast['ds'] >= (prophet_forecast['ds'].max() - timedelta(hours=28))]

    prophet_fig = go.Figure()
    prophet_fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Prices'))
    prophet_fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Predictions'))
    prophet_fig.update_layout(title='Meta Prophet Prediction Model', xaxis_title='Time', yaxis_title='Price')    
    return json.dumps(prophet_fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
    return np.array(X)

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
    return model, scaler, data

def predict_next_prices(model, scaler, dataset):
    last_sequence = dataset[-10:].reshape(1, 10, 1)
    next_predictions = []
    for _ in range(36):
        next_price = model.predict(last_sequence, verbose=0)
        next_predictions.append(next_price[0, 0])
        next_sequence = np.array([[next_price[0, 0]]])
        last_sequence = np.append(last_sequence[:, 1:, :], next_sequence.reshape(1, 1, 1), axis=1)
    return scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1)).flatten().tolist()


@app.route('/historical_plot', methods=['POST'])
def historical_plot():
    symbol = request.json.get('symbol')
    data = fetch_data(symbol)
    df = data.copy()
    if 'error' in data:
        return None
    data=create_features(data)
    model, scaler, data = training_model(data)
    
    next_predictions = predict_next_prices(model, scaler, data['Close'].values)  
    df = df[df['Datetime'] >= (df['Datetime'].max() - timedelta(days=1))]

    historical_fig = go.Figure()
    historical_fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Historical Prices',
    ))

    future_dates = pd.date_range(start=data['Datetime'].iloc[-1] + pd.Timedelta(minutes=5), periods=36, freq='5min')


    # next_predictions=[61523.45703125, 61535.1171875, 61546.578125, 61558.66015625, 61573.515625, 61584.98828125, 61595.36328125, 61610.4140625, 61620.92578125, 61631.8203125, 61643.625, 61655.5234375, 61667.359375, 61679.14453125, 61690.88671875, 61702.5, 61714.1171875, 61725.765625, 61737.29296875, 61748.85546875, 61760.42578125, 61771.9765625, 61783.4921875, 61794.98828125, 61806.46484375, 61817.91796875, 61829.36328125, 61840.79296875, 61852.203125, 61863.6015625, 61874.984375, 61886.35546875, 61897.7109375, 61909.0546875, 61920.3828125, 61931.703125]
    
    historical_fig.add_trace(go.Scatter(x=future_dates, y=next_predictions, mode='lines', name='LSTM Predictions'))
    historical_fig.update_layout(title='LSTM (Long Short-Term Memory) Prediction Model', xaxis_title='Time', yaxis_title='Price')
    

    return jsonify(historical_graphJSON=json.dumps(historical_fig, cls=plotly.utils.PlotlyJSONEncoder))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']

        
        # Create Prophet plot asynchronously
        # future_prophet = executor.submit(create_prophet_plot, symbol)
        
        future_prophet = executor.submit(create_prophet_plot, symbol)
        
        # Wait for results
        prophet_graphJSON = future_prophet.result()
        if not prophet_graphJSON:
            
            return render_template('index.html', prophet_graphJSON=None,error='error')
        
        # Wait for results
        # prophet_graphJSON = create_prophet_plot(symbol)

        return render_template('index.html', symbol=symbol, prophet_graphJSON=prophet_graphJSON)

    return render_template('index.html', prophet_graphJSON=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0',port=port)