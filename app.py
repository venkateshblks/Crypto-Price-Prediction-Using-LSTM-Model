from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from datetime import timedelta
from keras.layers import LSTM, Dense, Dropout
import json
import plotly.graph_objs as go
from prophet import Prophet
import json
import plotly

app = Flask(__name__)

def fetch_data(symbol):
    data = yf.download(symbol, period='5d', interval='5m')
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
    return model, scaler, dataset

def predict_next_prices(model, scaler, dataset):
    last_sequence = dataset[-10:].reshape(1, 10, 1)
    next_predictions = []
    for _ in range(36):
        next_price = model.predict(last_sequence, verbose=0)
        next_predictions.append(next_price[0, 0])
        next_sequence = np.array([[next_price[0, 0]]])
        last_sequence = np.append(last_sequence[:, 1:, :], next_sequence.reshape(1, 1, 1), axis=1)
    return scaler.inverse_transform(np.array(next_predictions).reshape(-1, 1))

def predict_prophet(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=37, freq='5T')
    return model.predict(future)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        data = fetch_data(symbol)
        df=data.copy()
        df = df[df['Datetime'] >= (df['Datetime'].max() - timedelta(days=1))]
        data = create_features(data)
        
        model, scaler, dataset = training_model(data)
        next_predictions = predict_next_prices(model, scaler, data['Close'].values)

        df_prophet = data[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        prophet_forecast = predict_prophet(df_prophet)
   
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Candlestick(x=df['Datetime'] ,
                open=df['Open'],
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'] , 
                name='Historical Prices',))
        future_dates = pd.date_range(start=data['Datetime'].iloc[-1] + pd.Timedelta(minutes=5), periods=36, freq='5T')
        historical_fig.add_trace(go.Scatter(x=future_dates, y=next_predictions.flatten(), mode='lines', name='LSTM Predictions'))
        historical_fig.update_layout( title='Last 24 Hours Historical Prices and Predictions', xaxis_title='Time', yaxis_title='Price')

        df_prophet=df_prophet[df_prophet['ds'] >= (df_prophet['ds'].max() - timedelta(days=1))]
        prophet_forecast=prophet_forecast[prophet_forecast['ds'] >= (prophet_forecast['ds'].max() - timedelta(hours=28))]
        prophet_fig = go.Figure()
        prophet_fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Prices'))
        prophet_fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='fb Predictions'))
        prophet_fig.update_layout( title='Another Prediction Model', xaxis_title='Time', yaxis_title='Price')


        historical_graphJSON = json.dumps(historical_fig, cls=plotly.utils.PlotlyJSONEncoder)
        prophet_graphJSON = json.dumps(prophet_fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('index.html', symbol=symbol, historical_graphJSON=historical_graphJSON, prophet_graphJSON=prophet_graphJSON)

    return render_template('index.html', historical_graphJSON=None, prophet_graphJSON=None)

if __name__ == '__main__':
    app.run(debug=True)