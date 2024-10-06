# Crypto Price Prediction Using LSTM Model

## Overview
This project leverages a Long Short-Term Memory (LSTM) model to predict the next 24 hours of cryptocurrency price data based on three months of hourly real live data. The model has been integrated into a Flask web application, allowing users to easily access price predictions through a user-friendly interface. Check my notebook to understand the process.

## Features
- **Real-Time Predictions**: Provides predictions for the next 24 hours based on the latest available data.
- **Interactive Interface**: Users can view predicted prices alongside actual historical data.
- **Visualizations**: Displays graphs that illustrate the alignment of actual and predicted prices.
- **Easy Deployment**: Built with Flask, making it easy to run and deploy on various platforms.

## Demo
Check out the live demo of the application <a href="https://married-jody-njnsdcns-ca3da3b3.koyeb.app/" target="_blank">[Click Here]</a>

## Installation
To set up the project locally, follow these steps or  simply use GitHub Codespaces and skip the cloning step.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/venkateshblks/Crypto-Price-Prediction-Using-LSTM-Model
   cd Crypto-Price-Prediction-Using-LSTM-Model

2. **Install the required packages:**

```
pip install -r requirements.txt
```
3. **Run the Flask application:**

```
python app.py
```

Open your browser and go to http://127.0.0.1:5000/ to access the web app.

## Usage
1. Navigate to the web application.
2. Upon entering, you will be prompted to enter a cryptocurrency symbol.
3. After entering the symbol, click submit. The application will train the model using the latest live data and provide predictions for the next 24 hours.
4. Review the plots to visualize the predictions in relation to historical prices.



## Screenshots

![actual-predicted](https://github.com/user-attachments/assets/d8656a92-321c-44ee-a807-6ae7adad13d5)


![crypto-screen](https://github.com/user-attachments/assets/f1f68a56-03ed-47d0-bac3-94e799d16e0e)



## Conclusion
This project demonstrates the application of LSTM models in time-series forecasting and provides an accessible web interface for users to explore cryptocurrency price predictions in a dynamic market environment.
