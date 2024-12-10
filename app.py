import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions", layout="wide")

# Title of the App
st.title("Trading Predictions with Machine Learning")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Sidebar for Inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)
risk_percent = st.sidebar.number_input("Enter Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
profit_percent = st.sidebar.number_input("Enter Profit Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to get financial sentiment using FinBERT
def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="60d", interval="1d")  # Fetch 60 days of historical data
        if data.empty:
            st.warning(f"No data available for {ticker}. Please check the ticker symbol or try again later.")
            return None, None
        current_price = data["Close"].iloc[-1] if not data["Close"].empty else None
        return current_price, data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to prepare features and target for ML model
def prepare_data(data):
    try:
        data['Returns'] = data['Close'].pct_change()  # Calculate daily returns
        data['Volatility'] = data['Close'].rolling(window=5).std()  # Rolling volatility
        data['Momentum'] = data['Close'] - data['Close'].rolling(window=5).mean()  # Momentum
        data = data.dropna()  # Drop rows with NaN values
        if data.empty:
            st.warning("Insufficient data to prepare features. Try again with a different ticker or timeframe.")
            return None, None
        X = data[['Open', 'High', 'Low', 'Volatility', 'Momentum']]
        y = data['Close']
        return X, y
    except Exception as e:
        st.write(f"Error preparing data: {e}")
        return None, None

# Function to train the ML model
def train_ml_model(data):
    X, y = prepare_data(data)
    if X is None or y is None:
        return None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model Mean Squared Error (MSE): {round(mse, 2)}")
        return model
    except Exception as e:
        st.write(f"Error training ML model: {e}")
        return None

# Function to fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1] if not data.empty else None
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Function to predict LTP for the selected ticker
def predict_ltp(model, X_latest):
    try:
        predicted_price = model.predict([X_latest])
        return round(predicted_price[0], 2)
    except Exception as e:
        st.write(f"Error predicting LTP: {e}")
        return None

# Main logic for prediction
def predict():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None or ticker_data is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    # Train the ML model
    model = train_ml_model(ticker_data)
    if model is None:
        st.warning("Failed to train the ML model.")
        return

    # Prepare the latest feature set for prediction
    X_latest, _ = prepare_data(ticker_data)
    if X_latest is None or X_latest.empty:
        st.warning("Insufficient data for prediction.")
        return
    X_latest = X_latest.iloc[-1].values  # Get the latest row of features

    # Predict LTP using ML model
    predicted_ltp = predict_ltp(model, X_latest)
    if predicted_ltp is None:
        return

    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    # Stop Loss and Max LTP
    stop_loss = round(predicted_ltp * (1 - (risk_percent / 100)), 2)
    max_ltp = round(predicted_ltp * (1 + (profit_percent / 100)), 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {max_ltp}")

    # Risk-to-Reward Ratio (RRR)
    rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and max_ltp else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    # Trading suggestion
    if rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

# Add a button to trigger prediction manually
if st.button("Get Prediction"):
    predict()
