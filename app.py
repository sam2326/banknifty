import yfinance as yf
import streamlit as st
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests

# Streamlit UI setup
st.title("Multi-Index Options Prediction App")
st.write("""
    This app predicts the next day's movement for options based on real-time market data.
    Now, you can select multiple indices or stocks like **BankNifty**, **Nifty 50**, and even individual stocks.
""")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Dropdown for selecting ticker
ticker_name = st.selectbox("Select Index/Stock", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]

# Input fields
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

# Function to predict LTP for the selected ticker
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix):
    sentiment_factor = india_vix * 0.1
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + strike_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main logic for prediction
if st.button("Get Prediction"):
    ticker_price = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {ticker_price}")

        # Fetch India VIX
        india_vix_ticker = yf.Ticker("^INDIAVIX")
        try:
            india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            india_vix = 15.0  # Default VIX value if fetching fails
            st.write("Warning: Using default India VIX value.")

        st.write(f"India VIX: {india_vix}")

        # Predict LTP
        predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix)
        st.write(f"Predicted LTP for next day: {predicted_ltp}")

        # Stop Loss and Max LTP
        stop_loss = predicted_ltp * 0.98
        max_ltp = predicted_ltp * 1.02
        st.write(f"Stop Loss: {round(stop_loss, 2)}")
        st.write(f"Maximum LTP: {round(max_ltp, 2)}")

        # Recommendation
        if predicted_ltp > ltp:
            st.write("Recommendation: Profit")
            st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
        else:
            st.write("Recommendation: Loss")
            st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")
