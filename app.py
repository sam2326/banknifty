import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from nsepy.derivatives import get_expiry_date  # Import for NSEpy

# Streamlit UI setup
st.title("Enhanced Multi-Index Options Prediction App")
st.write("""
    This app predicts the next day's movement for options based on real-time market data, sentiment analysis, and strike recommendations.
    You can select multiple indices or stocks like **BankNifty**, **Nifty 50**, **Reliance**, and more.
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

# Fetch Available Expirations with Fallback Logic
def get_available_expirations(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            st.warning("No expiration dates available. Using default expiry.")
            return ["2024-12-28"]  # Replace with a valid fallback date
        return expirations
    except Exception as e:
        st.write(f"Error fetching available expirations: {e}")
        # Fallback to NSEpy for BankNifty
        if ticker == "^NSEBANK":
            expirations = get_expiry_date(year=2024, month=12)
            st.write(f"Available Expiry Dates for BankNifty: {expirations}")
            return expirations if expirations else ["2024-12-28"]
        return ["2024-12-28"]  # Default fallback date

# Display Expiry Date Dropdown
available_expirations = get_available_expirations(ticker_symbol)
expiry_date = st.selectbox("Select Expiry Date", available_expirations)

# Input fields
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

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

# Fetch Option Chain Data with Expiration Validation
def fetch_option_chain(ticker, expiry_date):
    try:
        ticker_obj = yf.Ticker(ticker)
        available_expirations = ticker_obj.options

        if expiry_date not in available_expirations:
            st.warning(f"Selected expiration {expiry_date} not found. Available expirations: {available_expirations}")
            return None, None

        # Fetch option chain for the valid expiry date
        options = ticker_obj.option_chain(expiry_date)
        return options.calls, options.puts
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None, None

# Recommend Strikes Based on Proximity, OI, and IV
def recommend_strikes(calls, puts, current_price):
    calls['Proximity'] = abs(calls['strike'] - current_price)
    puts['Proximity'] = abs(puts['strike'] - current_price)
    recommended_calls = calls.sort_values(by=['Proximity', 'openInterest'], ascending=[True, False]).head(3)
    recommended_puts = puts.sort_values(by=['Proximity', 'openInterest'], ascending=[True, False]).head(3)
    return recommended_calls, recommended_puts

# Main logic
if st.button("Get Prediction and Recommended Strikes"):
    if expiry_date:
        ticker_price = fetch_ticker_data(ticker_symbol)
        if ticker_price is None:
            st.warning(f"Could not fetch data for {ticker_name}.")
        else:
            st.write(f"Current price for {ticker_name}: {ticker_price}")

            # Fetch Option Chain
            calls, puts = fetch_option_chain(ticker_symbol, expiry_date)
            if calls is None or puts is None:
                st.warning("Could not fetch option chain data.")
            else:
                # Recommend Strikes
                recommended_calls, recommended_puts = recommend_strikes(calls, puts, ticker_price)

                # Display Recommended Strikes
                st.subheader("Recommended Call Strikes")
                st.dataframe(recommended_calls[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']])
                st.subheader("Recommended Put Strikes")
                st.dataframe(recommended_puts[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']])
