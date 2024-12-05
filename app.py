import alpaca_trade_api as tradeapi
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Alpaca API credentials
api_key = "PK5GIYLW7TNM8DYGRYA6"
api_secret = "CgPffxsbhFZBATQ2F79C9OGONDVw5RHoray5TBPT"
endpoint = "https://paper-api.alpaca.markets/v2"

# Initialize Alpaca API connection
api = tradeapi.REST(api_key, api_secret, base_url=endpoint)

# Streamlit UI setup
st.title("BankNifty Options Prediction for Intraday Trading")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    Enter the details for the BankNifty option you're interested in.
""")

# Input fields for the user
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=35000)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Last Traded Price (LTP)", min_value=0, value=100)

# Function to get BankNifty data using Yahoo Finance
def get_banknifty_data():
    try:
        # Fetch BankNifty data using Yahoo Finance
        banknifty = yf.Ticker("^NSEBANK")  # ^NSEBANK is the BankNifty index symbol
        banknifty_data = banknifty.history(period="1d", interval="1m")  # 1-minute data for the last day
        current_price = banknifty_data["Close"].iloc[-1]  # Get the most recent closing price
        return current_price
    except Exception as e:
        st.write(f"Error fetching BankNifty data: {e}")
        return None

# Function to get real-time global market data (S&P500, AAPL) from Alpaca
def get_global_market_data():
    try:
        # Get real-time market data for S&P500 (SPY) and AAPL from Alpaca
        spy = api.get_last_trade("SPY")  # S&P 500 ETF
        aapl = api.get_last_trade("AAPL")  # Apple Inc.

        spy_price = spy.price
        aapl_price = aapl.price
        return spy_price, aapl_price
    except Exception as e:
        st.write(f"Error fetching global market data from Alpaca: {e}")
        return None, None

# Simulate prediction of the LTP for the next day based on current market conditions
def predict_ltp(current_price, global_sentiment_factor):
    # Simulating a prediction based on real-time market data (you can enhance this with actual ML models)
    market_change_percentage = random.uniform(-0.02, 0.02)  # Random change between -2% to +2%
    predicted_ltp = current_price * (1 + market_change_percentage + global_sentiment_factor)
    return round(predicted_ltp, 2)

# Simulate prediction of stop loss and maximum LTP
def predict_stop_loss_and_max_ltp(predicted_ltp):
    # Example logic to predict stop loss and maximum LTP (based on market volatility)
    stop_loss = predicted_ltp * 0.98  # 2% below the predicted LTP
    max_ltp = predicted_ltp * 1.02  # 2% above the predicted LTP
    return round(stop_loss, 2), round(max_ltp, 2)

# Predict if the option is profitable to buy
def predict_profit_or_loss(predicted_ltp, ltp, option_type):
    if option_type == "Call":
        if predicted_ltp > ltp:
            return "Profit", predicted_ltp - ltp
        else:
            return "Loss", ltp - predicted_ltp
    else:  # Put Option
        if predicted_ltp < ltp:
            return "Profit", ltp - predicted_ltp
        else:
            return "Loss", predicted_ltp - ltp

# Main logic
if st.button("Get Prediction"):
    # Get BankNifty real-time data
    real_time_data = get_banknifty_data()
    if real_time_data is None:
        st.warning("Could not fetch real-time BankNifty data. Please try again later.")
    else:
        st.write(f"Real-time BankNifty index price: {real_time_data}")

        # Get global market data (S&P500 and AAPL)
        spy_price, aapl_price = get_global_market_data()
        if spy_price is None or aapl_price is None:
            st.warning("Could not fetch global market data. Please try again later.")
        else:
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time AAPL price: {aapl_price}")

            # Global market sentiment factor (based on S&P 500 and AAPL)
            global_sentiment_factor = (spy_price + aapl_price) / 100000  # Simple example factor

            # Predict the LTP for the next day
            predicted_ltp = predict_ltp(real_time_data, global_sentiment_factor)
            st.write(f"Predicted LTP for next day: {predicted_ltp}")

            # Predict Stop Loss and Maximum LTP
            stop_loss, max_ltp = predict_stop_loss_and_max_ltp(predicted_ltp)
            st.write(f"Stop Loss: {stop_loss}")
            st.write(f"Maximum LTP: {max_ltp}")

            # Predict Profit or Loss
            recommendation, profit_loss = predict_profit_or_loss(predicted_ltp, ltp, option_type)
            st.write(f"Recommendation: {recommendation}")
            st.write(f"Expected Profit/Loss: {profit_loss}")
