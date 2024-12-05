import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# Streamlit UI setup
st.title("BankNifty Options Prediction for Intraday Trading")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    Enter the details for the BankNifty option you're interested in, and get predictions for the next day.
""")

# Input fields for the user
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])

# Function to get BankNifty current data using Yahoo Finance
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

# Function to get global market data (S&P500, Nifty 50) using Yahoo Finance
def get_global_market_data():
    try:
        # Get real-time market data for S&P500 and Nifty 50
        spy = yf.Ticker("^GSPC")  # S&P 500 Index
        nifty = yf.Ticker("^NSEI")  # Nifty 50 Index

        # Get the most recent closing price for both S&P500 and Nifty 50
        spy_data = spy.history(period="1d", interval="1m")
        nifty_data = nifty.history(period="1d", interval="1m")

        spy_price = spy_data["Close"].iloc[-1]  # Latest closing price for S&P500
        nifty_price = nifty_data["Close"].iloc[-1]  # Latest closing price for Nifty 50
        
        return spy_price, nifty_price
    except Exception as e:
        st.write(f"Error fetching global market data: {e}")
        return None, None

# Function to calculate LTP based on strike price and option type (approximation)
def calculate_ltp(banknifty_price, strike_price, option_type):
    if option_type == "Call":
        # For Call Option: LTP is roughly the difference between strike price and index price
        ltp = max(banknifty_price - strike_price, 0)  # Call options gain when the index is above strike
    elif option_type == "Put":
        # For Put Option: LTP is roughly the difference between strike price and index price
        ltp = max(strike_price - banknifty_price, 0)  # Put options gain when the index is below strike
    return round(ltp, 2)

# Function to calculate predicted LTP based on global market data
def predict_ltp(current_ltp, spy_price, nifty_price):
    # Using a simple correlation factor based on global market and Nifty 50 (for Indian market relevance)
    global_sentiment_factor = (spy_price * 0.0015) + (nifty_price * 0.005)
    predicted_ltp = current_ltp + global_sentiment_factor
    return round(predicted_ltp, 2)

# Simulate prediction of stop loss and maximum LTP based on volatility
def predict_stop_loss_and_max_ltp(predicted_ltp):
    # Based on market volatility, setting stop loss and max LTP (±1.5% for stop loss, ±2% for max LTP)
    stop_loss = predicted_ltp * 0.985  # 1.5% below the predicted LTP
    max_ltp = predicted_ltp * 1.02  # 2% above the predicted LTP
    return round(stop_loss, 2), round(max_ltp, 2)

# Predict if the option is profitable to buy based on predicted LTP
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
    # Fetch the current LTP for the selected option
    real_time_data = get_banknifty_data()
    
    if real_time_data is None:
        st.warning("Could not fetch real-time BankNifty data. Please try again later.")
    else:
        st.write(f"Current BankNifty index price: {real_time_data}")

        # Get global market data (S&P500 and Nifty 50)
        spy_price, nifty_price = get_global_market_data()
        if spy_price is None or nifty_price is None:
            st.warning("Could not fetch global market data. Please try again later.")
        else:
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time Nifty 50 price: {nifty_price}")

            # Calculate the LTP for the selected strike price and option type (Call/Put)
            ltp = calculate_ltp(real_time_data, strike_price, option_type)
            st.write(f"Calculated LTP for strike {strike_price} ({option_type} option): {ltp}")

            # Predict the LTP for the next day based on market data
            predicted_ltp = predict_ltp(ltp, spy_price, nifty_price)
            st.write(f"Predicted LTP for next day: {predicted_ltp}")

            # Predict Stop Loss and Maximum LTP
            stop_loss, max_ltp = predict_stop_loss_and_max_ltp(predicted_ltp)
            st.write(f"Stop Loss: {stop_loss}")
            st.write(f"Maximum LTP: {max_ltp}")

            # Predict Profit or Loss
            recommendation, profit_loss = predict_profit_or_loss(predicted_ltp, ltp, option_type)
            st.write(f"Recommendation: {recommendation}")
            st.write(f"Expected Profit/Loss: {profit_loss}")
