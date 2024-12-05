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

# Function to get BankNifty options data using Yahoo Finance (for specific option strike and type)
def get_banknifty_ltp(strike_price, option_type):
    try:
        # Fetch BankNifty data using Yahoo Finance
        banknifty = yf.Ticker("^NSEBANK")
        # Get options chain data (calls and puts) for BankNifty
        options = banknifty.option_chain(datetime.today().strftime('%Y-%m-%d'))

        if option_type == "Call":
            option_data = options.calls
        else:
            option_data = options.puts

        # Find the option contract that matches the strike price
        matching_option = option_data[option_data['strike'] == strike_price]

        if not matching_option.empty:
            return matching_option['lastPrice'].iloc[0]
        else:
            return None
    except Exception as e:
        st.write(f"Error fetching options data: {e}")
        return None

# Function to get global market data (S&P500, AAPL) using Yahoo Finance
def get_global_market_data():
    try:
        # Get real-time market data for S&P500 (SPY) and AAPL from Yahoo Finance
        spy = yf.Ticker("^GSPC")  # S&P 500 Index
        aapl = yf.Ticker("AAPL")  # Apple Inc.

        # Get the most recent closing price for both S&P500 and AAPL
        spy_data = spy.history(period="1d", interval="1m")
        aapl_data = aapl.history(period="1d", interval="1m")

        spy_price = spy_data["Close"].iloc[-1]  # Latest closing price for S&P500
        aapl_price = aapl_data["Close"].iloc[-1]  # Latest closing price for AAPL
        
        return spy_price, aapl_price
    except Exception as e:
        st.write(f"Error fetching global market data: {e}")
        return None, None

# Function to calculate predicted LTP based on global market correlation
def predict_ltp(current_ltp, spy_price, aapl_price):
    # Use a simple correlation factor based on the relationship between BankNifty and the S&P 500 + AAPL
    # Simulated correlation factor (this can be refined with actual historical data)
    global_sentiment_factor = (spy_price * 0.0015) + (aapl_price * 0.02)
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
    ltp = get_banknifty_ltp(strike_price, option_type)
    
    if ltp is None:
        st.warning(f"Could not fetch LTP for strike price {strike_price} and option type {option_type}. Please try again later.")
    else:
        st.write(f"Current LTP for strike {strike_price} ({option_type} option): {ltp}")

        # Get global market data (S&P500 and AAPL)
        spy_price, aapl_price = get_global_market_data()
        if spy_price is None or aapl_price is None:
            st.warning("Could not fetch global market data. Please try again later.")
        else:
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time AAPL price: {aapl_price}")

            # Predict the LTP for the next day based on market data
            predicted_ltp = predict_ltp(ltp, spy_price, aapl_price)
            st.write(f"Predicted LTP for next day: {predicted_ltp}")

            # Predict Stop Loss and Maximum LTP
            stop_loss, max_ltp = predict_stop_loss_and_max_ltp(predicted_ltp)
            st.write(f"Stop Loss: {stop_loss}")
            st.write(f"Maximum LTP: {max_ltp}")

            # Predict Profit or Loss
            recommendation, profit_loss = predict_profit_or_loss(predicted_ltp, ltp, option_type)
            st.write(f"Recommendation: {recommendation}")
            st.write(f"Expected Profit/Loss: {profit_loss}")
