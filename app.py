import yfinance as yf
import streamlit as st
from datetime import datetime
import random

# Streamlit UI setup
st.title("BankNifty Options Prediction with Volatility and Support/Resistance Levels")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    It also calculates Support and Resistance levels to guide your trading decisions and adjusts Stop Loss/Targets based on market volatility.
""")

# Input fields for the user
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to get BankNifty data using Yahoo Finance
def get_banknifty_data():
    try:
        banknifty = yf.Ticker("^NSEBANK")
        banknifty_data = banknifty.history(period="1d", interval="1m")  # 1-minute data for the last day
        current_price = banknifty_data["Close"].iloc[-1]
        high_price = banknifty_data["High"].max()
        low_price = banknifty_data["Low"].min()
        close_price = banknifty_data["Close"].iloc[-1]
        return current_price, high_price, low_price, close_price
    except Exception as e:
        st.write(f"Error fetching BankNifty data: {e}")
        return None, None, None, None

# Function to fetch India VIX data
def get_india_vix():
    try:
        vix = yf.Ticker("^INDIAVIX")
        vix_data = vix.history(period="1d", interval="1m")
        india_vix = vix_data["Close"].iloc[-1]
        return india_vix
    except Exception as e:
        st.write(f"Error fetching India VIX data: {e}")
        return None

# Function to calculate support and resistance levels
def calculate_support_resistance(high, low, close):
    pivot_point = (high + low + close) / 3
    r1 = 2 * pivot_point - low
    s1 = 2 * pivot_point - high
    r2 = pivot_point + (high - low)
    s2 = pivot_point - (high - low)
    return round(pivot_point, 2), round(r1, 2), round(s1, 2), round(r2, 2), round(s2, 2)

# Volatility adjustment for Stop Loss and Maximum LTP
def adjust_for_volatility(predicted_ltp, india_vix):
    stop_loss = predicted_ltp * (1 - 0.015 * (1 + india_vix / 20))  # Dynamic Stop Loss
    max_ltp = predicted_ltp * (1 + 0.02 * (1 + india_vix / 20))  # Dynamic Max LTP
    return round(stop_loss, 2), round(max_ltp, 2)

# Function to predict LTP
def predict_ltp(current_ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix):
    global_sentiment_factor = (spy_price * 0.0015) + (nifty_price * 0.005) + (india_vix * 0.1)

    if strike_price > banknifty_price:
        strike_impact_factor = (strike_price - banknifty_price) * -0.01
    else:
        strike_impact_factor = (banknifty_price - strike_price) * 0.01

    random_factor = random.uniform(-0.01, 0.02)

    predicted_ltp = current_ltp + global_sentiment_factor + strike_impact_factor + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main logic
if st.button("Get Prediction"):
    banknifty_price, high_price, low_price, close_price = get_banknifty_data()
    india_vix = get_india_vix()
    
    if banknifty_price is None or india_vix is None:
        st.warning("Could not fetch real-time BankNifty or volatility data. Please try again later.")
    else:
        st.write(f"Current BankNifty index price: {banknifty_price}")
        st.write(f"High Price: {high_price}, Low Price: {low_price}, Close Price: {close_price}")
        st.write(f"Current India VIX: {india_vix}")

        # Calculate and display support and resistance levels
        pivot, r1, s1, r2, s2 = calculate_support_resistance(high_price, low_price, close_price)
        st.write(f"Pivot Point: {pivot}")
        st.write(f"Resistance Levels: R1 = {r1}, R2 = {r2}")
        st.write(f"Support Levels: S1 = {s1}, S2 = {s2}")

        # Fetch global market data
        spy = yf.Ticker("^GSPC")
        nifty = yf.Ticker("^NSEI")

        try:
            spy_price = spy.history(period="1d", interval="1m")["Close"].iloc[-1]
            nifty_price = nifty.history(period="1d", interval="1m")["Close"].iloc[-1]
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time Nifty 50 price: {nifty_price}")
        except Exception as e:
            st.warning(f"Error fetching global market data: {e}")
            spy_price, nifty_price = None, None

        if spy_price is not None and nifty_price is not None:
            # Predict LTP
            predicted_ltp = predict_ltp(ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix)
            st.write(f"Predicted LTP for next day: {predicted_ltp}")

            # Adjust Stop Loss and Maximum LTP based on volatility
            stop_loss, max_ltp = adjust_for_volatility(predicted_ltp, india_vix)
            st.write(f"Dynamic Stop Loss: {stop_loss}")
            st.write(f"Dynamic Maximum LTP: {max_ltp}")

            # Profit/Loss Recommendation
            if predicted_ltp > ltp:
                st.write("Recommendation: Profit")
                st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
            else:
                st.write("Recommendation: Loss")
                st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")
