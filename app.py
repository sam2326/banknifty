import yfinance as yf
import streamlit as st
from datetime import datetime
import random
from nsepy import get_history
from nsepy.derivatives import get_expiry_date
import pandas as pd

# Streamlit UI setup
st.title("BankNifty Options Prediction for Intraday Trading with Dynamic Strike Recommendations")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    Enter the details for the BankNifty option you're interested in, and get predictions for the next day.
    You can manually enter the **LTP** for the selected strike price.
    The app also suggests **dynamic strike recommendations** based on market conditions.
""")

# Input fields for the user
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to get BankNifty current data using Yahoo Finance
def get_banknifty_data():
    try:
        banknifty = yf.Ticker("^NSEBANK")
        banknifty_data = banknifty.history(period="1d", interval="1m")
        current_price = banknifty_data["Close"].iloc[-1]
        return current_price
    except Exception as e:
        st.write(f"Error fetching BankNifty data: {e}")
        return None

# Function to get India VIX and global market data (S&P500, Nifty 50)
def get_market_data():
    try:
        spy = yf.Ticker("^GSPC")
        nifty = yf.Ticker("^NSEI")
        vix = yf.Ticker("^INDIAVIX")

        spy_price = spy.history(period="1d", interval="1m")["Close"].iloc[-1]
        nifty_price = nifty.history(period="1d", interval="1m")["Close"].iloc[-1]
        india_vix = vix.history(period="1d", interval="1m")["Close"].iloc[-1]

        st.write(f"S&P 500 Price: {spy_price}")
        st.write(f"Nifty 50 Price: {nifty_price}")
        st.write(f"India VIX: {india_vix}")

        return spy_price, nifty_price, india_vix
    except Exception as e:
        st.write(f"Error fetching market data: {e}")
        return None, None, None

# Function to fetch option chain data using NSEpy
def fetch_option_chain(expiry_date, banknifty_price):
    try:
        # Get the expiry date for BankNifty options manually based on the month and year
        year = expiry_date.year
        month = expiry_date.month
        
        # Use NSEpy to get the expiry date
        expiry = get_expiry_date(symbol="BANKNIFTY", year=year, month=month)
        
        # Handle the case if the expiry date is not found
        if not expiry:
            st.write(f"No expiry date found for BankNifty options in {month}/{year}")
            return None

        # Define a range of strikes around the current BankNifty price
        strikes = [banknifty_price - 200, banknifty_price - 100, banknifty_price, banknifty_price + 100, banknifty_price + 200]

        # Fetch option chain for Calls and Puts from NSEpy
        option_chain = []
        for strike in strikes:
            call_data = get_history(symbol="BANKNIFTY", index=True, start=expiry_date, end=expiry_date, option_type="CE", strike_price=strike, expiry_date=expiry)
            put_data = get_history(symbol="BANKNIFTY", index=True, start=expiry_date, end=expiry_date, option_type="PE", strike_price=strike, expiry_date=expiry)

            option_chain.append((call_data, put_data))

        return option_chain
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None

# Function to calculate predicted LTP
def predict_ltp(current_ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix):
    global_sentiment_factor = (spy_price * 0.0015) + (nifty_price * 0.005) + (india_vix * 0.1)

    if strike_price > banknifty_price:
        strike_impact_factor = (strike_price - banknifty_price) * -0.01
    else:
        strike_impact_factor = (banknifty_price - strike_price) * 0.01

    random_factor = random.uniform(-0.01, 0.02)

    predicted_ltp = current_ltp + global_sentiment_factor + strike_impact_factor + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Function to recommend strikes based on proximity to BankNifty
def recommend_strikes(option_chain, banknifty_price):
    recommendations = []

    # Calculate the proximity of each strike to the current BankNifty price
    for call_data, put_data in option_chain:
        if not call_data.empty and not put_data.empty:
            call_strike = call_data['Strike Price'].iloc[0]
            put_strike = put_data['Strike Price'].iloc[0]

            # Calculate proximity to BankNifty
            proximity_call = abs(call_strike - banknifty_price)
            proximity_put = abs(put_strike - banknifty_price)

            recommendations.append({
                'strike': call_strike,
                'proximity_call': proximity_call,
                'proximity_put': proximity_put
            })

    # Sort the recommendations by proximity to BankNifty price
    recommendations.sort(key=lambda x: min(x['proximity_call'], x['proximity_put']))

    return recommendations

# Main logic
if st.button("Get Prediction"):
    banknifty_price = get_banknifty_data()
    if banknifty_price is None:
        st.warning("Could not fetch real-time BankNifty data. Please try again later.")
    else:
        st.write(f"Current BankNifty index price: {banknifty_price}")

        spy_price, nifty_price, india_vix = get_market_data()
        if spy_price is None or nifty_price is None or india_vix is None:
            st.warning("Could not fetch global market data. Please try again later.")
        else:
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time Nifty 50 price: {nifty_price}")
            st.write(f"Real-time India VIX: {india_vix}")

            st.write(f"Manually input LTP: {ltp}")

            # Fetch option chain for dynamic strikes
            option_chain = fetch_option_chain(expiry_date, banknifty_price)
            if option_chain is None:
                st.warning("Could not fetch option chain. Please try again later.")
            else:
                recommendations = recommend_strikes(option_chain, banknifty_price)
                st.write("**Recommended Strikes Based on Proximity to BankNifty**")
                for rec in recommendations[:5]:  # Show top 5 recommended strikes
                    st.write(f"Strike: {rec['strike']}, Proximity to BankNifty: {min(rec['proximity_call'], rec['proximity_put'])}")

                predicted_ltp = predict_ltp(ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix)
                st.write(f"Predicted LTP for next day: {predicted_ltp}")

                stop_loss, max_ltp = adjust_for_volatility(predicted_ltp, india_vix)
                st.write(f"Stop Loss: {round(stop_loss, 2)}")
                st.write(f"Maximum LTP: {round(max_ltp, 2)}")

                if predicted_ltp > ltp:
                    st.write("Recommendation: Profit")
                    st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
                else:
                    st.write("Recommendation: Loss")
                    st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")
