import streamlit as st
import requests
from datetime import datetime, timedelta

# Constants
BASE_URL = "https://api.icicidirect.com/breezeapi/api/v1/"
API_KEY = "%659227P~g54$16430J5W&2I449991ab"
SECRET_KEY = "607Yr~3k0308933gk54iyW5962m66+67"
AUTH_CODE = "49518548"

# Function to authenticate and get access token
@st.cache_data
def get_access_token():
    url = f"{BASE_URL}login"
    payload = {"api_key": API_KEY, "secret_key": SECRET_KEY, "auth_code": AUTH_CODE}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["access_token"]

# Fetch available strikes and expiry dates
def fetch_options_data(token):
    url = f"{BASE_URL}optionchain"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"symbol": "BANKNIFTY"}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

# Fetch real-time LTP and Open Interest
def fetch_real_time_data(token, symbol, expiry, option_type, strike):
    url = f"{BASE_URL}optiondata"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "symbol": symbol,
        "expiry_date": expiry,
        "option_type": option_type,
        "strike_price": strike
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

# Placeholder for predictions
def predict_price(current_ltp, open_interest):
    # Example prediction logic: Adjust this based on your ML model or algorithm
    predicted_ltp = current_ltp * (1 + 0.02)  # Assume a 2% increase for now
    return predicted_ltp

# App starts here
st.title("Bank Nifty Prediction App")

# Authenticate and get access token
access_token = get_access_token()

# Fetch options data for strike prices and expiry dates
options_data = fetch_options_data(access_token)
strike_prices = sorted(set([data["strike_price"] for data in options_data]))
expiry_dates = sorted(set([data["expiry_date"] for data in options_data]))

# Input fields
option_type = st.selectbox("Option Type", ["CE (Call)", "PE (Put)"])
expiry_date = st.selectbox("Expiry Date", expiry_dates)
strike_price = st.selectbox("Strike Price", strike_prices)

# Display real-time LTP and Open Interest
if st.button("Fetch Real-Time Data"):
    symbol = "BANKNIFTY"
    real_time_data = fetch_real_time_data(
        access_token, symbol, expiry_date, option_type, strike_price
    )
    current_ltp = real_time_data["ltp"]
    open_interest = real_time_data["open_interest"]

    st.write(f"**Current LTP:** ₹{current_ltp:.2f}")
    st.write(f"**Open Interest:** {open_interest:,}")

# Predict and display output
if st.button("Predict"):
    # Ensure we have LTP and Open Interest from the real-time data fetch
    if "current_ltp" not in locals():
        st.error("Please fetch real-time data first!")
    else:
        predicted_ltp = predict_price(current_ltp, open_interest)
        profit_or_loss = predicted_ltp - current_ltp

        st.write(f"**Predicted LTP:** ₹{predicted_ltp:.2f}")
        st.write(f"**Profit/Loss:** ₹{profit_or_loss:.2f}")

        recommendation = "Buy" if profit_or_loss > 0 else "Do Not Buy"
        st.write(f"**Recommendation:** {recommendation}")
