import yfinance as yf
import streamlit as st
from datetime import datetime, date
from nsepy import get_history

# Streamlit UI setup
st.title("BankNifty Options App with Option Chain Viewer")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    It also includes a live Option Chain Viewer to analyze multiple strike prices, Open Interest, and LTP.
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
        banknifty_data = banknifty.history(period="1d", interval="1m")
        current_price = banknifty_data["Close"].iloc[-1]
        high_price = banknifty_data["High"].max()
        low_price = banknifty_data["Low"].min()
        close_price = banknifty_data["Close"].iloc[-1]
        return current_price, high_price, low_price, close_price
    except Exception as e:
        st.write(f"Error fetching BankNifty data: {e}")
        return None, None, None, None

# Corrected fetch_option_chain function
def fetch_option_chain(expiry_date):
    try:
        # Fetch Call Options (CE) data
        calls = get_history(
            symbol="BANKNIFTY",
            index=True,
            start=date.today(),
            end=date.today(),
            option_type="CE",  # Call options
            strike_price=None,  # Fetch all strikes
            expiry_date=expiry_date
        )
        
        # Fetch Put Options (PE) data
        puts = get_history(
            symbol="BANKNIFTY",
            index=True,
            start=date.today(),
            end=date.today(),
            option_type="PE",  # Put options
            strike_price=None,  # Fetch all strikes
            expiry_date=expiry_date
        )

        return calls, puts
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None, None

# Display option chain in the app
def display_option_chain(calls, puts):
    st.subheader("BankNifty Option Chain")

    # Display Call Options
    st.write("**Call Options**")
    st.dataframe(calls[['Strike Price', 'Last Traded Price', 'Open Interest']])

    # Display Put Options
    st.write("**Put Options**")
    st.dataframe(puts[['Strike Price', 'Last Traded Price', 'Open Interest']])

# Main logic
if st.button("Fetch Option Chain"):
    calls, puts = fetch_option_chain(expiry_date)
    if calls is None or puts is None:
        st.warning("Could not fetch option chain. Please try again later.")
    else:
        display_option_chain(calls, puts)
