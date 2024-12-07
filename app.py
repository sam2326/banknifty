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

# Fetch option chain using NSEpy for a range of strikes
def fetch_option_chain(expiry_date, banknifty_price):
    try:
        # Define a range of strikes around the current BankNifty price
        strikes = [banknifty_price - 200, banknifty_price - 100, banknifty_price, banknifty_price + 100, banknifty_price + 200]

        calls = []
        puts = []

        # Fetch data for each strike price
        for strike in strikes:
            call_data = get_history(
                symbol="BANKNIFTY",
                index=True,
                start=date.today(),
                end=date.today(),
                option_type="CE",  # Call options
                strike_price=strike,
                expiry_date=expiry_date
            )
            put_data = get_history(
                symbol="BANKNIFTY",
                index=True,
                start=date.today(),
                end=date.today(),
                option_type="PE",  # Put options
                strike_price=strike,
                expiry_date=expiry_date
            )
            
            # Debugging: Print the fetched data
            st.write(f"Fetched data for Strike {strike} (Call):")
            st.write(call_data.head())  # Print first few rows

            st.write(f"Fetched data for Strike {strike} (Put):")
            st.write(put_data.head())  # Print first few rows

            calls.append(call_data)
            puts.append(put_data)

        return calls, puts
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None, None

# Display option chain in the app
def display_option_chain(calls, puts):
    st.subheader("BankNifty Option Chain")

    # Display Call Options
    st.write("**Call Options**")
    for call in calls:
        st.write(f"Data for Strike {call['Strike Price'].iloc[0]} (Call):")
        st.write(call.head())  # Display the first few rows for debugging

        if not call.empty:
            st.dataframe(call[['Strike Price', 'Last', 'Open Interest']])
        else:
            st.write(f"No data available for Strike {call['Strike Price'].iloc[0]} (Call).")

    # Display Put Options
    st.write("**Put Options**")
    for put in puts:
        st.write(f"Data for Strike {put['Strike Price'].iloc[0]} (Put):")
        st.write(put.head())  # Display the first few rows for debugging

        if not put.empty:
            st.dataframe(put[['Strike Price', 'Last', 'Open Interest']])
        else:
            st.write(f"No data available for Strike {put['Strike Price'].iloc[0]} (Put).")

# Main logic
if st.button("Fetch Option Chain"):
    banknifty_price, high_price, low_price, close_price = get_banknifty_data()
    
    if banknifty_price is None:
        st.warning("Could not fetch real-time BankNifty data. Please try again later.")
    else:
        st.write(f"Current BankNifty index price: {banknifty_price}")
        st.write(f"High Price: {high_price}, Low Price: {low_price}, Close Price: {close_price}")

        # Fetch option chain for dynamic strikes
        calls, puts = fetch_option_chain(expiry_date, banknifty_price)
        if calls is None or puts is None:
            st.warning("Could not fetch option chain. Please try again later.")
        else:
            display_option_chain(calls, puts)
