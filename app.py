import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Streamlit UI setup
st.set_page_config(page_title="Enhanced Multi-Index Options Prediction", layout="wide")

# Title and description
st.title("Enhanced Multi-Index Options Prediction App")
st.markdown("""
    This app predicts the next day's movement for options based on real-time market data, including sentiment analysis and machine learning predictions.
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

# Sidebar for Inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price, data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to plot Matplotlib chart
def plot_matplotlib_chart(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Closing Price', color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f'Closing Price for {ticker_name}')
    ax.legend(loc='best')
    st.pyplot(fig)

# Function to plot Plotly candlestick chart
def plot_plotly_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick Chart"
    )])

    fig.update_layout(
        title=f'Candlestick Chart for {ticker_name}',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig)

# Main logic for prediction with auto-refresh
def auto_refresh():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {ticker_price}")

    # Plotting the Matplotlib chart (Historical Price)
    plot_matplotlib_chart(ticker_data)
    
    # Plotting the Plotly Candlestick Chart
    plot_plotly_candlestick_chart(ticker_data)

    # Add the rest of the logic for predictions and sentiment analysis here...
    # (Predict LTP, Stop Loss, Max LTP, etc. as done earlier)

# Add a button to start the auto-refresh
if st.button("Start Auto-Refresh"):
    with st.spinner("Fetching data..."):
        auto_refresh()
    st.success("Data fetched successfully!")
