import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions", layout="wide")

# Title of the App
st.title("Trading Predictions")

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
risk_percent = st.sidebar.number_input("Enter Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
profit_percent = st.sidebar.number_input("Enter Profit Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

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

# Function to fetch Option Chain data
def fetch_option_chain(ticker):
    url = f'https://www.nseindia.com/api/option-chain-indices?symbol={ticker}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.nseindia.com/',
        'Cache-Control': 'no-cache'
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        # Fetch initial cookies
        session.get("https://www.nseindia.com/")
        
        # Get option chain data
        response = session.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract option chain data
        option_data = data.get('records', {}).get('data', [])

        # Separate calls and puts
        calls = [item['CE'] for item in option_data if 'CE' in item]
        puts = [item['PE'] for item in option_data if 'PE' in item]

        # Create DataFrames for Calls and Puts
        if calls:
            calls_df = pd.DataFrame(calls)
            calls_df = calls_df.loc[:, calls_df.columns.intersection(['strikePrice', 'openInterest', 'changeinOpenInterest', 'lastPrice'])]
        else:
            calls_df = pd.DataFrame(columns=['strikePrice', 'openInterest', 'changeinOpenInterest', 'lastPrice'])

        if puts:
            puts_df = pd.DataFrame(puts)
            puts_df = puts_df.loc[:, puts_df.columns.intersection(['strikePrice', 'openInterest', 'changeinOpenInterest', 'lastPrice'])]
        else:
            puts_df = pd.DataFrame(columns=['strikePrice', 'openInterest', 'changeinOpenInterest', 'lastPrice'])

        return calls_df, puts_df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching option chain data: {e}")
        return None, None

# Option Chain Analysis
def option_chain_analysis():
    st.write("Fetching Option Chain Data for selected strike price...")
    calls_df, puts_df = fetch_option_chain(ticker_symbol)

    if calls_df is not None and puts_df is not None:
        st.write("Calls Option Chain Data:")
        st.dataframe(calls_df)

        st.write("Puts Option Chain Data:")
        st.dataframe(puts_df)
    else:
        st.error("No option chain data available.")

# Add buttons to trigger actions
if st.button("Get Option Chain Data"):
    option_chain_analysis()
