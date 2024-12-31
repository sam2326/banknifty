import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import json
import websocket

# OpenAI API Key
OPENAI_API_KEY = "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"

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions", layout="wide")
st.title("Trading Predictions")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Sidebar Inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date")
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)
risk_percent = st.sidebar.number_input("Enter Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
profit_percent = st.sidebar.number_input("Enter Profit Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Load FinBERT
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function: Get financial sentiment using FinBERT
def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

# Fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price, data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None, None

# Fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Determine market trend using moving averages
def determine_market_trend():
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo")
        data['5_MA'] = data['Close'].rolling(window=5).mean()
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        if data['5_MA'].iloc[-1] > data['20_MA'].iloc[-1]:
            return "up"
        elif data['5_MA'].iloc[-1] < data['20_MA'].iloc[-1]:
            return "down"
        else:
            return "neutral"
    except Exception as e:
        st.write(f"Error determining market trend: {e}")
        return "neutral"

# WebSocket Handler for GPT Analysis
def gpt_via_websocket(prompt):
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = [
        f"Authorization: Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta: realtime=v1"
    ]

    try:
        # Connect to WebSocket
        ws = websocket.create_connection(url, header=headers)

        # Send the prompt
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": prompt
            }
        }
        ws.send(json.dumps(event))

        # Receive the response
        message = ws.recv()
        ws.close()

        # Parse and return the response
        server_event = json.loads(message)
        return server_event.get("response", {}).get("text", "No response available.")
    except Exception as e:
        return f"Error connecting to WebSocket: {e}"

# Main Prediction Function
def predict():
    ticker_price, _ = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    st.write(f"Current price for {ticker_name}: {ticker_price}")

    # Fetch India VIX
    try:
        india_vix = yf.Ticker("^INDIAVIX").history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0
        st.write("Warning: Using default India VIX value.")
    st.write(f"India VIX: {india_vix}")

    # Fetch SP500 data
    sp500_price = fetch_sp500_data()
    if sp500_price is None:
        st.warning("Could not fetch S&P 500 data.")
    else:
        st.write(f"Current S&P 500 price: {sp500_price}")

    # Market Trend
    market_trend = determine_market_trend()
    st.write(f"Market Trend: {market_trend}")

    # Sentiment Analysis
    sentiment_score = gpt_via_websocket(f"Analyze sentiment for {ticker_name}")
    st.write(f"Sentiment Score: {sentiment_score}")

    # Market Insights
    insights = gpt_via_websocket(f"The market trend for {ticker_name} is {market_trend}. Provide insights.")
    st.write("Market Insights:")
    st.write(insights)

    # Risk and Reward Analysis
    risk_analysis = gpt_via_websocket(f"Given LTP {ltp}, risk {risk_percent}%, and profit {profit_percent}%, suggest targets.")
    st.write("Risk and Reward Analysis:")
    st.write(risk_analysis)

# Run Prediction on Button Click
if st.button("Get Prediction"):
    predict()
