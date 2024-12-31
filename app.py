import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import asyncio
import websockets
import json
import openai

# OpenAI API Key
openai.api_key = "your_openai_api_key"

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
async def gpt_via_websocket(prompt):
    uri = "ws://localhost:8000"
    async with websockets.connect(uri) as websocket:
        # Send the prompt
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": prompt
            }
        }
        await websocket.send(json.dumps(event))

        # Wait for the response
        message = await websocket.recv()
        server_event = json.loads(message)
        return server_event.get("content", "No response available.")

# Asynchronous GPT Sentiment Analysis
async def gpt_sentiment_analysis(news_headlines):
    prompt = f"Analyze the sentiment of these financial news headlines: {news_headlines}. Provide a score between -1 (negative) and 1 (positive)."
    return await gpt_via_websocket(prompt)

# Asynchronous GPT Market Insights
async def gpt_market_insights(ticker_name, trend, sentiment_score):
    prompt = f"The market trend for {ticker_name} is {trend}. The sentiment score is {sentiment_score}. Provide an analysis of what this means for traders considering a Call option."
    return await gpt_via_websocket(prompt)

# Asynchronous GPT Risk and Reward Analysis
async def gpt_risk_reward(current_ltp, risk_percent, profit_percent, market_trend):
    prompt = (f"Given the current LTP of {current_ltp}, risk percentage of {risk_percent}, "
              f"and profit percentage of {profit_percent}, suggest an optimal stop loss and target "
              f"price for a {market_trend} market trend.")
    return await gpt_via_websocket(prompt)

# Main Prediction Function
async def predict():
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
    sentiment_score = await gpt_sentiment_analysis(ticker_name)
    st.write(f"Sentiment Score: {sentiment_score}")

    # Market Insights
    insights = await gpt_market_insights(ticker_name, market_trend, sentiment_score)
    st.write("Market Insights:")
    st.write(insights)

    # Risk Reward Analysis
    risk_analysis = await gpt_risk_reward(ltp, risk_percent, profit_percent, market_trend)
    st.write("Risk and Reward Analysis:")
    st.write(risk_analysis)

# Run Prediction on Button Click
if st.button("Get Prediction"):
    asyncio.run(predict())
