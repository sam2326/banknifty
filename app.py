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
ticker_name = st.sidebar.selectbox("Select Index/Stock", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to get financial sentiment using FinBERT
def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

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

# Fetching additional data functions...
# (same as your current functions)

# Function to plot the stock data
def plot_stock_data(data, ticker_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker_name} Price'))
    fig.update_layout(title=f'{ticker_name} Stock Data', xaxis_title='Time', yaxis_title='Price', template="plotly_dark")
    st.plotly_chart(fig)

# Function to plot sentiment trend
def plot_sentiment_trend(sentiment_scores, ticker_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sentiment_scores, label=f'Sentiment for {ticker_name}', color='orange')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('News Sentiment Trend')
    ax.legend(loc='best')
    st.pyplot(fig)

# Auto-refresh Section with visual indicators
def auto_refresh():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {ticker_price}")
        # Plotting stock data
        plot_stock_data(ticker_data, ticker_name)

    india_vix_ticker = yf.Ticker("^INDIAVIX")
    try:
        india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0
        st.write("Warning: Using default India VIX value.")
    st.write(f"India VIX: {india_vix}")

    sp500_price = fetch_sp500_data()
    if sp500_price is None:
        st.warning("Could not fetch S&P 500 data.")
    else:
        st.write(f"Current S&P 500 price: {sp500_price}")

    sentiment_score = get_news_sentiment(ticker_name)
    st.write(f"Sentiment Score based on news: {sentiment_score}")

    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    stop_loss = predicted_ltp * 0.98
    max_ltp = predicted_ltp * 1.02
    st.write(f"Stop Loss: {round(stop_loss, 2)}")
    st.write(f"Maximum LTP: {round(max_ltp, 2)}")

    if predicted_ltp > ltp:
        st.write("Recommendation: Profit")
        st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
    else:
        st.write("Recommendation: Loss")
        st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")

# Add a button to start the auto-refresh
if st.button("Start Auto-Refresh"):
    with st.spinner("Fetching data..."):
        auto_refresh()
    st.success("Data fetched successfully!")
