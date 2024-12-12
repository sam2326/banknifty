import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
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
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53500)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=516.0, step=0.05)
risk_percent = st.sidebar.number_input("Enter Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
profit_percent = st.sidebar.number_input("Enter Profit Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

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

# Function to determine market trend with stronger downtrend penalty
def determine_market_trend(ticker_symbol):
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo", interval="1d")
        data['5_day_MA'] = data['Close'].rolling(window=5).mean()
        data['20_day_MA'] = data['Close'].rolling(window=20).mean()
        if len(data) >= 20:
            current_5_day_MA = data['5_day_MA'].iloc[-1]
            current_20_day_MA = data['20_day_MA'].iloc[-1]
            if current_5_day_MA > current_20_day_MA:
                return "up"
            elif current_5_day_MA < current_20_day_MA:
                return "down"
        return "neutral"
    except Exception as e:
        st.warning(f"Error determining market trend: {e}")
        return "neutral"

# Function to fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Function to get news sentiment for a given index/stock
def get_news_sentiment(ticker_name):
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Replace with your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={ticker_name}&apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'articles' in data and data['articles']:
            articles = data['articles']
            headlines = [article['title'] for article in articles if article['title']]
            sentiment_score = get_sentiment_score(headlines)
            return sentiment_score
        else:
            st.write("Warning: No articles found.")
            return 0
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return 0

# Function to calculate sentiment score from headlines
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        try:
            if isinstance(headline, str) and headline.strip():
                sentiment_score += TextBlob(headline).sentiment.polarity
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    return round(sentiment_score / len(news_headlines), 2) if news_headlines else 0

# Enhanced prediction model
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    trend_factor = -0.05 if market_trend == "down" else 0.05 if market_trend == "up" else 0
    sentiment_factor = calculate_sentiment_adjustment(sentiment_score, market_trend)
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + trend_factor + sentiment_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Display sentiment with timestamp
def display_sentiment_with_time():
    sentiment_score = get_news_sentiment(ticker_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {timestamp})")
    return sentiment_score

# Risk-to-Reward adjustments
def calculate_risk_reward(predicted_ltp, risk_percent, profit_percent, market_trend):
    risk_adjustment = 0.02 if market_trend == "down" else -0.02 if market_trend == "up" else 0
    stop_loss = round(predicted_ltp * (1 - (risk_percent / 100) - risk_adjustment), 2)
    target_price = round(predicted_ltp * (1 + (profit_percent / 100) + risk_adjustment), 2)
    rrr = round((target_price - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and target_price else None
    return stop_loss, target_price, rrr

# Main prediction logic
def predict():
    ticker_price, _ = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return
    st.write(f"Current price for {ticker_name}: {ticker_price}")

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

    market_trend = determine_market_trend(ticker_symbol)
    st.write(f"Market Trend: {market_trend}")

    sentiment_score = display_sentiment_with_time()

    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    stop_loss, target_price, rrr = calculate_risk_reward(predicted_ltp, risk_percent, profit_percent, market_trend)
    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {target_price}")
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    if rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

if st.button("Get Prediction"):
    predict()
