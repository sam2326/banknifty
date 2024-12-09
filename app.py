import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta

# Streamlit UI setup
st.set_page_config(page_title="Enhanced Multi-Index Options Prediction", layout="wide")

# Title and description
st.title("Enhanced Multi-Index Options Prediction App")
st.markdown("""
    This app predicts the next day's movement for options based on real-time market data, including sentiment analysis and machine learning predictions.
    You can select multiple indices or stocks like **BankNifty**, **Nifty 50**, **Reliance**, and more.
    It also provides **Trading Suggestions** with risk management.
""")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Sidebar for user inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Index/Stock", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)
risk_tolerance = st.sidebar.number_input("Risk Tolerance (%)", min_value=1, max_value=10, value=2, step=1)
target_profit = st.sidebar.number_input("Target Profit (%)", min_value=1, max_value=20, value=5, step=1)

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
        return current_price
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

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
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates" OR "monetary policy" OR "banking sector" OR "GDP growth" OR "inflation" OR "earnings report" OR "trade wars" OR "interest rate hikes" OR "acquisitions" OR "merger" OR "quarterly results"&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
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
        return 0  # Return 0 if there's an error

# Function to calculate sentiment score from headlines
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        try:
            if isinstance(headline, str) and headline.strip():
                sentiment_score += TextBlob(headline).sentiment.polarity
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    
    return sentiment_score / len(news_headlines) if news_headlines else 0

# Function to predict LTP for the selected ticker
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score):
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.05
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Function to calculate trading suggestions
def get_trading_suggestion(predicted_ltp, ltp, risk_tolerance, target_profit):
    stop_loss = ltp * (1 - risk_tolerance / 100)
    target_price = predicted_ltp * (1 + target_profit / 100)
    expected_profit = target_price - ltp
    potential_loss = ltp - stop_loss
    rrr = expected_profit / potential_loss if potential_loss > 0 else 0

    suggestion = "Avoid"  # Default
    if rrr >= 1:
        suggestion = "Buy"
    elif rrr < 1 and potential_loss < expected_profit:
        suggestion = "Hold"

    return round(stop_loss, 2), round(target_price, 2), round(rrr, 2), suggestion

# Main logic for prediction
if st.button("Get Prediction"):
    ticker_price = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"**Current Price for {ticker_name}:** {ticker_price}")

        india_vix_ticker = yf.Ticker("^INDIAVIX")
        try:
            india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            india_vix = 15.0  # Default India VIX value if unavailable
            st.write("Warning: Using default India VIX value.")

        st.write(f"**India VIX:** {india_vix}")

        sp500_price = fetch_sp500_data()
        if sp500_price is None:
            st.warning("Could not fetch S&P 500 data.")
        else:
            st.write(f"**Current S&P 500 Price:** {sp500_price}")

        sentiment_score = get_news_sentiment(ticker_name)
        st.write(f"**Sentiment Score Based on News:** {round(sentiment_score, 2)}")

        predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score)
        st.write(f"**Predicted LTP for Next Day:** {predicted_ltp}")

        stop_loss, target_price, rrr, suggestion = get_trading_suggestion(predicted_ltp, ltp, risk_tolerance, target_profit)

        # Display trading suggestion
        st.markdown("### **Trading Suggestion**")
        st.write(f"**Stop Loss:** {stop_loss}")
        st.write(f"**Target Price:** {target_price}")
        st.write(f"**Risk-to-Reward Ratio (RRR):** {rrr}")
        st.write(f"**Suggestion:** {suggestion}")
