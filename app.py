import yfinance as yf
import streamlit as st
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta

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
uploaded_file = st.sidebar.file_uploader("Upload NSE Option Chain CSV", type="csv")

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
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates" OR "monetary policy"&apiKey={api_key}'

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

# Function to determine market trend using moving averages
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

# Function to display sentiment score with timestamp
def display_sentiment_with_time():
    sentiment_score = get_news_sentiment(ticker_name)  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {timestamp})")
    return sentiment_score

# Function to load and process the uploaded option chain CSV file
def load_and_clean_csv(file):
    try:
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]
        df = df[["STRIKE \n", "OPTION TYPE \n", "LTP \n"]]
        df.rename(columns={"STRIKE \n": "Strike Price", "LTP \n": "LTP", "OPTION TYPE \n": "Option Type"}, inplace=True)
        call_df = df[df["Option Type"] == "Call"][["Strike Price", "LTP"]].rename(columns={"LTP": "Call LTP"})
        put_df = df[df["Option Type"] == "Put"][["Strike Price", "LTP"]].rename(columns={"LTP": "Put LTP"})
        option_chain = pd.merge(call_df, put_df, on="Strike Price", how="outer").fillna(0)
        option_chain["Strike Price"] = option_chain["Strike Price"].astype(float)
        option_chain["Call LTP"] = option_chain["Call LTP"].astype(float)
        option_chain["Put LTP"] = option_chain["Put LTP"].astype(float)
        return option_chain
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Enhanced prediction function for option chain
def predict_option_chain(df, ticker_price, india_vix, sentiment_score, market_trend):
    predictions = []
    for _, row in df.iterrows():
        predicted_call = predict_ltp(row["Call LTP"], ticker_price, row["Strike Price"], india_vix, 0, market_trend)
        predicted_put = predict_ltp(row["Put LTP"], ticker_price, row["Strike Price"], india_vix, 0, market_trend)
        predictions.append({"Strike Price": row["Strike Price"], "Predicted Call LTP": predicted_call, "Predicted Put LTP": predicted_put})
    return pd.DataFrame(predictions)

if uploaded_file:
    option_chain = load_and_clean_csv(uploaded_file)
    if option_chain is not None:
        ticker_price = 51233  # Example price
        india_vix = 15  # Default India VIX
        sentiment = display_sentiment_with_time()
        market_trend = determine_market_trend()
        result = predict_option_chain(option_chain, ticker_price, india_vix, sentiment, market_trend)
        st.write("Option Chain Predictions", result)
        st.download_button("Download Predictions", result.to_csv(index=False), mime="text/csv")
