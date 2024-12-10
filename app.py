import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
        data = ticker_obj.history(period="1y", interval="1d")  # Fetch 1 year of data
        return data
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
    
    return round(sentiment_score / len(news_headlines), 2) if news_headlines else 0

# Train a machine learning model (RandomForestRegressor)
def train_ml_model(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday
    
    # We can add more features like moving averages, RSI, etc.
    data['Moving_Avg'] = data['Close'].rolling(window=5).mean()
    data['Volatility'] = data['Close'].pct_change()
    
    # Define X and y
    X = data[['Year', 'Month', 'Day', 'Weekday', 'Moving_Avg', 'Volatility']].dropna()
    y = data['Close'].shift(-1).dropna()  # Target: Next day's closing price
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test and evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    st.write(f"Model Evaluation (RMSE): {rmse}")
    
    return model

# Predict the LTP using the trained model
def predict_ltp_ml(model, ticker_data):
    features = ticker_data[['Year', 'Month', 'Day', 'Weekday', 'Moving_Avg', 'Volatility']].dropna().iloc[-1]
    predicted_ltp = model.predict([features])[0]
    return round(predicted_ltp, 2)

# Main logic for prediction with machine learning
def predict():
    ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_data is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {ticker_data['Close'].iloc[-1]}")
        
        model = train_ml_model(ticker_data)  # Train the model with historical data
        predicted_ltp = predict_ltp_ml(model, ticker_data)  # Predict next day's LTP
        
        st.write(f"Predicted LTP for next day (ML Model): {predicted_ltp}")
        
        # Existing features for Stop Loss, Target Price, RRR, etc.
        stop_loss = round(predicted_ltp * (1 - (risk_percent / 100)), 2)
        max_ltp = round(predicted_ltp * (1 + (profit_percent / 100)), 2)

        st.write(f"Stop Loss: {stop_loss}")
        st.write(f"Target Price: {max_ltp}")

        rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and max_ltp else None
        st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

        if rrr and rrr > 1:
            st.write("Suggestion: Buy")
        else:
            st.write("Suggestion: Avoid")

# Add a button to trigger prediction manually
if st.button("Get Prediction"):
    predict()
