import yfinance as yf
import streamlit as st
import random
from textblob import TextBlob
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit UI setup
st.title("Enhanced Multi-Index Options Prediction App with Machine Learning")
st.write("""
    This app predicts the next day's movement for options based on real-time market data, including sentiment analysis and machine learning predictions.
""")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Dropdown for selecting ticker
ticker_name = st.selectbox("Select Index/Stock", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]

# Input fields
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="30d", interval="1d")  # Fetching last 30 days of data
        if data.empty:
            st.write(f"Error: No data returned for {ticker}.")
            return None
        return data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

# Function to fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="30d", interval="1d")
        return data["Close"]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Function to fetch news sentiment for a given index/stock
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

# Function to add technical indicators manually
def add_technical_indicators(data):
    # Moving Average (50 periods)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Moving Average (200 periods)
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD: Moving Average Convergence Divergence
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data.dropna()  # Remove any rows with NaN values

# Function to train the Random Forest model
def train_ml_model(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Trained. Test MSE: {mse:.2f}")

    return model

# Function to predict the next day's LTP using the trained model
def predict_ltp_using_ml(model, data):
    features = data.drop(columns=['LTP'])  # Drop the target column
    predicted_ltp = model.predict(features.tail(1))  # Predict for the last row
    return round(predicted_ltp[0], 2)

# Main logic for prediction
if st.button("Get Prediction"):
    data = fetch_ticker_data(ticker_symbol)
    if data is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {data['Close'].iloc[-1]}")

        # Fetch India VIX
        india_vix_ticker = yf.Ticker("^INDIAVIX")
        try:
            india_vix = india_vix_ticker.history(period="30d", interval="1d")["Close"].iloc[-1]
        except:
            india_vix = 15.0  # Default VIX value if fetching fails
            st.write("Warning: Using default India VIX value.")

        st.write(f"India VIX: {india_vix}")

        # Fetch S&P 500 data
        sp500_data = fetch_sp500_data()
        if sp500_data is None:
            st.warning("Could not fetch S&P 500 data.")
        else:
            st.write(f"Current S&P 500 price: {sp500_data.iloc[-1]}")

        # Fetch news sentiment
        sentiment_score = get_news_sentiment(ticker_name)
        st.write(f"Sentiment Score based on news: {sentiment_score}")

        # Add technical indicators to the data
        data_with_indicators = add_technical_indicators(data)

        # Train the ML model (this could be done offline and saved for real-time use)
        model = train_ml_model(data_with_indicators, target_column='LTP')

        # Predict LTP using the ML model
        predicted_ltp = predict_ltp_using_ml(model, data_with_indicators)
        st.write(f"Predicted LTP for next day (ML-based): {predicted_ltp}")

        # Stop Loss and Max LTP
        stop_loss = predicted_ltp * 0.98
        max_ltp = predicted_ltp * 1.02
        st.write(f"Stop Loss: {round(stop_loss, 2)}")
        st.write(f"Maximum LTP: {round(max_ltp, 2)}")

        # Recommendation
        if predicted_ltp > ltp:
            st.write("Recommendation: Profit")
            st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
        else:
            st.write("Recommendation: Loss")
            st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")
