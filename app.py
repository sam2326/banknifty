import yfinance as yf
import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import requests
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks import analytical

# Streamlit UI setup
st.title("BankNifty Options Prediction for Intraday Trading with Dynamic Strike Recommendations")
st.write("""
    This app predicts the next day's movement for BankNifty based on real-time market data.
    Enter the details for the BankNifty option you're interested in, and get predictions for the next day.
    You can manually enter the **LTP** for the selected strike price.
    The app also suggests **dynamic strike recommendations** based on market conditions.
""")

# Input fields for the user
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Function to get BankNifty current data using Yahoo Finance
def get_banknifty_data():
    try:
        banknifty = yf.Ticker("^NSEBANK")
        banknifty_data = banknifty.history(period="1d", interval="1m")
        current_price = banknifty_data["Close"].iloc[-1]
        return current_price
    except Exception as e:
        st.write(f"Error fetching BankNifty data: {e}")
        return None

# Function to get India VIX and global market data (S&P500, Nifty 50)
def get_market_data():
    try:
        spy = yf.Ticker("^GSPC")
        nifty = yf.Ticker("^NSEI")
        vix = yf.Ticker("^INDIAVIX")

        spy_price = spy.history(period="1d", interval="1m")["Close"].iloc[-1]
        nifty_price = nifty.history(period="1d", interval="1m")["Close"].iloc[-1]
        india_vix = vix.history(period="1d", interval="1m")["Close"].iloc[-1]

        st.write(f"S&P 500 Price: {spy_price}")
        st.write(f"Nifty 50 Price: {nifty_price}")
        st.write(f"India VIX: {india_vix}")

        return spy_price, nifty_price, india_vix
    except Exception as e:
        st.write(f"Error fetching market data: {e}")
        return None, None, None

# Function to calculate the expiry date (last Thursday of the given month)
def calculate_expiry_date(year, month):
    next_month = month + 1 if month < 12 else 1
    next_month_year = year if month < 12 else year + 1
    first_day_next_month = datetime(next_month_year, next_month, 1)

    days_to_subtract = (first_day_next_month.weekday() + 4) % 7
    expiry_date = first_day_next_month - timedelta(days=days_to_subtract)
    return expiry_date

# Function to fetch option chain data using NSEpy (simplified for now)
def fetch_option_chain(expiry_date, banknifty_price):
    try:
        year = expiry_date.year
        month = expiry_date.month
        expiry = calculate_expiry_date(year, month)

        strikes = [banknifty_price - 200, banknifty_price - 100, banknifty_price, banknifty_price + 100, banknifty_price + 200]

        option_chain = []
        for strike in strikes:
            option_chain.append((strike, strike + random.uniform(0, 50), strike - random.uniform(0, 50)))
        return option_chain
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None

# Function to calculate predicted LTP
def predict_ltp(current_ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix):
    global_sentiment_factor = (spy_price * 0.0015) + (nifty_price * 0.005) + (india_vix * 0.1)

    strike_impact_factor = (strike_price - banknifty_price) * (0.01 if strike_price < banknifty_price else -0.01)

    random_factor = random.uniform(-0.01, 0.02)

    predicted_ltp = current_ltp + global_sentiment_factor + strike_impact_factor + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Sentiment Analysis for real-time news (Placeholder news)
def get_sentiment_analysis(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        sentiment_score += TextBlob(headline).sentiment.polarity
    return sentiment_score / len(news_headlines) if news_headlines else 0

# Function to fetch real-time news and calculate sentiment
def fetch_news_and_sentiment():
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q=BankNifty&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        if 'articles' in data:
            articles = data['articles']
            headlines = [article['title'] for article in articles]
            sentiment_score = get_sentiment_analysis(headlines)
            return sentiment_score
        else:
            st.write("Error: No articles found in the response.")
            return 0
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return 0  # Return 0 in case of error

# Volatility-based adjustments for stop loss and max LTP
def adjust_for_volatility(predicted_ltp, india_vix):
    volatility_factor = india_vix / 100  # Normalize VIX to a usable factor

    stop_loss = predicted_ltp * (1 - (0.01 * volatility_factor))
    max_ltp = predicted_ltp * (1 + (0.02 * volatility_factor))

    return stop_loss, max_ltp

# Adding technical indicators (RSI, MACD, SMA)
def add_technical_indicators(data):
    data['RSI'] = calculate_rsi(data['BankNifty_Close'])
    data['SMA_50'] = calculate_moving_averages(data['BankNifty_Close'], 50)
    data['SMA_200'] = calculate_moving_averages(data['BankNifty_Close'], 200)
    data['MACD'], data['MACD_signal'] = calculate_macd(data['BankNifty_Close'])
    return data

# Main logic to get predictions
if st.button("Get Prediction"):
    banknifty_price = get_banknifty_data()
    if banknifty_price is None:
        st.warning("Could not fetch real-time BankNifty data.")
    else:
        st.write(f"Current BankNifty index price: {banknifty_price}")

        spy_price, nifty_price, india_vix = get_market_data()
        if spy_price is None or nifty_price is None or india_vix is None:
            st.warning("Could not fetch global market data.")
        else:
            st.write(f"Real-time S&P 500 price: {spy_price}")
            st.write(f"Real-time Nifty 50 price: {nifty_price}")
            st.write(f"Real-time India VIX: {india_vix}")

            st.write(f"Manually input LTP: {ltp}")

            option_chain = fetch_option_chain(expiry_date, banknifty_price)
            if option_chain is None:
                st.warning("Could not fetch option chain.")
            else:
                predictions = predict_ltp(ltp, spy_price, nifty_price, strike_price, banknifty_price, india_vix)
                st.write(f"Predicted LTP for next day: {predictions}")

                stop_loss, max_ltp = adjust_for_volatility(predictions, india_vix)
                st.write(f"Stop Loss: {round(stop_loss, 2)}")
                st.write(f"Maximum LTP: {round(max_ltp, 2)}")

                sentiment_score = fetch_news_and_sentiment()
                st.write(f"Sentiment Score: {sentiment_score}")

                if predictions > ltp:
                    st.write("Recommendation: Profit")
                    st.write(f"Expected Profit: {round(predictions - ltp, 2)}")
                else:
                    st.write("Recommendation: Loss")
                    st.write(f"Expected Loss: {round(ltp - predictions, 2)}")
