import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from textblob import TextBlob

# Function to fetch BankNifty historical data from Yahoo Finance
def get_banknifty_data():
    banknifty = yf.download("^NSEBANK", start="2015-01-01", end="2023-12-31")
    banknifty['Returns'] = banknifty['Adj Close'].pct_change()
    return banknifty

# Function to fetch S&P 500 global market data from Yahoo Finance
def get_global_market_data():
    sp500 = yf.download("^GSPC", start="2015-01-01", end="2023-12-31")
    sp500['Returns'] = sp500['Adj Close'].pct_change()
    return sp500

# Sentiment Analysis using TextBlob
def get_sentiment(news_headline):
    analysis = TextBlob(news_headline)
    return analysis.sentiment.polarity

# Preprocess BankNifty data and create features for model training
def preprocess_data(banknifty_data):
    banknifty_data['SMA_10'] = banknifty_data['Adj Close'].rolling(window=10).mean()
    banknifty_data['SMA_50'] = banknifty_data['Adj Close'].rolling(window=50).mean()
    banknifty_data['Volatility'] = banknifty_data['Returns'].rolling(window=10).std()
    banknifty_data['Sentiment'] = np.random.choice([1, -1], size=len(banknifty_data))  # Placeholder sentiment
    
    banknifty_data.dropna(inplace=True)
    
    X = banknifty_data[['SMA_10', 'SMA_50', 'Volatility', 'Sentiment']]
    y = (banknifty_data['Returns'].shift(-1) > 0).astype(int)  # 1 if price goes up next day, else 0
    
    return X, y

# Function to train a Random Forest model for prediction
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Backtesting the model to simulate performance
def backtest(model, X, y):
    y_pred = model.predict(X)
    correct_predictions = np.sum(y_pred == y)
    accuracy = correct_predictions / len(y)
    
    print(f"Backtesting Accuracy: {accuracy * 100:.2f}%")

# Streamlit App to display results
def main():
    st.title("BankNifty Prediction Based on Global Market Conditions")
    
    st.markdown("""
    This app predicts the next day's movement of BankNifty based on global market conditions and sentiment.
    The prediction is based on a machine learning model trained with historical BankNifty and global market data.
    """)
    
    if st.button("Fetch and Predict for Tomorrow"):
        # Step 1: Fetch BankNifty and global market data
        banknifty_data = get_banknifty_data()
        sp500_data = get_global_market_data()
        
        # Step 2: Preprocess data and train model
        X, y = preprocess_data(banknifty_data)
        model = train_model(X, y)
        
        # Step 3: Simulate sentiment (For now, using random sentiment)
        sentiment = np.random.choice([1, -1])  # Replace with real sentiment analysis
        
        # Step 4: Predict for the next day
        latest_data = X.tail(1)
        prediction = model.predict(latest_data)
        prediction_label = "BUY" if prediction == 1 else "SELL"
        
        # Step 5: Display the prediction and sentiment
        st.write(f"Prediction for tomorrow: {prediction_label}")
        st.write(f"Simulated Sentiment Score: {sentiment}")
        
        # Backtest the model (optional)
        backtest(model, X, y)

if __name__ == "__main__":
    main()
