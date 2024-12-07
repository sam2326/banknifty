import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Streamlit UI setup
st.title("Enhanced ML Predictions for BankNifty Options")
st.write("""
    This app uses a machine learning model (Random Forest) to predict the next day's movement for options.
    Enter the details for the option you're interested in, and get ML-powered predictions for the next day.
""")

# Input fields
ticker_name = st.selectbox("Select Ticker", ["BankNifty", "Nifty 50", "Reliance", "HDFC Bank"])
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=790.0, step=0.05)

# Fetch historical data for ML training
def fetch_historical_data(ticker):
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.history(period="1y")  # 1 year of data for training

# Prepare training data
def prepare_training_data(data):
    data['Prev_Close'] = data['Close'].shift(1)
    data['Change'] = data['Close'] - data['Prev_Close']
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    data.dropna(inplace=True)
    return data[['Prev_Close', 'Change', 'Volatility']], data['Close']

# Train Random Forest Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Trained. Test MSE: {mse:.2f}")
    return model

# Save model to file
def save_model(model, filename="rf_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Load pre-trained model
def load_model(filename="rf_model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Main prediction function
def predict_next_day_price(model, ltp, strike_price):
    # Simulate features for prediction
    volatility = 0.02  # Assume a 2% daily volatility
    change = strike_price - ltp  # Strike price impact
    features = np.array([[ltp, change, volatility]])
    return model.predict(features)[0]

# Main logic
if st.button("Get Prediction"):
    # Fetch data
    ticker_map = {
        "BankNifty": "^NSEBANK",
        "Nifty 50": "^NSEI",
        "Reliance": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS"
    }
    ticker = ticker_map[ticker_name]
    data = fetch_historical_data(ticker)
    
    if data is None or data.empty:
        st.warning("Could not fetch historical data. Please try again.")
    else:
        st.write(f"Fetched data for {ticker_name}. Training ML model...")

        # Prepare training data and train model
        X, y = prepare_training_data(data)
        model = train_model(X, y)
        save_model(model)  # Save model for future use

        # Predict LTP for the next day
        predicted_ltp = predict_next_day_price(model, ltp, strike_price)
        st.write(f"Predicted LTP for next day: {predicted_ltp:.2f}")

        # Stop Loss and Maximum LTP
        stop_loss = predicted_ltp * 0.98
        max_ltp = predicted_ltp * 1.02
        st.write(f"Stop Loss: {stop_loss:.2f}")
        st.write(f"Maximum LTP: {max_ltp:.2f}")

        # Recommendation
        if predicted_ltp > ltp:
            st.write("Recommendation: Profit")
            st.write(f"Expected Profit: {predicted_ltp - ltp:.2f}")
        else:
            st.write("Recommendation: Loss")
            st.write(f"Expected Loss: {ltp - predicted_ltp:.2f}")
