import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and clean CSV
def load_and_clean_data(uploaded_file):
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Clean and format data (basic cleaning)
    df.columns = df.columns.str.replace(' ', '_').str.lower()  # Standardize column names
    df = df.dropna()  # Drop rows with missing data
    
    # Filter and clean the data to show only necessary columns (modify as per your data structure)
    df = df[['datetime', 'stock_code', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort the data by datetime
    df = df.sort_values(by='datetime')
    
    return df

# Function to train the prediction model (simple regression model)
def train_model(df):
    # Feature extraction (You may include more sophisticated features)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_of_day'] = df['datetime'].dt.hour
    X = df[['volume', 'day_of_week', 'hour_of_day']]
    y = df['close']
    
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a Linear Regression Model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

# Function to make predictions
def make_prediction(model, scaler, volume, day_of_week, hour_of_day):
    X_pred = np.array([[volume, day_of_week, hour_of_day]])
    X_pred_scaled = scaler.transform(X_pred)
    predicted_price = model.predict(X_pred_scaled)
    return predicted_price[0]

# Streamlit App Layout
st.title("BankNifty Stock Prediction App")
st.write("Upload a CSV file with the historical data, and the app will predict the future LTP for a given strike.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    # Show the cleaned data preview
    st.subheader("Cleaned Data")
    st.write(df.head())

    # Train prediction model
    model, scaler = train_model(df)
    
    # User Input: Enter data for prediction
    strike_price = st.selectbox('Select Strike Price', df['stock_code'].unique())
    volume = st.number_input('Enter Volume', min_value=1, value=1000000)
    day_of_week = st.selectbox('Select Day of Week', [0, 1, 2, 3, 4, 5, 6])  # 0=Monday, 6=Sunday
    hour_of_day = st.selectbox('Select Hour of Day', [i for i in range(24)])
    
    # Make Prediction
    predicted_ltp = make_prediction(model, scaler, volume, day_of_week, hour_of_day)
    
    # Calculate Profit/Loss
    current_ltp = df[df['stock_code'] == strike_price]['close'].iloc[-1]
    profit_loss = predicted_ltp - current_ltp
    recommendation = "Buy" if profit_loss > 0 else "Do not Buy"
    
    # Output Results
    st.subheader("Prediction Results")
    st.write(f"Predicted LTP: {predicted_ltp:.2f}")
    st.write(f"Current LTP: {current_ltp:.2f}")
    st.write(f"Profit/Loss: {profit_loss:.2f}")
    st.write(f"Recommendation: {recommendation}")
    
    # Display a plot of historical data (optional)
    st.subheader("Historical Data Visualization")
    fig, ax = plt.subplots()
    ax.plot(df['datetime'], df['close'])
    ax.set_title("Stock Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)
