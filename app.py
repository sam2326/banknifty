import streamlit as st
import pandas as pd
import numpy as np
import datetime
import hashlib

# Function to load and clean the data
def load_and_clean_data(uploaded_file):
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Print column names to debug
    st.write("Column names in the uploaded CSV:")
    st.write(df.columns)
    
    # Standardize column names by replacing spaces with underscores and converting to lowercase
    df.columns = df.columns.str.replace(' ', '_').str.lower()  # Standardize column names
    
    # Drop rows with missing data
    df = df.dropna()
    
    # Inspect the first few rows of data to understand the structure
    st.write("First few rows of data:")
    st.write(df.head())
    
    # Modify column names based on the CSV structure
    # You can adjust the following line depending on your actual column names
    try:
        df = df[['datetime', 'stock_code', 'close', 'volume']]
    except KeyError:
        st.error("The required columns are missing or misnamed. Please check the CSV structure.")
        return None

    # Convert the datetime column to proper datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Sort the data by datetime
    df = df.sort_values(by='datetime')

    return df

# Function to predict the LTP (Example Prediction)
def predict_ltp(df):
    # Dummy prediction model based on simple moving average (or use any model of your choice)
    df['predicted_ltp'] = df['close'].rolling(window=5).mean().shift(-1)  # Simple moving average prediction
    return df

# Function to calculate profit/loss based on current LTP and predicted LTP
def calculate_profit_loss(df):
    df['profit_loss'] = df['predicted_ltp'] - df['close']
    df['recommendation'] = np.where(df['profit_loss'] > 0, 'Buy', 'Do not Buy')
    return df

# Main app
def main():
    st.title("Bank Nifty Option Prediction App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        # Load and clean the data
        df = load_and_clean_data(uploaded_file)

        # If data cleaning fails, do not proceed
        if df is None:
            return
        
        # Show the cleaned data (optional)
        st.write("Cleaned Data:")
        st.write(df)

        # Predict LTP
        df = predict_ltp(df)
        
        # Calculate Profit/Loss and Recommendation
        df = calculate_profit_loss(df)
        
        # Show predictions
        st.write("Predictions (LTP, Profit/Loss, Recommendation):")
        st.write(df[['datetime', 'stock_code', 'close', 'predicted_ltp', 'profit_loss', 'recommendation']])

if __name__ == "__main__":
    main()
