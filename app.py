import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Function to clean and adjust the format of the uploaded file
def load_and_clean_data(uploaded_file):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Automatically adjust and clean the data format
        # Renaming columns for easier handling
        df.rename(columns={
            'Instrument': 'instrument',
            'Symbol': 'stock_code',
            'Expiry Date': 'expiry_date',
            'Strike Price': 'strike_price',
            'Option Type': 'option_type',
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close',
            'LTP': 'ltp',
            'Settle Price': 'settle_price',
            'Contracts': 'contracts',
            'Value In Lakh': 'value_in_lakh',
            'Open Interest': 'open_interest',
            'Change in OI': 'change_in_oi'
        }, inplace=True)
        
        # Ensuring the necessary columns exist
        required_columns = ['stock_code', 'strike_price', 'ltp', 'contracts', 'value_in_lakh']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in uploaded file: {', '.join(missing_columns)}")
            return None

        # Convert necessary columns to numeric for calculations
        for col in ['strike_price', 'ltp', 'contracts', 'value_in_lakh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN in required columns
        df.dropna(subset=required_columns, inplace=True)

        return df
    except Exception as e:
        st.error(f"Error while processing the file: {str(e)}")
        return None

# Function to predict LTP and calculate profit or loss
def predict_ltp(df):
    # Simulated prediction logic (placeholder)
    df['predicted_ltp'] = df['ltp'] * np.random.uniform(0.98, 1.02, len(df))

    # Calculate profit or loss
    df['profit_loss'] = df['predicted_ltp'] - df['ltp']

    # Add a recommendation column
    df['recommendation'] = np.where(df['profit_loss'] > 0, 'Buy', 'Don’t Buy')
    
    return df

# Streamlit app interface
st.title("BankNifty Stock Prediction App")
st.write("Upload a CSV file containing BankNifty options data.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and clean the data
    df = load_and_clean_data(uploaded_file)

    if df is not None:
        st.write("### Cleaned Data")
        st.dataframe(df)

        # Predict and display results
        df = predict_ltp(df)
        st.write("### Predictions")
        st.dataframe(df[['stock_code', 'strike_price', 'ltp', 'predicted_ltp', 'profit_loss', 'recommendation']])
        
        # Option to download the results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='banknifty_predictions.csv',
            mime='text/csv'
        )

        # Input fields for user-specific strike price and LTP
        st.write("### Analyze Specific Strike Price")
        strike_price_input = st.number_input("Enter Strike Price:", value=0.0)
        current_ltp_input = st.number_input("Enter Current LTP:", value=0.0)

        if strike_price_input > 0 and current_ltp_input > 0:
            # Filter data for user input and calculate potential profit/loss
            specific_df = df[df['strike_price'] == strike_price_input]
            if not specific_df.empty:
                specific_df['input_profit_loss'] = specific_df['predicted_ltp'] - current_ltp_input
                st.write(f"### Analysis for Strike Price {strike_price_input}")
                st.dataframe(specific_df[['stock_code', 'ltp', 'predicted_ltp', 'input_profit_loss', 'recommendation']])
            else:
                st.warning("No data found for the entered strike price.")
