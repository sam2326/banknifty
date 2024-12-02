import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to load and clean data
def load_and_clean_data(uploaded_file):
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Inspect the columns and data
    st.write("Columns in uploaded file:")
    st.write(df.columns)

    st.write("First few rows of the data:")
    st.write(df.head())

    # Perform cleaning and adjust based on available columns
    # Check for necessary columns and rename if needed
    required_columns = ['datetime', 'symbol', 'strikePrice', 'ltp', 'contracts', 'valueInLakh']

    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing columns in uploaded file: {', '.join(missing_columns)}")
        return None

    # Proceed with cleaning if all required columns are present
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
    df['ltp'] = pd.to_numeric(df['ltp'], errors='coerce')
    df['contracts'] = pd.to_numeric(df['contracts'], errors='coerce')
    df['valueInLakh'] = pd.to_numeric(df['valueInLakh'], errors='coerce')

    # Drop rows with missing or invalid data
    df.dropna(subset=['datetime', 'ltp', 'strikePrice'], inplace=True)

    # Filter and clean data as needed
    return df

# Function to predict future LTP (dummy model for illustration)
def predict_ltp(df):
    # Create a linear regression model for LTP prediction based on strike price
    model = LinearRegression()
    model.fit(df[['strikePrice']], df['ltp'])

    # Predict LTP for the next data point (for example, predicting for the next strike price)
    predicted_ltp = model.predict([[df['strikePrice'].max() + 10]])  # Example prediction for a higher strike price

    return predicted_ltp[0]

# Main function to run the Streamlit app
def main():
    st.title('Bank Nifty Option Prediction App')

    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)

        if df is not None:
            # Display cleaned data
            st.write("Cleaned Data Preview:")
            st.write(df.head())

            # Allow the user to select strike price and LTP
            strike_price = st.selectbox("Select Strike Price", df['strikePrice'].unique())
            selected_data = df[df['strikePrice'] == strike_price]

            current_ltp = selected_data['ltp'].iloc[0] if not selected_data.empty else 0
            st.write(f"Current LTP for strike {strike_price}: {current_ltp}")

            # Get prediction
            predicted_ltp = predict_ltp(df)
            st.write(f"Predicted LTP for strike {strike_price}: {predicted_ltp}")

            # Calculate profit or loss
            profit_or_loss = predicted_ltp - current_ltp
            if profit_or_loss > 0:
                recommendation = "Profit - Buy"
            else:
                recommendation = "Loss - Don't Buy"

            st.write(f"Predicted Profit/Loss: {profit_or_loss}")
            st.write(f"Recommendation: {recommendation}")

# Run the app
if __name__ == "__main__":
    main()
