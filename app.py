import yfinance as yf
import streamlit as st
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from textblob import TextBlob
from datetime import datetime

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions with Option Chain Analysis", layout="wide")

# Title of the App
st.title("Trading Predictions with Option Chain Analysis")

# Sidebar for Inputs
st.sidebar.title("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload NSE Option Chain CSV", type="csv")

# Function to clean and process the uploaded file
def load_and_clean_csv(file):
    try:
        # Load the CSV file
        df = pd.read_csv(file)
        
        # Clean column names (strip spaces and newlines)
        df.columns = [col.strip().replace("\n", "").replace(" ", "_") for col in df.columns]

        # Filter necessary columns and clean data
        df = df[["STRIKE_", "OPTION_TYPE_", "LTP_"]]
        df.rename(columns={"STRIKE_": "Strike Price", "LTP_": "LTP", "OPTION_TYPE_": "Option Type"}, inplace=True)

        # Separate Call and Put data into different rows
        call_df = df[df["Option Type"] == "Call"][["Strike Price", "LTP"]].rename(columns={"LTP": "Call LTP"})
        put_df = df[df["Option Type"] == "Put"][["Strike Price", "LTP"]].rename(columns={"LTP": "Put LTP"})

        # Merge Call and Put data on Strike Price
        option_chain = pd.merge(call_df, put_df, on="Strike Price", how="outer").fillna(0)

        # Convert numeric columns to float
        option_chain["Strike Price"] = option_chain["Strike Price"].str.replace(",", "").astype(float)
        option_chain["Call LTP"] = option_chain["Call LTP"].astype(float)
        option_chain["Put LTP"] = option_chain["Put LTP"].astype(float)

        return option_chain
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Function to predict LTP
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sentiment_score, market_trend):
    trend_factor = 0.02 if market_trend == "up" else -0.02 if market_trend == "down" else 0
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.1
    strike_impact = (strike_price - ticker_price) * (-0.02 if market_trend == "down" else 0.01)
    random_factor = random.uniform(-0.005, 0.01)
    predicted_ltp = current_ltp + trend_factor + sentiment_factor + strike_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main option chain prediction logic
def predict_option_chain(df, ticker_price, india_vix, sentiment_score, market_trend):
    predictions = []
    for _, row in df.iterrows():
        predicted_call_ltp = predict_ltp(row["Call LTP"], ticker_price, row["Strike Price"], india_vix, sentiment_score, market_trend)
        predicted_put_ltp = predict_ltp(row["Put LTP"], ticker_price, row["Strike Price"], india_vix, sentiment_score, market_trend)
        predictions.append({
            "Strike Price": row["Strike Price"],
            "Predicted Call LTP": predicted_call_ltp,
            "Predicted Put LTP": predicted_put_ltp
        })
    return pd.DataFrame(predictions)

# Load and process file if uploaded
if uploaded_file:
    option_chain_df = load_and_clean_csv(uploaded_file)
    if option_chain_df is not None:
        st.write("Processed Option Chain Data", option_chain_df)

        # Example ticker price and market details
        ticker_price = 51233.00  # Replace with live data or user input
        india_vix = 15.0         # Default India VIX value
        sentiment_score = 0.5    # Replace with sentiment analysis results
        market_trend = "neutral" # Replace with trend analysis results

        # Predict option chain
        predicted_option_chain = predict_option_chain(option_chain_df, ticker_price, india_vix, sentiment_score, market_trend)
        st.write("Predicted Option Chain", predicted_option_chain)

        # Downloadable CSV
        st.download_button(
            label="Download Predictions as CSV",
            data=predicted_option_chain.to_csv(index=False).encode("utf-8"),
            file_name="predicted_option_chain.csv",
            mime="text/csv"
        )
