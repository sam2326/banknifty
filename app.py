import alpaca_trade_api as tradeapi
import asyncio
import websockets
import json
import streamlit as st

# Your new Alpaca API credentials
api_key = "PKA1TOTBSAIUFXKRUEVG"
api_secret = "ZU7hlWX9OZR2eY01GgXSeHdoRn5DKoSyuT8fhYyY"
ws_url = "wss://stream.data.alpaca.markets/v2/iex"  # Alpaca WebSocket URL for real-time data

# Initialize Alpaca REST API
api = tradeapi.REST(api_key, api_secret, base_url="https://paper-api.alpaca.markets")

# Streamlit UI setup
st.title("Real-Time BankNifty Prediction App")
st.write("This app predicts the next day's movement of BankNifty based on real-time market data.")

prediction_display = st.empty()  # Placeholder to update prediction dynamically

# Function to connect to Alpaca WebSocket and receive real-time data
async def connect_to_alpaca_websocket():
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    # Connect to Alpaca WebSocket
    async with websockets.connect(ws_url, extra_headers=headers) as websocket:
        subscribe_message = {
            "action": "subscribe",
            "symbols": ["AAPL", "SPY", "GOOG"]  # Modify with relevant symbols
        }
        await websocket.send(json.dumps(subscribe_message))  # Send subscribe message
        
        while True:
            message = await websocket.recv()  # Receive real-time data
            market_data = json.loads(message)  # Parse the data
            
            # Process and predict based on real-time data
            prediction = process_real_time_data(market_data)
            prediction_display.write(f"Prediction: {prediction}")  # Update Streamlit UI with prediction

# Function to process real-time market data
def process_real_time_data(data):
    # Extract real-time price and sentiment (you can modify the logic as needed)
    price = data.get("AAPL", {}).get("price", 0)  # Example for AAPL, modify accordingly
    sentiment = 1  # Placeholder sentiment (can be fetched from NLP models)
    
    # Predict BankNifty movement based on real-time data
    return predict_banknifty(price, sentiment)

# Function to predict BankNifty's next movement based on live data
def predict_banknifty(price, sentiment):
    if sentiment > 0 and price > 35000:  # Example threshold for prediction
        return "BUY"
    else:
        return "SELL"

# Streamlit button to start real-time prediction
if st.button("Start Real-Time Prediction"):
    asyncio.run(connect_to_alpaca_websocket())
