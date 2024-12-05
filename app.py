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
            try:
                message = await websocket.recv()  # Receive real-time data
                market_data = json.loads(message)  # Parse the data

                # Debugging: Print the entire received market data
                print(f"Received market data: {market_data}")

                # Process and predict based on real-time data
                prediction = process_real_time_data(market_data)
                prediction_display.write(f"Prediction: {prediction}")  # Update Streamlit UI with prediction

            except Exception as e:
                print(f"Error processing message: {e}")
                continue  # Skip the message if there's an error

# Function to process real-time market data
def process_real_time_data(data):
    # Debugging: Check the structure of the data
    print(f"Processing data: {data}")

    # Example: Check if the expected keys exist in the data before accessing them
    try:
        # Debugging: List the keys in the incoming data for inspection
        print(f"Keys in received data: {data.keys()}")
        
        # Check if AAPL data exists in the incoming data
        if "AAPL" in data:
            price = data["AAPL"].get("price", 0)  # Safely access the price for AAPL
            sentiment = 1  # Placeholder sentiment (can be fetched from NLP models)

            # Predict BankNifty movement based on real-time data
            return predict_banknifty(price, sentiment)
        else:
            print("AAPL data not found in message")
            return "No AAPL Data"

    except KeyError as e:
        print(f"Error accessing key {e} in market data")
        return "Error processing data"

# Function to predict BankNifty's next movement based on live data
def predict_banknifty(price, sentiment):
    if sentiment > 0 and price > 35000:  # Example threshold for prediction
        return "BUY"
    else:
        return "SELL"

# Streamlit button to start real-time prediction
if st.button("Start Real-Time Prediction"):
    asyncio.run(connect_to_alpaca_websocket())
