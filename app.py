import streamlit as st
import requests
import hashlib
import datetime
import json

# Constants
APP_KEY = "1Zn5837j4439dt5_I%601255l3w6%2328d%289"
SECRET_KEY = "1645%24kX4a37C9lY6G90873Q3617%608%2528"
AUTHORIZATION_CODE = "49601661"
REDIRECT_URI = "http://localhost:3000/callback"
BASE_URL = "https://api.icicidirect.com"

# Generate Checksum Function
def generate_checksum(timestamp, json_data, secret_key):
    raw_data = timestamp + json_data + secret_key
    return hashlib.sha256(raw_data.encode('utf-8')).hexdigest()

# Exchange Authorization Code for Session Token
def get_session_token():
    endpoint = f"{BASE_URL}/oauth2/token"
    payload = {
        "grant_type": "authorization_code",
        "code": AUTHORIZATION_CODE,
        "redirect_uri": REDIRECT_URI,
        "client_id": APP_KEY,
        "client_secret": SECRET_KEY
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(endpoint, data=payload, headers=headers)
    if response.status_code == 200:
        token = response.json().get("access_token")
        if token:
            return token
        else:
            st.error("Token not found in response.")
            return None
    else:
        st.error(f"Failed to get session token: {response.text}")
        return None

# Fetch Live Data for BankNifty
def get_banknifty_data(session_token):
    endpoint = f"{BASE_URL}/equity/getquote"
    timestamp = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    payload = {
        "SessionToken": session_token,
        "Idirect_Userid": "NIFTY123",  # Replace with your ICICI user ID
        "STCK_CD": "BANKNIFTY"
    }
    json_data = json.dumps(payload)
    checksum = generate_checksum(timestamp, json_data, SECRET_KEY)

    headers = {
        "Content-Type": "application/json",
        "AppKey": APP_KEY,
        "Checksum": checksum
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json().get("Success")
        if data:
            return data
        else:
            st.error("No data found in response.")
            return None
    else:
        st.error(f"Failed to fetch BankNifty data: {response.text}")
        return None

# Dummy Prediction Model (Replace with your ML model)
def predict_banknifty(data):
    # Example: If the price increases, predict "BUY", else "SELL"
    ltp = float(data.get("LTP", 0))
    day_open = float(data.get("DayOpen", 0))
    if ltp > day_open:
        return "BUY"
    else:
        return "SELL"

# Place Order
def place_order(session_token, action):
    endpoint = f"{BASE_URL}/equity/placement"
    timestamp = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    payload = {
        "SessionToken": session_token,
        "Idirect_Userid": "NIFTY123",  # Replace with your ICICI user ID
        "order_stock_cd": "BANKNIFTY",
        "order_xchng_cd": "NSE",
        "order_product": "CASH",
        "order_type": "M",  # Market Order
        "order_validity": "T",  # Day
        "order_quantity": "1",  # Adjust quantity
        "order_rate": None,
        "order_flow": "B" if action == "BUY" else "S",
        "order_stp_loss_price": None,
        "order_disclosed_qty": "0",
        "order_trade_dt": timestamp.split(" ")[0]
    }
    json_data = json.dumps(payload)
    checksum = generate_checksum(timestamp, json_data, SECRET_KEY)

    headers = {
        "Content-Type": "application/json",
        "AppKey": APP_KEY,
        "Checksum": checksum
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        st.success(f"Order placed successfully: {response.json()}")
    else:
        st.error(f"Failed to place order: {response.text}")

# Streamlit Application
def main():
    st.title('BankNifty Prediction and Trading App')

    # Step 1: Button to get session token
    if st.button("Get Session Token"):
        session_token = get_session_token()
        if session_token:
            st.success("Session Token Retrieved Successfully!")
            st.session_state.session_token = session_token

    if 'session_token' not in st.session_state:
        st.warning("Please get the session token first.")
        return

    session_token = st.session_state.session_token

    # Step 2: Fetch BankNifty data
    if st.button("Get BankNifty Data"):
        data = get_banknifty_data(session_token)
        if data:
            st.write(data)

    # Step 3: Predict action based on data
    if 'data' in st.session_state:
        action = predict_banknifty(data)
        st.write(f"Predicted Action: {action}")

        # Step 4: Place Order Button
        if st.button(f"Place {action} Order"):
            place_order(session_token, action)

if __name__ == "__main__":
    main()
