import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Streamlit App Title
st.title("Intraday Trading Predictor")
st.write("Upload your trading CSV file in the prescribed format to predict LTP for the next day.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    try:
        # Load CSV into DataFrame
        data = pd.read_csv(uploaded_file)

        # Clean column names: Strip spaces and convert to lowercase
        data.columns = data.columns.str.strip().str.lower()

        # Display cleaned column names to check for any issues
        st.write("Uploaded Data:")
        st.write(f"Columns in the CSV: {data.columns}")
        
        # Handle 'expiry date' column carefully
        if 'expiry date' not in data.columns:
            st.error("The 'Expiry Date' column is missing in the uploaded file!")
        else:
            # Try to parse the 'expiry date' column, coercing errors
            data['expiry date'] = pd.to_datetime(data['expiry date'], errors='coerce', dayfirst=True)
            if data['expiry date'].isnull().any():
                st.warning("Some 'Expiry Date' entries could not be parsed and were set to NaT (Not a Time).")
            else:
                st.write("'Expiry Date' column parsed successfully.")
        
        # Handle 'ltp' column
        if 'ltp' not in data.columns:
            st.error("The 'LTP' column is missing in the uploaded file!")
        else:
            data = data.sort_values(by='expiry date')

        # Display the processed data
        st.write("Processed Data:")
        st.dataframe(data.head())

        # Sidebar Filters: Add checks for column existence
        if 'expiry date' not in data.columns:
            st.error("Column 'Expiry Date' not found in the data!")
        else:
            selected_expiry = st.sidebar.selectbox("Select Expiry Date", data['expiry date'].unique())
        
        if 'symbol' not in data.columns:
            st.error("Column 'Symbol' not found in the data!")
        else:
            selected_symbol = st.sidebar.selectbox("Select Symbol", data['symbol'].unique())

        if 'option type' not in data.columns:
            st.error("Column 'Option Type' not found in the data!")
        else:
            selected_option_type = st.sidebar.selectbox("Select Option Type", data['option type'].unique())

        # Filter the data based on user input
        filtered_data = data[
            (data['expiry date'] == selected_expiry) &
            (data['symbol'] == selected_symbol) &
            (data['option type'] == selected_option_type)
        ]
        
        st.write("Filtered Data:")
        st.dataframe(filtered_data)

        if not filtered_data.empty:
            # Features and Target for Prediction
            X = np.arange(len(filtered_data)).reshape(-1, 1)  # Using indices as a placeholder feature
            y = filtered_data['ltp']  # Target variable is 'LTP'
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Linear Regression Model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Prediction for next day
            next_day_index = np.array([len(filtered_data)]).reshape(-1, 1)
            predicted_ltp = model.predict(next_day_index)[0]
            
            # Display Prediction Results
            st.write("Prediction Results:")
            st.write(f"Predicted LTP for the next day: {predicted_ltp:.2f}")
            current_ltp = filtered_data['ltp'].iloc[-1]
            ltp_difference = predicted_ltp - current_ltp
            st.write(f"Current LTP: {current_ltp:.2f}")
            st.write(f"Difference (Predicted - Current): {ltp_difference:.2f}")
            
            # Buy or Not Decision
            decision = "Buy" if ltp_difference > 0 else "Do Not Buy"
            profit_or_loss = "Profit" if ltp_difference > 0 else "Loss"
            st.write(f"Recommendation: {decision}")
            st.write(f"Expected Outcome: {profit_or_loss}")
        else:
            st.warning("No data available for the selected criteria.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
