import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
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

        # Clean column names: Strip spaces and convert to uppercase
        data.columns = data.columns.str.strip().str.upper()

        # Check for required columns
        required_columns = ['EXPIRY DATE', 'OPTION TYPE', 'STRIKE', 'LTP']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {', '.join(missing_columns)}")
            st.stop()

        # Handle 'EXPIRY DATE' column
        data['EXPIRY DATE'] = pd.to_datetime(data['EXPIRY DATE'], errors='coerce', dayfirst=True)
        data['LTP'] = data['LTP'].replace({',': ''}, regex=True)
        data['LTP'] = pd.to_numeric(data['LTP'], errors='coerce')
        data = data.dropna(subset=['LTP'])  # Drop rows with invalid 'LTP'

        # Sidebar Filters
        selected_expiry = st.sidebar.selectbox("Select Expiry Date", data['EXPIRY DATE'].dropna().unique())
        selected_option_type = st.sidebar.selectbox("Select Option Type", data['OPTION TYPE'].dropna().unique())
        selected_strike = st.sidebar.selectbox("Select Strike Price", sorted(data['STRIKE'].dropna().unique()))

        # Filter the data
        filtered_data = data[
            (data['EXPIRY DATE'] == selected_expiry) &
            (data['OPTION TYPE'] == selected_option_type) &
            (data['STRIKE'] == selected_strike)
        ]

        st.write("Filtered Data:")
        st.dataframe(filtered_data)

        if filtered_data.empty:
            st.warning("No data available for the selected criteria.")
        else:
            # Current LTP
            current_ltp = filtered_data['LTP'].iloc[-1]

            # Check if we have enough data for prediction
            if len(filtered_data) < 2:
                st.warning("Not enough data points for prediction. Displaying current LTP only.")
                st.write(f"Current LTP: {current_ltp:.2f}")
            else:
                # Features and Target
                X = np.arange(len(filtered_data)).reshape(-1, 1)
                y = filtered_data['LTP']

                # Train Linear Regression
                model = LinearRegression()
                model.fit(X, y)

                # Predict the next day's LTP
                next_day_index = np.array([len(filtered_data)]).reshape(-1, 1)
                predicted_ltp = model.predict(next_day_index)[0]

                # Calculate the difference
                ltp_difference = predicted_ltp - current_ltp

                # Display Results
                st.write("Prediction Results:")
                st.write(f"Predicted LTP for the next day: {predicted_ltp:.2f}")
                st.write(f"Current LTP: {current_ltp:.2f}")
                st.write(f"Difference (Predicted - Current): {ltp_difference:+.2f}")

                # Recommendation and Outcome
                if ltp_difference > 0:
                    recommendation = "Buy"
                    outcome = "Profit"
                else:
                    recommendation = "Do Not Buy"
                    outcome = "Loss"

                st.write(f"Recommendation: {recommendation}")
                st.write(f"Expected Outcome: {outcome}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
