import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

# Title
st.title("Intraday Trading Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and clean data
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.strip().str.replace(r'\s+\n', '')  # Clean headers
        data['EXPIRY DATE'] = pd.to_datetime(data['EXPIRY DATE'], format='%d-%b-%Y', errors='coerce')  # Parse dates
        data['STRIKE'] = data['STRIKE'].replace({',': ''}, regex=True).astype(float)  # Remove commas, convert to float
        data['LTP'] = data['LTP'].replace({',': ''}, regex=True).astype(float)  # Remove commas, convert to float
        data['OPTION TYPE'] = data['OPTION TYPE'].str.strip().str.upper()  # Standardize option types
        
        # Drop rows with invalid data
        data.dropna(subset=['EXPIRY DATE', 'STRIKE', 'LTP', 'OPTION TYPE'], inplace=True)
        
        # Sidebar filters
        selected_expiry = st.sidebar.selectbox("Select Expiry Date", sorted(data['EXPIRY DATE'].dropna().unique()))
        selected_option_type = st.sidebar.selectbox("Select Option Type", data['OPTION TYPE'].dropna().unique())
        selected_strike = st.sidebar.selectbox("Select Strike Price", sorted(data['STRIKE'].dropna().unique()))

        # Filter data based on selections
        filtered_data = data[
            (data['EXPIRY DATE'] == selected_expiry) &
            (data['OPTION TYPE'] == selected_option_type) &
            (data['STRIKE'] == selected_strike)
        ]

        if filtered_data.empty:
            st.error("No data available for the selected criteria. Please verify your selection.")
        else:
            st.write("Filtered Dataset:", filtered_data)

            # Current LTP
            current_ltp = filtered_data['LTP'].iloc[-1]

            # Check for sufficient data points
            if len(filtered_data) < 2:
                st.warning("Not enough data points for prediction. Displaying current LTP only.")
                st.write(f"Current LTP: {current_ltp:.2f}")
            else:
                # Prepare features and target
                X = np.arange(len(filtered_data)).reshape(-1, 1)  # Use indices as features
                y = filtered_data['LTP']

                # Train Linear Regression model
                model = LinearRegression()
                model.fit(X, y)

                # Predict next day's LTP
                next_day_index = np.array([len(filtered_data)]).reshape(-1, 1)
                predicted_ltp = model.predict(next_day_index)[0]

                # Calculate difference
                ltp_difference = predicted_ltp - current_ltp

                # Display results
                st.subheader("Prediction Results")
                st.write(f"Predicted LTP for the next day: {predicted_ltp:.2f}")
                st.write(f"Current LTP: {current_ltp:.2f}")
                st.write(f"Difference (Predicted - Current): {ltp_difference:+.2f}")

                # Recommendation
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
