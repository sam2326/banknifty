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

        # Clean column names: Strip spaces and convert to uppercase
        data.columns = data.columns.str.strip().str.upper()

        # Display cleaned column names to check for any issues
        st.write("Uploaded Data:")
        st.write(f"Columns in the CSV: {data.columns}")
        
        # Check for required columns: Expiry Date, LTP, Option Type
        required_columns = ['EXPIRY DATE', 'OPTION TYPE', 'LTP']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {', '.join(missing_columns)}")
            st.stop()

        # Handle 'EXPIRY DATE' column
        if 'EXPIRY DATE' in data.columns:
            # Try to parse the 'EXPIRY DATE' column, coercing errors
            data['EXPIRY DATE'] = pd.to_datetime(data['EXPIRY DATE'], errors='coerce', dayfirst=True)
            if data['EXPIRY DATE'].isnull().any():
                st.warning("Some 'EXPIRY DATE' entries could not be parsed and were set to NaT (Not a Time).")
            else:
                st.write("'EXPIRY DATE' column parsed successfully.")
        
        # Clean the 'LTP' column: remove commas and handle invalid values like '-'
        if 'LTP' in data.columns:
            # Remove commas and convert to numeric, invalid parsing will result in NaN
            data['LTP'] = data['LTP'].replace({',': ''}, regex=True)
            data['LTP'] = pd.to_numeric(data['LTP'], errors='coerce')
            
            # Handle rows where 'LTP' is NaN (due to invalid values)
            data = data.dropna(subset=['LTP'])
        
        if 'LTP' not in data.columns or data['LTP'].isnull().all():
            st.error("The 'LTP' column is missing or invalid in the uploaded file!")
            st.stop()

        # Display the processed data
        st.write("Processed Data:")
        st.dataframe(data.head())

        # Sidebar Filters: Add checks for column existence
        if 'EXPIRY DATE' in data.columns:
            selected_expiry = st.sidebar.selectbox("Select Expiry Date", data['EXPIRY DATE'].dropna().unique())
        
        if 'OPTION TYPE' in data.columns:
            selected_option_type = st.sidebar.selectbox("Select Option Type", data['OPTION TYPE'].dropna().unique())

        # Filter the data based on user input
        filtered_data = data[
            (data['EXPIRY DATE'] == selected_expiry) &
            (data['OPTION TYPE'] == selected_option_type)
        ]
        
        st.write("Filtered Data:")
        st.dataframe(filtered_data)

        if not filtered_data.empty:
            # Features and Target for Prediction
            X = np.arange(len(filtered_data)).reshape(-1, 1)  # Using indices as a placeholder feature
            y = filtered_data['LTP']  # Target variable is 'LTP'
            
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
            current_ltp = filtered_data['LTP'].iloc[-1]
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
