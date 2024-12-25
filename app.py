    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get timestamp
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {timestamp})")
    return sentiment_score  # Return sentiment score for use in the prediction

# Function to predict LTP
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    trend_factor = 0.02 if market_trend == "up" else -0.02 if market_trend == "down" else 0
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.1
    strike_impact = (strike_price - ticker_price) * (-0.02 if market_trend == "down" else 0.01)
    sp500_impact = sp500_price * 0.003
    random_factor = random.uniform(-0.005, 0.01)
    momentum_factor = (ticker_price - ticker_price * 0.99) * (0.05 if market_trend == "up" else -0.05)

    predicted_ltp = current_ltp + trend_factor + sentiment_factor + strike_impact + sp500_impact + momentum_factor + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Function to load and clean Option Chain CSV
def load_and_clean_csv(file):
    try:
        # Load the CSV file
        df = pd.read_csv(file)
        
        # Correct column names (modify based on NSE CSV structure)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        # Drop rows with missing essential data
        df.dropna(subset=["Strike Price", "Call LTP", "Put LTP"], inplace=True)
        
        # Convert numeric columns to float
        numeric_cols = ["Strike Price", "Call LTP", "Put LTP"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Return cleaned DataFrame
        return df
    except Exception as e:
        st.error(f"Error loading or cleaning the file: {e}")
        return None

# Function to predict for the entire option chain
def predict_for_option_chain(df, current_price, india_vix, sp500_price, sentiment_score, market_trend):
    predictions = []
    for _, row in df.iterrows():
        try:
            predicted_call_ltp = predict_ltp(row["Call LTP"], current_price, row["Strike Price"], india_vix, sp500_price, sentiment_score, market_trend)
            predicted_put_ltp = predict_ltp(row["Put LTP"], current_price, row["Strike Price"], india_vix, sp500_price, sentiment_score, market_trend)
            predictions.append({
                "Strike Price": row["Strike Price"], 
                "Predicted Call LTP": predicted_call_ltp, 
                "Predicted Put LTP": predicted_put_ltp
            })
        except Exception as e:
            st.write(f"Error predicting for row {row['Strike Price']}: {e}")
    return pd.DataFrame(predictions)

# Main prediction logic
def predict():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    st.write(f"Current price for {ticker_name}: {ticker_price}")

    india_vix_ticker = yf.Ticker("^INDIAVIX")
    try:
        india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0
        st.write("Warning: Using default India VIX value.")
    st.write(f"India VIX: {india_vix}")

    sp500_price = fetch_sp500_data()
    if sp500_price is None:
        st.warning("Could not fetch S&P 500 data.")
    else:
        st.write(f"Current S&P 500 price: {sp500_price}")

    market_trend = determine_market_trend()
    st.write(f"Market Trend: {market_trend}")

    sentiment_score = display_sentiment_with_time()

    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    stop_loss_factor = 0.02 if market_trend == "up" else 0.01
    profit_factor = 0.05 if market_trend == "up" else 0.03

    stop_loss = round(predicted_ltp * (1 - stop_loss_factor), 2)
    max_ltp = round(predicted_ltp * (1 + profit_factor), 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {max_ltp}")

    rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and max_ltp else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    if rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

# Check if Option Chain CSV is uploaded
if uploaded_file:
    df = load_and_clean_csv(uploaded_file)
    if df is not None:
        st.write("Cleaned Option Chain Data", df)
        
        # Use previously defined market inputs
        ticker_price, _ = fetch_ticker_data(ticker_symbol)
        india_vix = 15.0  # Default or fetched India VIX
        sp500_price = fetch_sp500_data()
        market_trend = determine_market_trend()
        sentiment_score = display_sentiment_with_time()
        
        # Predict for the entire option chain
        predictions_df = predict_for_option_chain(df, ticker_price, india_vix, sp500_price, sentiment_score, market_trend)
        st.write("Predicted Option Chain", predictions_df)
        
        # Downloadable CSV
        st.download_button(
            label="Download Predictions as CSV",
            data=predictions_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_option_chain.csv",
            mime="text/csv"
        )

# Button to trigger main prediction logic
if st.button("Get Prediction"):
    predict()
