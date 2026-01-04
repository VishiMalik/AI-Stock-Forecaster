import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta

from src.data_fetch import fetch_history
from src.preprocess import scale_series, create_windows
from src.model_lstm import build_lstm
from src.decision import recommend
from src.news_integration import fetch_news, summarize_with_gemini


# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="AI Stock Forecaster", layout="wide")
st.title("📈 AI-Powered Stock Prediction & News Insights")

# -------------------- User Input --------------------
ticker = st.text_input("Enter Stock Ticker", value="AAPL")

# -------------------- Forecast Button --------------------
if st.button("Run Forecast"):
    # Fetch historical data
    df = fetch_history(ticker, period="2y")
    st.subheader("📊 Historical Closing Prices")
    st.line_chart(df['Close'])

    # -------------------- Ask Forecast Days --------------------
    forecast_days = st.number_input(
        "Enter forecast horizon (days)", 
        min_value=1, 
        max_value=365, 
        value=30, 
        step=1
    )

    # -------------------- Preprocessing --------------------
    scaled_close, scaler = scale_series(df['Close'].values)
    X, y = create_windows(scaled_close.flatten(), window_size=60)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build & train LSTM model
    model = build_lstm((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # -------------------- Forecasting --------------------
    last_sequence = scaled_close[-60:]
    future_predictions = []

    for _ in range(forecast_days):
        X_pred = last_sequence.reshape((1, 60, 1))
        next_price = model.predict(X_pred, verbose=0)[0][0]
        future_predictions.append(next_price)
        last_sequence = np.append(last_sequence[1:], next_price)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_predictions
    }).set_index("Date")

    combined = pd.concat([df[['Close']], forecast_df])

    st.subheader(f"🔮 Forecast for next {forecast_days} days")
    st.line_chart(combined)

    # -------------------- Recommendation --------------------
    current_price = df['Close'].iloc[-1]
    pred = future_predictions[-1]
    rec, diff = recommend(pred, current_price)

    st.metric("Last Predicted Close", f"${pred:.2f}")
    st.metric("Current Close", f"${current_price:.2f}")
    st.subheader(f"Recommendation: {rec} ({diff:.2%})")

    # -------------------- News Section --------------------
    st.subheader("📰 Latest News")
    news = fetch_news(f"{ticker} stock news")

    if news:
        st.write(news[:1500])  # show snippet

        # Always try Gemini summarization if API key is set
        try:
            summary = summarize_with_gemini(news)
            if summary:
                st.subheader("✨ Gemini Summary")
                st.write(summary)
        except Exception as e:
            st.warning("Gemini summarization unavailable. Showing raw news only.")
    else:
        st.write("No recent news found.")
