import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
import datetime
import matplotlib.pyplot as plt


def fetch_yfinance_data(ticker, years):
    """Fetch historical stock data using yfinance."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365)

    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"No data available for {ticker}")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def prepare_prophet_data(df):
    """Prepare data for the Prophet model."""
    df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    df['y'] = pd.to_numeric(df['y'], errors= 'coerce')  # Convert to numeric, coerce invalid values
    df = df.dropna(subset=['y'], inplace =True)  # Drop rows with NaN values in 'y'
    return df


def add_features(df, window):
    """Add technical indicators like moving averages and RSI."""
    # Moving Averages
    df[f'MA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # Adding a small number prevents division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def train_and_predict_prophet(data, periods=1):
    """Train the Prophet model and predict future values."""
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    next_day_prediction = forecast['yhat'].iloc[-1]

    return next_day_prediction, forecast, model


def evaluate_prophet(data, forecast, test_size=0.2):
    """Evaluate the Prophet model's performance."""
    split_index = int(len(data) * (1 - test_size))
    y_true = data['y'].iloc[split_index:]
    y_pred = forecast['yhat'].iloc[split_index:]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("Prophet Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.6f}%")

    return mse, rmse, mape


def main():
    ticker = input("Enter stock ticker: ").upper()
    years = 10
    window = 20  # Use a market standard for moving average window

    raw_data = fetch_yfinance_data(ticker, years)
    if raw_data is None:  # Handle case where fetch_yfinance_data returns None
        return

    data_with_indicators = add_features(raw_data.copy(), window)  # Add technical indicators
    prophet_df = prepare_prophet_data(data_with_indicators)  # Prepare data for Prophet

    next_day_price, forecast, model = train_and_predict_prophet(prophet_df)  # Train Prophet

    evaluate_prophet(prophet_df, forecast)  # Evaluate the Prophet model

    print(f"Predicted closing price for next day using Prophet: {next_day_price:.2f}")

    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title(f'{ticker} Price Forecast with Prophet')
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.show()

if __name__ == "__main__":
    main()
