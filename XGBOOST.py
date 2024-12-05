from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import datetime

# Helper Functions (RSI calculation remains unchanged)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_yfinance_data(ticker, years):
    """Fetches historical data using yfinance for the past 'years'."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365)
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        # Rename columns for consistency
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        }, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def add_features(df):
    df['Moving Average'] = df['close'].rolling(window=5).mean()
    df['Volatility'] = df['close'].rolling(window=5).std()
    df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(4)
    df['RSI'] = calculate_rsi(df['close'], period=14)
    df.dropna(inplace=True)
    return df

def train_and_predict(data, model, scaler):
    features = ['Moving Average', 'Volatility', 'EMA', 'Momentum', 'RSI']
    X = data[features]
    y = data['close']
    
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    latest_data = data.tail(1)
    latest_X = latest_data[features]
    latest_X_scaled = scaler.transform(latest_X)

    next_day_prediction = model.predict(latest_X_scaled)
    return next_day_prediction[0]

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
    return mse, rmse, mape


def walk_forward_validation(data, model, scaler, train_size=0.8):  # Added train_size parameter
    features = ['Moving Average', 'Volatility', 'EMA', 'Momentum', 'RSI']
    X = data[features]
    y = data['close']
    
    split_index = int(len(X) * train_size)  # Determine split point based on train_size
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model.fit(X_train_scaled, y_train)  # Train on the in-sample set

    # Predict on both the in-sample (for comparison) and out-of-sample (true next-day prediction)
    in_sample_predictions = model.predict(X_train_scaled)
    out_of_sample_predictions = model.predict(X_test_scaled)

    # Evaluate In-Sample Predictions
    mse_in, rmse_in, mape_in = evaluate_model(model, X_train_scaled, y_train)

    # Evaluate Out-of-Sample Predictions
    mse_out, rmse_out, mape_out = evaluate_model(model, X_test_scaled, y_test)

    return in_sample_predictions, out_of_sample_predictions

def main():
    ticker = input("Enter stock ticker: ").upper()
    years = 10 

    raw_data = fetch_yfinance_data(ticker, years)
    if raw_data.empty:
        return

    data = add_features(raw_data)

    param_grid = {  # Example parameter grid, adjust as needed
        'n_estimators': [200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.08],
        'max_depth': [3, 5, 7 , 9]
    }

    model = XGBRegressor(random_state=42)
    scaler = StandardScaler()

    tscv = TimeSeriesSplit(n_splits=3)  # Reduced for faster tuning
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)

    features = ['Moving Average', 'Volatility', 'EMA', 'Momentum', 'RSI']
    X = data[features]
    y = data['close']
    X_scaled = scaler.fit_transform(X)  # Fit on all data for GridSearchCV

    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    walk_forward_validation(data.copy(), best_model, scaler)

    # Walk-forward validation and prediction
    in_sample_preds, out_of_sample_preds = walk_forward_validation(data.copy(), best_model, scaler, train_size=0.9)

    next_day_price = train_and_predict(data, best_model, scaler)
    print(f"Predicted closing price for next day: {next_day_price:.2f}")

    # Plot predictions and actual values
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Actual Close Price')
    plt.plot(data.index[:len(in_sample_preds)], in_sample_preds, label='In-Sample Predictions', linestyle='--')

    # Plot out-of-sample predictions, aligning the index to the test set's index.
    plt.plot(data.index[-len(out_of_sample_preds):], out_of_sample_preds, label='Out-of-Sample Predictions', linestyle='-.')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Price Predictions')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
