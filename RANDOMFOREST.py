# This script is focusing on predicting the next day's closing price using a Random Forest Regressor. 
# It calculates technical indicators, trains the model, and evaluates its performance on a test dataset.
# It also prints the predicted and actual prices for a specific date.

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import datetime

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data available for {symbol} between {start_date} and {end_date}.")
            return None  # Return None to indicate no data
        stock_data = calculate_technical_indicators(stock_data)
        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:  # Catch any other exceptions during download
        print(f"An error occurred while fetching data for {symbol}: {e}")
        return None

indicator_window = {
    'SMA': 20,  # Simple Moving Average - 20 days
    'RSI': 14,  # Relative Strength Index - 14 days
    # ... add other indicators and window sizes
}

def calculate_technical_indicators(data):
    """Calculates technical indicators.  Handles potential errors."""
    try:
        # Efficiently calculate indicators using rolling apply and handling zero division
        for indicator in indicator_window:
            window = indicator_window[indicator]
            if indicator == 'RSI':  # Handle RSI separately
                delta = data['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=window, min_periods=1).mean()
                avg_loss = loss.rolling(window=window, min_periods=1).mean()
                rs = avg_gain / (avg_loss + 1e-9) # add small number to avg_loss to prevent division by zero
                data[indicator] = 100 - (100 / (1 + rs))
            
            else: # Calculate simple moving averages as before
                data[indicator] = data['Close'].rolling(window=window, min_periods=1).mean()



        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14, min_periods=1).mean()

        data['Target'] = data['Close'].shift(-1)  # Predict next day's price

        return data
    except Exception as e: # handle exceptions during indicator calculation 
        print(f"Error calculating indicators: {e}")
        return None

def prepare_features(data):
    """
    Prepare features for machine learning model.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR']
    X = data[features]
    y = data['Target']
    return X, y

def train_model(X_train, y_train, hyperparams=None):
    """
    Train a Random Forest Regressor model with hyperparameter tuning (optional).

    Args:
        X_train: Training features.
        y_train: Training targets.
        hyperparams (dict, optional): Hyperparameter dictionary for GridSearchCV.

    Returns:
        tuple: Trained model and scaler (if hyperparams is None) or best model and params (if hyperparams is provided).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if hyperparams is None:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        return model, scaler
    else:
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, hyperparams, scoring='neg_mean_squared_error', cv=5, verbose=2)
        grid_search.fit(X_train_scaled, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

def predict_next_day(model, scaler, latest_data):
    """
    Predict the next day's stock price.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR']
    X_latest = latest_data[features].iloc[-1].values.reshape(1, -1)
    X_latest_scaled = scaler.transform(X_latest)
    predicted_price = model.predict(X_latest_scaled)[0]
    return predicted_price

if __name__ == "__main__":
    stock_symbols = ["AAPL", "GOOG", "MSFT"]
    end_date = datetime.date(2024, 12, 3)  # Ensures data up to December 2nd is included
    start_date = end_date - datetime.timedelta(days=3650) # 10 years of data
    test_size = 0.2


    hyperparams = {  # Example values. Tune as needed.
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    all_predictions = {}

    for stock in stock_symbols:
        stock_data = fetch_stock_data(stock, start_date, end_date)

        if stock_data is None: # skip if no data available for a stock
            print(f"Skipping {stock} due to missing data.")
            continue

        X, y = prepare_features(stock_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False) # do not shuffle for timeseries

        model, best_params = train_model(X_train, y_train, hyperparams)
        
        # Refit model on all scaled training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)

        latest_data = stock_data.iloc[-1:]  # Get the latest available data for prediction
        predicted_price = predict_next_day(model, scaler, latest_data)

        try:
            actual_price = float(stock_data['Close'].iloc[-1])  # Handle potential errors
            print(f"Actual Price for {stock} on {latest_data.index[-1].date()}: ${actual_price:.2f}") # Print date and price
            print(f"Predicted Price for {stock} on {latest_data.index[-1].date() + pd.DateOffset(days=1)}: ${predicted_price:.2f}")


            # Evaluate the model
            y_pred = model.predict(scaler.transform(X_test))
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Percentage Error: {mape:.6f}%")

        except (IndexError, TypeError, ValueError) as e:
            print(f"Error evaluating predictions: {e}. Check for missing or inconsistent data.")

        all_predictions[stock] = predicted_price # store all predictions
    
    print("All Predictions:", all_predictions)
