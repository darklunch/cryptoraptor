import http.client
import json
import talib
import requests
import numpy as np
import time
import pandas as pd
import csv
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from collections import deque

# Define the API endpoint and webhook URL
api_endpoint = ""
webhook_url = ""

# Other parameters
show_only_usd_pairs = True
excluded_pairs = ["EUROC-USD", "USDT-USD", "DAI-USD", "GUSD-USD", "BUSD-USD"]

# Make an HTTP GET request to retrieve all products
while True:
    try:
        conn = http.client.HTTPSConnection(api_endpoint)
        payload = ''
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'CryptoRaptor/1.13'
        }
        conn.request("GET", "/products", payload, headers)
        res = conn.getresponse()
        product_data = res.read()
        products = json.loads(product_data)

        # Extract the product IDs from the list of products
        if show_only_usd_pairs:
            product_ids = [product['id'] for product in products if product['quote_currency'] == 'USD' and product['id'] not in excluded_pairs]
        else:
            product_ids = [product['id'] for product in products if product['id'] not in excluded_pairs]

        # Break out of the while loop if the product list is retrieved successfully
        break

    except Exception as e:
        # Output the error message to the console
        print("Error:", e)

        # Output the error message to the webhook
        payload = {
            "content": f"Error: {e}"
        }
        requests.post(webhook_url, json=payload)

        # Wait 2 minutes before trying again
        time.sleep(120)

# Set the machine learning parameters
enable_ml_signals = True

# Setup signal dictionary
n = 1  # number of last signals to keep
last_signals = {
    product_id: {"ml": deque(maxlen=n)} for product_id in product_ids
}

# Make a function to generate machine learning buy/sell signals
def generate_ml_signals(product_id, candles):
    global last_signals
    global enable_ml_signals
    global excluded_pairs

    # Check if the product ID is in the excluded_pairs list
    if product_id in excluded_pairs:
        return

    # Check if machine learning signals are enabled
    if not enable_ml_signals:
        return

    # Load historical data from a CSV file
    filename = f"{product_id}.csv"
    if not os.path.isfile(filename):
        print(f"Error: Historical data file not found for {product_id}")
        return
    data = pd.read_csv(filename)
    if data.empty:
        print(f"Error: Inputs are all NaN for {product_id}")
        return

    data.drop_duplicates(subset=['Time'], inplace=True)
    data.set_index('Time', inplace=True)

    # Combine the historical data with the current candles
    for candle in candles:
        timestamp = candle[0]
        if timestamp not in data.index:
            data.loc[timestamp] = candle[1:]

    # Convert the candlestick data to a Pandas DataFrame
    data = pd.DataFrame(candles, columns=["time", "low", "high", "open", "close", "volume"])

    # Add technical analysis indicators as features
    data["rsi"] = talib.RSI(data["close"], timeperiod=14)
    data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    data["bb_upper"], data["bb_middle"], data["bb_lower"] = talib.BBANDS(data["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    if len(candles) >= 2:
        high = max(data["high"])
        low = min(data["low"])
        for level in [0.236, 0.382, 0.5, 0.618, 0.764]:
            fib_price = high - level * (high - low)
            data[f"fib_{level}"] = fib_price
    for period in [5, 10, 20, 50, 100, 200]:  
        # Add Exponential Moving Averages (EMAs)
        data[f"ema_{period}"] = talib.EMA(data["close"], timeperiod=period)

    # Add lagged price and volume features
    for i in range(1, 5):
        data[f"close_lag{i}"] = data["close"].shift(i)
        data[f"volume_lag{i}"] = data["volume"].shift(i)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Extract the features and target variable
    features = data.drop(["time", "close"], axis=1)
    target = data["close"].shift(-1) > data["close"]

    # Fill missing values with the mean of the column
    features.fillna(features.mean(), inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a TensorFlow model on the training data
    num_features = X_train.shape[1]
    learning_rate_init = 0.00001
    max_iter = 9999

    # Check if a saved model exists
    model_file = f"{product_id}.h5"
    if os.path.isfile(model_file):
        model = tf.keras.models.load_model(model_file)
    else:
        # Create an early stopping callback to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Create a model checkpoint callback to save the best model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, save_best_only=True, save_weights_only=False)

        # Create a learning rate scheduler callback to prevent overfitting
        def lr_scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        # Train a TensorFlow model on the training data
        model = Sequential()
        model.add(Dense(64, input_dim=num_features, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate_init), metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=max_iter, validation_data=(X_test, y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

        # Save the model
        model.save(model_file)

    # Generate buy/sell signals based on the predicted probability
    if len(X_test) > 0:
        proba = model.predict(X_test)[-1][0]
        tf.keras.backend.clear_session() # Clear after each prediction to release memory
        if proba > 0.6:
            signal_type = "BUY"
            if proba > 0.8:
                signal_type = "STRONG BUY"
        elif proba < 0.4:
            signal_type = "SELL"
            if proba < 0.2:
                signal_type = "STRONG SELL"
        else:
            signal_type = None

        # Calculate the percent difference from the previous close price
        if len(candles) >= 2:
            prev_close = candles[-2][4]
            curr_close = candles[-1][4]
            percent_diff = round((curr_close - prev_close) / prev_close * 100, 2)
            if percent_diff >= 0:
                percent_diff = f"+{percent_diff}"

        # Check if the signal is different from the last signal received
        if signal_type != last_signals[product_id]["ml"]:
            prev_signal_type = last_signals[product_id]["ml"]  # Store the previous signal type
            last_signals[product_id]["ml"] = signal_type  # Update the last signal type with the current signal type

            # Output the signal to the webhook
            if signal_type:
                # Determine the color for the signal type
                color = None
                if signal_type == "STRONG BUY":
                    color = 0x00ff00  # green
                elif signal_type == "BUY" or signal_type == "SELL":
                    color = 0xffff00  # yellow
                elif signal_type == "STRONG SELL":
                    color = 0xff0000  # red

                # Construct the embed object
                description = f"{signal_type}"
                if prev_signal_type:
                    description += f" (prev: {prev_signal_type}, diff: {percent_diff}%)"
                else:
                    description += f" (diff: {percent_diff}%)"

                embed = {
                    "title": f"Signal received for {product_id}",
                    "description": description,
                    "color": color,
                    "footer": {
                        "text": "Not financial advice, not a financial advisor."
                    }
                }

                payload = {
                    "embeds": [embed]
                }

                requests.post(webhook_url, json=payload)

# Make a function to send a confirmation message to the webhook when the script starts
def send_confirmation():
    payload = {
        "content": "Crypto Raptor 1.13 has started, monitoring for signals."
    }
    requests.post(webhook_url, json=payload)

# Send a confirmation message to the webhook when the script starts
send_confirmation()

# Continuously run the script and check for signals every 2 minutes
while True:
    try:
        for product_id in product_ids:
            # Make an HTTP GET request to retrieve candlestick data for the current product
            conn = http.client.HTTPSConnection(api_endpoint)
            payload = ''
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'CryptoRaptor/1.13'
            }
            conn.request("GET", f"/products/{product_id}/candles", payload, headers)
            res = conn.getresponse()
            candle_data = res.read()
            candles = json.loads(candle_data)

            # Generate machine learning signals for the current product
            generate_ml_signals(product_id, candles)

        # Pause the script for 2 minutes before checking for signals again
        time.sleep(3600)

    except Exception as e:
        # Output the error message to the console
        print("Error:", e)

        # Output the error message to the webhook
        payload = {
            "content": f"Error: {e}"
        }
        requests.post(webhook_url, json=payload)

        # Wait for 1 second before trying again
        time.sleep(1)
