import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_prices(stocks):
    stock_data = pd.DataFrame()
    for stock_name in stocks:
        stock = yf.Ticker(stock_name)
        stock_history = stock.history(period="10y")
        stock_history.rename(columns={"Close": stock_name}, inplace=True)
        stock_data = pd.concat([stock_data, stock_history[stock_name]], axis=1)
    return stock_data

def get_latest_stock_prices(stock_list):
    latest_prices = {}
    for stock in stock_list:
        try:
            ticker = yf.Ticker(stock)
            latest_price = ticker.history(period='1d')['Close'][0]
            latest_prices[stock] = latest_price
        except:
            print(f"Error fetching data for {stock}")
    return latest_prices

def get_ML_prediction(stock_ticker):
    # Fetch historical data for Apple (AAPL)
    apple_data = yf.download(stock_ticker, start="2000-01-01", end="2023-07-18")

    # Feature engineering (you can add more relevant features)
    apple_data["SMA_50"] = apple_data["Close"].rolling(window=50).mean()
    apple_data["SMA_200"] = apple_data["Close"].rolling(window=200).mean()

    # Drop missing values and select relevant features
    apple_data.dropna(inplace=True)
    X = apple_data[["Open", "High", "Low", "Volume", "SMA_50", "SMA_200"]]
    y = apple_data["Close"]

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    #print(f"Mean Squared Error: {mse}")

    # Predict stock price for tomorrow
    last_day_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(last_day_data)[0]
    #print(f"Predicted {stock_ticker} Stock Price for Tomorrow: {predicted_price}")
    return predicted_price

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

def get_DL_prediction():
    # Fetch historical data for Apple (AAPL)
    apple_data = yf.download("AAPL", start="2000-01-01", end="2023-07-17")
    stock_prices = apple_data["Close"].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_prices = scaler.fit_transform(stock_prices)

    # Create training and testing datasets
    train_size = int(len(normalized_prices) * 0.8)
    test_size = len(normalized_prices) - train_size
    train_data, test_data = normalized_prices[0:train_size, :], normalized_prices[train_size:len(normalized_prices), :]

    # Prepare training data
    look_back = 5
    X_train, Y_train = create_dataset(train_data, look_back)

    # Prepare testing data
    X_test, Y_test = create_dataset(test_data, look_back)

    # Build LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(look_back, 1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

    # Make predictions for tomorrow
    last_day_data = test_data[-look_back:]
    last_day_data = last_day_data.reshape(1, look_back, 1)
    predicted_price = model.predict(last_day_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    print("Predicted Apple Stock Price for Tomorrow:", predicted_price[0][0])

def assemble_ML_stock_prediction(stock_list):
    stock_dictionary = {}
    for stock in stock_list:
        ticker = yf.Ticker(stock)
        latest_price = ticker.history(period='1d')['Close'][0]
        stock_dictionary[stock] = [latest_price, get_ML_prediction(stock)]
    return stock_dictionary

def main():
    # List of stock symbols/tickers
    stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "SGML", "ASML", "TSM"]

    # Fetch stock data
    stock_data = fetch_stock_prices(stock_list)

    # Export data to Excel
    file_name = "stock_data.csv"
    stock_data.to_csv(file_name)
    print(f"Stock data saved to {file_name}")

    # Get the machine learning stock price prediction for these stocks
    stock_dict = assemble_ML_stock_prediction(stock_list)
    print(stock_dict)

if __name__ == "__main__":
    main()
