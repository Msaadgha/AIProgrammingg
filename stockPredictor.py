
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import yfinance as yf
from datetime import date, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Parameters for retrieving the stock data
start_date = "2024-01-01"
end_date = date.today().strftime("%Y-%m-%d")
selected_stock = input("Enter a stock ticker: ")

# Variable to control the length of the prediction
prediction_days = input("Enter the number of days to predict: ")
prediction_days = int(prediction_days)

#Days to show on the graph
days_to_show = input("Days to show on graph: ")
days_to_show = int(days_to_show) * -1

def get_stock_data(ticker, start, end):
    ticker_data = yf.download(ticker, start, end)  # downloading the stock data from START to TODAY
    ticker_data.reset_index(inplace=True)  # put date in the first column
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date']).dt.tz_localize(None)
    return ticker_data

# Get the Dataset
df = get_stock_data(selected_stock, start_date, end_date)

# Convert index to DateTimeIndex
df.set_index('Date', inplace=True)

# Set Target Variable
output_var = pd.DataFrame(df['Adj Close'])
# Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

# Adding Moving Averages as features
df['MA10'] = df['Adj Close'].rolling(window=10).mean()
df['MA50'] = df['Adj Close'].rolling(window=50).mean()

# Updating the features list
features.extend(['MA10', 'MA50'])

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features + ['Adj Close']].dropna())
scaled_df = pd.DataFrame(columns=features + ['Adj Close'], data=scaled_data, index=df.dropna().index)

# Splitting to Training set and Test set
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(scaled_df):
    X_train, X_test = scaled_df[features].iloc[train_index], scaled_df[features].iloc[test_index]
    y_train, y_test = scaled_df['Adj Close'].iloc[train_index], scaled_df['Adj Close'].iloc[test_index]

# Process the data for LSTM
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(32, activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Fit the LSTM model
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
lstm.fit(X_train, y_train, epochs=50, batch_size=50, verbose=1, callbacks=[early_stopping])

# Predict
y_pred = lstm.predict(X_test)

# Inverse transform the predictions and true values
y_test_inv = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.values.reshape(-1, 1)), axis=1))[:, -1]
y_pred_inv = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred), axis=1))[:, -1]

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f'Mean Absolute Error: ${mae:.2f}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'Mean Squared Error: ${mse:.2f}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: ${rmse:.2f}')

# Forecast the next few days based on the prediction_days variable
last_sequence = np.array(scaled_df[features].iloc[-1]).reshape(1, 1, len(features))
additional_days = pd.date_range(df.index[-1] + timedelta(days=1), df.index[-1] + timedelta(days=prediction_days))

forecast = []
for _ in range(len(additional_days)):
    next_pred = lstm.predict(last_sequence)
    forecast.append(next_pred[0][0])
    next_sequence = np.concatenate((last_sequence[:, :, 1:], next_pred.reshape(1, 1, 1)), axis=2)
    last_sequence = next_sequence

forecast = np.array(forecast).reshape(-1, 1)
last_sequence_repeated = np.tile(last_sequence[0, 0, :], (len(forecast), 1))

forecast_inv = scaler.inverse_transform(np.concatenate((last_sequence_repeated, forecast), axis=1))[:, -1]

# Print the x and y data for the forecast
print("Forecast Dates:", additional_days)
print("Forecast Values:", forecast_inv)


# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='True Value')
plt.plot(df.index[-len(y_pred_inv):], y_pred_inv, label='Pred Value', linestyle='dashed')
plt.plot(additional_days, forecast_inv, label='Forecast', linestyle='dashed', color='red')
plt.title("Prediction by LSTM")
plt.xlabel('Date')
plt.ylabel('USD')
plt.legend()

# Set the x-axis limit to show only the last 7 days
plt.xlim(df.index[days_to_show], additional_days[-1])

plt.show()


def get_implied_volatility(ticker):
    # Fetch options data for the given ticker
    stock = yf.Ticker(ticker)

    try:
        exp_dates = stock.options
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {e}")
        return None

    # If no expiration dates are available, return None
    if not exp_dates:
        print(f"No options data available for {ticker}.")
        return None

    # Fetch options data for the first available expiration date
    options_chain = stock.option_chain(exp_dates[0])

    # Get the mid-price for calls and puts
    calls = options_chain.calls
    puts = options_chain.puts

    # Calculate the implied volatility (IV)
    # Note: IV is typically provided by the data source or calculated using complex models
    # Here, we'll simply average the implied volatility for illustration purposes

    call_iv = calls['impliedVolatility'].dropna().values
    put_iv = puts['impliedVolatility'].dropna().values

    if len(call_iv) == 0 or len(put_iv) == 0:
        print(f"No implied volatility data available for {ticker}.")
        return None

    avg_call_iv = np.mean(call_iv)
    avg_put_iv = np.mean(put_iv)

    avg_iv = (avg_call_iv + avg_put_iv) / 2

    return avg_iv


iv = get_implied_volatility(selected_stock)
print("\n\n\n")

if iv is not None:
    print(f"The IV index for {selected_stock} is {iv:.2%}")
else:
    print(f"Could not retrieve the implied volatility index for {selected_stock}.")