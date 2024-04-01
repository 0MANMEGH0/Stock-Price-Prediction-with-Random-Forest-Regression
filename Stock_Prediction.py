import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Function to fetch historical stock prices using Yahoo Finance API
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Choose a stock and date range
ticker = 'NVDA'
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch historical stock prices
stock_prices = get_stock_data(ticker, start_date, end_date)

# Create features and labels
features = stock_prices.shift(1)  # Use previous day's closing price as a feature
labels = stock_prices

# Drop missing values
data = pd.concat([features, labels], axis=1)
data.columns = ['Previous_Close', 'Actual_Close']
data = data.dropna()

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data['Previous_Close'].values.reshape(-1, 1),
    data['Actual_Close'].values,
    test_size=0.2,
    random_state=42
)

# Choose and train a model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_data, train_labels)

# Make predictions on the test set
predictions = model.predict(test_data)

# Evaluate the model
mae = mean_absolute_error(test_labels, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize predicted vs. actual prices
plt.plot(test_labels, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()