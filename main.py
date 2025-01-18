import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

#get historical stock data
msft = yf.Ticker("MSFT")
hist = msft.history(period="12mo")

#split data into training/validation sets
train_data = hist.iloc[:-21]
validation_data = hist.iloc[-21:]

train_days = (train_data.index - train_data.index[0]).days
validation_days = (validation_data.index - train_data.index[0]).days
train_data_array = np.array([train_days, train_data['Close'].values]).T
validation_data_array = np.array([validation_days, validation_data['Close'].values]).T

# Fit a polynomial to the training data using numpy
degree = 6
coefficients = np.polyfit(train_days, train_data['Close'], degree)
polynomial = np.poly1d(coefficients)
train_poly_values = polynomial(train_days)
validation_poly_values = polynomial(validation_days)

# Fit a polynomial to the training data using manual method
degree_manual = 7
X_train = np.vander(train_days, degree_manual)
coefficients_manual = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ train_data['Close']
train_poly_values_manual = X_train @ coefficients_manual
validation_poly_values_manual = np.vander(validation_days, degree_manual) @ coefficients_manual

# Plot the data and the polynomial fits using numpy
plt.figure(figsize=(10, 5))
plt.plot(train_data.index, train_data['Close'], label='Training Data')
plt.plot(validation_data.index, validation_data['Close'], label='Validation Data')
plt.plot(train_data.index, train_poly_values, label='Polynomial Fit (numpy)')
plt.plot(validation_data.index, validation_poly_values, label='Polynomial Fit (numpy)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('MSFT Closing Prices - Polynomial Fit (numpy)')
plt.legend()
plt.show()

# Plot the data and the polynomial fits using manual method
plt.figure(figsize=(10, 5))
plt.plot(train_data.index, train_data['Close'], label='Training Data')
plt.plot(validation_data.index, validation_data['Close'], label='Validation Data')
plt.plot(train_data.index, train_poly_values_manual, label='Polynomial Fit (manual)', linestyle='--')
plt.plot(validation_data.index, validation_poly_values_manual, label='Polynomial Fit (manual)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('MSFT Closing Prices - Polynomial Fit (manual)')
plt.legend()
plt.show()
