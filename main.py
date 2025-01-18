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

# Standardize the input data
mean_train_days = np.mean(train_days)
std_train_days = np.std(train_days)
train_days_standardized = (train_days - mean_train_days) / std_train_days
validation_days_standardized = (validation_days - mean_train_days) / std_train_days

# Fit a polynomial to the training data manually
degree = 6  # Degree of the polynomial
X_train = np.vander(train_days_standardized, degree + 1)
y_train = train_data['Close'].values

# Solve for the polynomial coefficients using the normal equation
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Generate polynomial values for training and validation data
train_poly_values = X_train @ coefficients
X_validation = np.vander(validation_days_standardized, degree + 1)
validation_poly_values = X_validation @ coefficients

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
plt.plot(train_data.index, train_poly_values, label='Polynomial Fit', linestyle='--')
plt.plot(validation_data.index, validation_poly_values, label='Polynomial Fit', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('MSFT Closing Prices - Polynomial Fit')
plt.legend()
plt.show()
