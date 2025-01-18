import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

#get historical stock data
msft = yf.Ticker("MSFT")
hist = msft.history(period="12mo")

print(hist.index)
print(hist['Close'])