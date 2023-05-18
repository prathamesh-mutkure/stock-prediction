# Importing all the required libraries

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from nsepython import nsefetch


# Reading the CSV fileS

# df = pd.read_csv('C:\\Users\\umale\\Desktop\\project files\\DATASET\\TATAMOTORS.NS.csv')
symbol = "TATAMOTORS"
series = "EQ"
start_date = "18-03-2023"
end_date ="22-03-2023"
payload = nsefetch("https://www.nseindia.com/api/historical/cm/equity?symbol="+symbol+"&series=[%22"+series+"%22]&from="+start_date+"&to="+end_date+"")
df = pd.DataFrame.from_records(payload["data"])
# df = nsefetch('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY')
df.set_index('Date', inplace=True)
df.tail()
#GET THE NUMBER OF TRAING DAYS
df.shape
#find null values in dataset
df.isnull().sum()

df.info()
# Visualizing the stock prices

df['Adj Close'].plot(label='TATAMOTRS', figsize=(16, 6), title='Adjusted Closing Price', color='red', linewidth=1.0, grid=True)
plt.legend()

# Rolling Mean / Moving Average to remove the noise in the graph and smoothen it

close_col = df['Adj Close']
mvag = close_col.rolling(window=100).mean()     # Taking an average over the window size of 100.
# Increasing the window size can make it more smoother, but less informative and vice-versa.

# Visualizing Rolling Mean and Adjusted Closing Price together

df['Adj Close'].plot(label='TATAMOTRS', figsize=(16,6), title='Adjusted Closing Price vs Moving Average', color='red', linewidth=1.0, grid=True)
mvag.plot(label='MVAG', color='blue')
plt.legend()


# Return Deviation measures the Mean of the Probability Distribution of Investment Returns if it has a positive/negative Average Net Outcome

rd = close_col / close_col.shift(1) - 1
rd.plot(label='Return', figsize=(16, 6), title='Return Deviation', color='red', linewidth=1.0, grid=True)
plt.legend()

# Number of days for which to predict the stock prices

predict_days = 30
# Shifting by the Number of Predict days for Prediction array

df['Prediction'] = df['Adj Close'].shift(-predict_days)
# print(df['Prediction'])
# print(df['Adj Close'])

# Dropping the Prediction Row

X = np.array(df.drop(['Prediction'], axis = 1))
X = X[:-predict_days]      # Size upto predict days
# print(X)
print(X.shape)

# Creating the Prediction Row

y = np.array(df['Prediction'])
y = y[:-predict_days]      # Size upto predict_days
# print(y)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = RandomForestRegressor().fit(X_train, y_train)


model_score = model.score(X_train, y_train)
print('Random Model score:', model_score)


model_score = model.score(X_test, y_test)
print('Random Model score:', model_score)


predicted=model.predict(X_test)


print(X_test)
# Define the Real & Prediction Values

X_predict = np.array(df.drop(['Prediction'], 1))[-predict_days:]

model_predict_prediction = model.predict(X_predict)
model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))


predicted_dates = []
recent_date = df.index.max()
display_at = 1000
alpha = 0.5

for i in range(predict_days):
    recent_date += str(timedelta(days=1))
    predicted_dates.append(recent_date)


plt.figure(figsize=(16, 6))
plt.plot(df.index[display_at:], model_real_prediction[display_at:], label='Random Forest Prediction', color='blue', alpha=alpha)
plt.plot(predicted_dates, model_predict_prediction, label='365 days Prediction', color='green', alpha=alpha)
plt.plot(df.index[display_at:], df['Adj Close'][display_at:], label='Actual', color='red')
plt.legend()
