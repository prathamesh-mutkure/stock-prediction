# Importing all the required libraries
import matplotlib
matplotlib.use('Agg')
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import mpld3


plt.rcParams["figure.figsize"] = [16, 6]
app = Flask(__name__)


def generate_array(n):
    return list(range(1, n+1))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.form["symbol"]
    start = str(date.today() - relativedelta(years=10))
    end = str(date.today())
    df = yf.download(symbol, start=start, end=end)

    # Rolling Mean / Moving Average to remove the noise in the graph and smoothen it
    # close_col = df["Adj Close"]
    # mvag = close_col.rolling(window=100).mean()  # Taking an average over the window size of 100.

    # Increasing the window size can make it more smoother, but less informative and vice-versa.
    predict_days = int(request.form["predict_days"])

    # Shifting by the Number of Predict days for Prediction array
    df["Prediction"] = df["Adj Close"].shift(-predict_days)

    # Dropping the Prediction Row
    X = np.array(df.drop(["Prediction"], axis=1))
    X = X[:-predict_days]  # Size upto predict days

    # Creating the Prediction Row
    y = np.array(df["Prediction"])
    y = y[:-predict_days]  # Size upto predict_days

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = int(request.form["model_name"])
    if model_choice == 1:
        model = RandomForestRegressor().fit(X_train, y_train)
    elif model_choice == 2:
        model = SVR().fit(X_train, y_train)
    elif model_choice == 3:
        model = LinearRegression().fit(X_train, y_train)
    elif model_choice == 4:
        model = DecisionTreeRegressor().fit(X_train, y_train)

    # model_score = model.score(X_train, y_train)
    model_score = model.score(X_test, y_test)
    # predicted = model.predict(X_test)

    # Define the Real & Prediction Values
    X_predict = np.array(df.drop(["Prediction"], axis=1))[-predict_days:]

    model_predict_prediction = model.predict(X_predict)
    model_predict_prediction = list(np.around(np.array(model_predict_prediction),2))
    model_real_prediction = model.predict(np.array(df.drop(["Prediction"], axis=1)))
    predicted_dates = []
    recent_date = df.index.max()
    display_at = 1000
    alpha = 0.5

    for i in range(predict_days):
        recent_date += timedelta(days=1)
        predicted_dates.append(recent_date)

    fig = plt.figure(figsize=(16, 6))
    plt.title(symbol)
    plt.plot(df.index[display_at:], model_real_prediction[display_at:], label='Model Prediction', color='blue', alpha=alpha)
    plt.plot(predicted_dates, model_predict_prediction, label='{} days Prediction'.format(predict_days), color='green', alpha=alpha)
    plt.plot(df.index[display_at:], df['Adj Close'][display_at:], label='Actual', color='red')
    plt.legend()
    plt.show()
    html_str = mpld3.fig_to_html(fig)
    Html_file= open("templates/plot.html","w")
    Html_file.write(html_str)
    Html_file.close()
    plt.savefig('static/images/new_plot.png')

    return render_template("result.html", model_score="Model Testing Accuracy: {:.2f}".format(model_score), prediction_text="Predicted Stock Prices for {}".format(symbol), arr=model_predict_prediction, days=generate_array(predict_days))

@app.route("/plot_image", methods=["POST"])
def plot_image():
    # return render_template("plot_image.html", url='/static/images/new_plot.png')
    return render_template("plot.html")

if __name__ == "__main__":
    app.run(debug=True)
