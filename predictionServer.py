import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import db
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, json, request, jsonify
import defaultValue


scaler = MinMaxScaler(feature_range=(0, 1))


def LSTMModel(symbol):
    df = pd.DataFrame.from_dict(db.fetchFirstCandles(symbol, '1m', 1000))

    plt.figure(figsize=(16, 8))
    plt.plot(df["Close"], label='Close Price history')

    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(
        0, len(df)), columns=['timestamp', 'Close'])

    for i in range(0, len(data)):
        new_dataset["timestamp"][i] = data['timestamp'][i]
        new_dataset["Close"][i] = data["Close"][i]

    new_dataset.index = new_dataset.timestamp
    new_dataset.drop("timestamp", axis=1, inplace=True)

    final_dataset = new_dataset.values

    train_data = final_dataset[0:1000, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data,
                   epochs=5, batch_size=1, verbose=2)

    lstm_model.save("lstm_" + symbol + ".h5")

    # plt.plot(valid_data[['Close', "Predictions"]])


def prediction(df, model, interval):
    df.index = df['timestamp']
    new_data = pd.DataFrame(index=range(0, len(df)),
                            columns=['timestamp', 'Close'])

    for i in range(0, len(df)):
        new_data['timestamp'][i] = df['timestamp'].iloc[i]
        new_data['Close'][i] = df['Close'].iloc[i]

    lastTimestamp = new_data["timestamp"].iloc[-1]
    for x in range(1, 11):
        new_data.loc[len(new_data.index)] = [lastTimestamp + x * interval, 0]

    new_data.index = new_data.timestamp
    new_data.drop("timestamp", axis=1, inplace=True)
    dataset = new_data.iloc[:1000].values
    # train = dataset[0:1000, :]
    # valid = dataset[1000:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # x_train, y_train = [], []

    # for i in range(60, len(train)):
    #     x_train.append(scaled_data[i-60:i, 0])
    #     y_train.append(scaled_data[i, 0])

    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    loaded_model = load_model(model)

    inputs = new_data[len(new_data)-10-60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    closing_price = loaded_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    # train = new_data[:1000]
    valid = new_data[1000:]

    valid['Predictions'] = closing_price
    print(valid)
    return valid


# LSTMModel('btcusdt')

# print(valid)
# predictionModel("./BTC-USD.csv", "btc")
# predictionModel("./ETH-USD.csv", "eth")

api = Flask(__name__)


def prepareAllModel():
    print("Model prep")
    LSTMModel('btcusdt')


@api.route('/predict', methods=["GET"])
def update_socket():
    body = request.get_json()
    inputInterval = body["interval"]
    currency = body["currency"]

    if (inputInterval and currency):
        data = db.getRecent(1000)
        valid = prediction(data, 'lstm_btcusdt.h5', inputInterval)
        returnData = valid['Predictions'].to_json()
        return jsonify({
            'msg': f'Prediction returned',
            'data': returnData,
            'status': 200
        })
    else:
        return jsonify({
            'msg': f'Error',
            'status': 400
        })


if __name__ == '__main__':
    # prepareAllModel()
    api.run(port=3000)
