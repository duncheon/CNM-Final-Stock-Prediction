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
import xgboostModel
import LSTMModel
import RNNModel

scaler = MinMaxScaler(feature_range=(0, 1))


# LSTMModel('btcusdt')

# print(valid)
# predictionModel("./BTC-USD.csv", "btc")
# predictionModel("./ETH-USD.csv", "eth")

api = Flask(__name__)


def prepareAllModel():
    print("Model prep")
    for symbol in defaultValue.currencyList:
        for interval in defaultValue.intervalKey:
            RNNModel.RNNmodel(symbol, interval)
    for symbol in defaultValue.currencyList:
        for interval in defaultValue.intervalKey:
            LSTMModel.LSTMModel(symbol, interval)
    # for symbol in defaultValue.currencyList:
    #     for interval in defaultValue.intervalKey:
    #         xgboostModel.XGBoost(symbol, interval)


@api.route('/predict', methods=["GET"])
def update_socket():
    body = request.get_json()
    model = body["model"]
    inputInterval = body["interval"]
    currency = body["currency"]
    total = 500
    # total = body["total"]

    if (inputInterval and currency):
        data = db.getRecent(1000)

        valid = []
        if model == 'XGBoost':
            valid = xgboostModel.XGBoost(currency, inputInterval, total)
            # print(valid['Predictions'])
        elif model == 'LSTM':
            # LSTMModel.LSTMModel(currency)
            valid = LSTMModel.prediction(currency, inputInterval, total)
            # print(valid['Predictions'])
        else:
            # RNNModel.RNNmodel(currency)
            valid = RNNModel.prediction(currency, inputInterval, total)
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
    prepareAllModel()
    api.run(port=3000)
