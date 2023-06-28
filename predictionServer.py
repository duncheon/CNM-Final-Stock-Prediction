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

scaler = MinMaxScaler(feature_range=(0, 1))


# LSTMModel('btcusdt')

# print(valid)
# predictionModel("./BTC-USD.csv", "btc")
# predictionModel("./ETH-USD.csv", "eth")

api = Flask(__name__)


def prepareAllModel():
    print("Model prep")
    # LSTMModel('btcusdt')


@api.route('/predict', methods=["GET"])
def update_socket():
    body = request.get_json()
    model = body["model"]
    inputInterval = body["interval"]
    currency = body["currency"]
    total = body["total"]

    if (inputInterval and currency):
        data = db.getRecent(1000)

        valid = []
        if model is 'XGBoost':
            # valid = xgboostModel.XGBoost(currency, inputInterval, 10)
            valid = []
        else:
            # valid = prediction(data, 'lstm_btcusdt.h5', inputInterval)
            valid = []
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
