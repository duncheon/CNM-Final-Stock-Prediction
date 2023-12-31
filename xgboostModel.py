import os
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import defaultValue
import db


def createFeature(df):
    df['hour'] = df.timestamp.dt.hour
    df['dayofweek'] = df.timestamp.dt.dayofweek
    df['quarter'] = df.timestamp.dt.quarter
    df['month'] = df.timestamp.dt.month
    df['year'] = df.timestamp.dt.year
    df['dayofyear'] = df.timestamp.dt.dayofyear
    df['second'] = df.timestamp.dt.second
    df['minute'] = df.timestamp.dt.minute
    return df


def XGBoost(symbol, interval, total):
    df = pd.DataFrame.from_dict(db.fetchFirstCandles(
        symbol, defaultValue.getIntervalKey(interval), defaultValue.trainCandlesSize))
    # df = db.getRecent(1000)
    # df = df.drop('ID', axis=1)

    lastTimestamp = df["timestamp"].iloc[-1]
    for x in range(1, total + 1):
        df.loc[len(df.index)] = [0, 0, 0, 0, 0,
                                 lastTimestamp + x * interval, symbol.upper()]

    df.index = df['timestamp']
    df['timestamp'] = df['timestamp'].astype('int64')
    df['timestamp'] = df['timestamp'].div(1000)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s", utc=False)

    # df.index = df.timestamp

    train = df[0:defaultValue.trainSplitSize - 1]
    test = df[defaultValue.trainSplitSize:defaultValue.trainCandlesSize - 1]
    pred = df[defaultValue.trainCandlesSize:]

    pred = createFeature(pred)
    train = createFeature(train)
    test = createFeature(test)

    features = ['hour', 'dayofweek', 'quarter', 'month',
                'year', 'dayofyear', 'second', 'minute']

    print(pred)
    X_train = train[features]
    y_train = train['Close']

    X_test = test[features]
    y_test = test['Close']

    x_pred = pred[features]
    reg = xgb.XGBRegressor(
        n_estimators=1000, early_stopping_rounds=10, learning_rate=0.7)
    reg.fit(X_train, y_train, eval_set=[
            (X_train, y_train), (X_test, y_test)], verbose=True)

    x_pred['Predictions'] = reg.predict(x_pred)

    return x_pred


# LSTMModel('btcusdt')

# print(valid)
# predictionModel("./BTC-USD.csv", "btc")
# predictionModel("./ETH-USD.csv", "eth")
# XGBoost('btcusdt', 60000, 100)
