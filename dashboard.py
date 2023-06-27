from dash_extensions.enrich import DashProxy, html, dcc, Input, Output, State
from dash_extensions import WebSocket
import pandas as pd
import json as json
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import db as mydb
import defaultValue
import requests
from tzlocal import get_localzone

app = DashProxy(prevent_initial_callbacks=True)


def convertTimeStamp(df, localize=False, convert=False):
    if (convert):
        df['timestamp'] = df['timestamp'].astype('int64')
    df['timestamp'] = df['timestamp'].div(1000)
    if localize:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=False).dt.tz_localize(
            'UTC').dt.tz_convert(get_localzone())
    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit='s', utc=False).dt.tz_convert(get_localzone())
    return df


def predictionDf(currency, interval):
    data = {
        "currency": currency.lower(),
        "interval": interval,
    }

    predictResult = requests.get('http://localhost:3000/predict', json=data, headers={
        'Content-type': 'application/json',
        'Accept': 'application/json'
    })

    predictResult = json.loads(predictResult.text)
    data = eval(predictResult['data'])
    predictDf = pd.DataFrame(columns=['timestamp', 'Close'])

    for timestamp in data:
        predictDf.loc[len(predictDf.index)] = [timestamp, data[timestamp]]

    predictDf = convertTimeStamp(predictDf, localize=True, convert=True)

    return predictDf
# def on_message(ws, message):
#     data = message

# def on_error(ws, error):
#     print(error)

# def on_close(close_msg):
#     print(close_msg)

# def streamKline(symbol, interval):
#     socket =  f'wss://stream.binance.us:9443/ws/{symbol}@kline_{interval}'
#     ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error, on_close=on_close)
#     ws.run_forever()


# def prepareRecentData(symbol, interval, limit):
#     mydb.prepareDb(defaultValue.defaultInterval, "new")

#     db = f'candle${interval}.db'
#     df = mydb.getRecent(limit, symbol, db)

#     data = json.dumps(pd.DataFrame.to_dict(df))

#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert(get_localzone())

#     return df, data

startDf = mydb.getAll()

startDf = convertTimeStamp(startDf, localize=True)

# predictDf = predictionDf(defaultValue.defaultSymbol,
#                          defaultValue.defaultInterval)

app.layout = html.Div([
    html.H1(id="Header"),
    html.Div([
        html.Div([
            html.P(children="Coin symbol", style={
                   "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='currency',
                         options=[{'label': 'BTC-USDT', 'value': 'BTCUSDT'},
                                  {'label': 'Etherium', 'value': 'ETHUSDT'}],
                         multi=False, value='BTCUSDT', style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
        ], style={"display": "flex"}),
        html.Div([
            html.P(children="Prediction model", style={
                   "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='model',
                         options=[{'label': 'XGBoost', 'value': 'XGB'},
                                  {'label': 'RNN', 'value': 'RNN'},
                                  {'label': 'LSTM', 'value': 'LSTM'}],
                         multi=False, value='XGB', style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
        ], style={"display": "flex"}),
        html.Div([
            html.P(children="Interval", style={
                 "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='interval',
                         options=[{'label': '1m', 'value': '1m'},
                                  {'label': '5m', 'value': '5m'},
                                  {'label': '30m', 'value': '30m'},
                                  {'label': '1h', 'value': '1h'}],
                         multi=False, value='1m', style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
        ], style={"display": "flex"}),
        html.Div([
            html.P(children="Based on", style={
                   "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='based',
                         options=[{'label': 'Close', 'value': 'C'},
                                  {'label': 'ROC', 'value': 'ROC'}],
                         multi=True, value=['C', 'ROC'], style={"width": "200px", "display": "flex", "alignItems": "center", "textAlign": "right", "margin": "0 10px 0 10px"}, clearable=False)
        ], style={"display": "flex"})
    ], style={"display": "flex", "margin-left": "auto",
              "margin-right": "auto", "width": "100%", "align-items": "center", "justify-content": "center", "flexWrap": "wrap"}),
    dcc.Graph(id="candles", figure=go.Figure(data=[
        go.Candlestick(x=startDf.timestamp, open=startDf.Open,
                       high=startDf.High, low=startDf.Low, close=startDf.Close)])),
    # dcc.Store(id='graphdata', data=startData),
    # WebSocket(
    #     id="ws", url=f'wss://stream.binance.us:9443/ws/{defaultValue.defaultSymbol.lower()}@kline_{defaultValue.defaultInterval}'),
    dcc.Interval(id="intervalCounter",
                 interval=defaultValue.intervalVal[defaultValue.defaultInterval])
])


@app.callback(Output("candles", "figure", allow_duplicate=True), Input("intervalCounter", "n_intervals"), Input("intervalCounter", "interval"), Input("currency", "value"))
def update_figure(n_intervals, interval, currency):
    data_df = mydb.getAll()
    # if (message == None):
    #     raise PreventUpdate("WS update")
    # json_message = json.loads(message["data"])

    # data_df = None
    # if json_message["k"]["x"] is True:
    #     json_message["Open"] = json_message["k"]["o"]
    #     json_message["Close"] = json_message["k"]["c"]
    #     json_message["Volume"] = json_message["k"]["v"]
    #     json_message["High"] = json_message["k"]["h"]
    #     json_message["Low"] = json_message["k"]["l"]
    #     json_message["timestamp"] = json_message["k"]["T"]

    #     del json_message["k"]
    #     del json_message["E"]
    #     del json_message["e"]

    #     df = pd.DataFrame(json_message, index=[0])
    #     if type(data) is not type(None):
    #         data_dict = json.loads(data)
    #         data_df = pd.DataFrame.from_dict(data_dict)

    #         data_df = pd.concat([data_df, df], ignore_index=True)
    #         data_df.reindex()
    #         # print(data_df)
    #     else:
    #         data_df = df

    predictDf = predictionDf(currency, interval)

    display_df = data_df.copy(deep=True)
    display_df = convertTimeStamp(display_df, localize=True, convert=False)

    candles = go.Figure(data=[
        go.Candlestick(x=display_df.timestamp, open=display_df.Open,
                       high=display_df.High, low=display_df.Low, close=display_df.Close),
        go.Scatter(x=predictDf.timestamp, y=predictDf.Close)])
    candles.update_layout(uirevision="Don't change")

    return candles


@app.callback(Output("candles", "figure", allow_duplicate=True), Output("intervalCounter", "interval"), Input("currency", "value"), Input("interval", "value"))
def update_firgure(currency, interval):
    data = {
        "currency": currency.lower(),
        "interval": interval,
    }

    updateSocket = requests.post('http://localhost:5000/updateSocket', json=data, headers={
        'Content-type': 'application/json',
        'Accept': 'application/json'
    })

    data_df = mydb.getAll()

    display_df = data_df.copy(deep=True)
    display_df = convertTimeStamp(display_df, localize=True, convert=False)

    candles = go.Figure(data=[
        go.Candlestick(x=display_df.timestamp, open=display_df.Open,
                       high=display_df.High, low=display_df.Low, close=display_df.Close),
    ])
    candles.update_layout(uirevision="Don't change")

    return candles, defaultValue.intervalVal[f"{interval}"]


if __name__ == '__main__':
    app.run_server(debug=True)
