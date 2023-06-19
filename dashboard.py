from dash_extensions.enrich import DashProxy, html, dcc, Input, Output, State
from dash_extensions import WebSocket
import pandas as pd
import json as json
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import requests
import sqlite3

app = DashProxy(prevent_initial_callbacks=True)

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

symbol = "BTCUSDT"
interval = "1m"
limit = 1000


def prepareHistoryData(symbol, interval, limit):
    api = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    print(api)
    data = requests.get(api)

    data = data.json()
    formatedData = []

    for candle in data:
        candleObj = {}
        candleObj["Open"] = candle[1]
        candleObj["High"] = candle[2]
        candleObj["Low"] = candle[3]
        candleObj["Close"] = candle[4]
        candleObj["Volume"] = candle[5]
        candleObj["timestamp"] = candle[6]
        candleObj["s"] = symbol
        formatedData.append(candleObj)

    df = pd.DataFrame(formatedData)
    return df


startDf = prepareHistoryData(
    symbol, interval, limit)

startData = json.dumps(pd.DataFrame.to_dict(startDf))

startDf["timestamp"] = pd.to_datetime(
    startDf["timestamp"], unit='ms', utc=False)


app.layout = html.Div([
    html.H1(id="Header"),
    html.Div([
        html.Div([
            html.P(children="Coin symbol", style={
                   "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='currency',
                         options=[{'label': 'BTC-USDT', 'value': 'BTC'},
                                  {'label': 'Etherium', 'value': 'ETH'},
                                  {'label': 'Cardano', 'value': 'ADA'}],
                         multi=False, value='BTC', style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
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
            html.P(children="Based on", style={
                   "marginLeft": "20px", "width": "70px"}),
            dcc.Dropdown(id='based',
                         options=[{'label': 'Close', 'value': 'C'},
                                  {'label': 'ROC', 'value': 'ROC'}],
                         multi=True, value=['C', 'ROC'], style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
        ], style={"display": "flex"})
    ], style={"display": "flex", "margin-left": "auto",
              "margin-right": "auto", "width": "100%", "align-items": "center", "justify-content": "center", "flexWrap": "wrap"}),
    dcc.Graph(id="candles", figure=go.Figure(data=[
        go.Candlestick(x=startDf.timestamp, open=startDf.Open, high=startDf.High, low=startDf.Low, close=startDf.Close)])),
    dcc.Store(id='graphdata', data=startData),
    WebSocket(
        id="ws", url=f'wss://stream.binance.us:9443/ws/{symbol.lower()}@kline_{interval}')
])


@app.callback(Output("candles", "figure"), Output("graphdata", "data"), Input("ws", "message"), Input("graphdata", "data"))
def update_figure(message, data):
    json_message = json.loads(message["data"])

    data_df = None
    if json_message["k"]["x"] is True:
        json_message["Open"] = json_message["k"]["o"]
        json_message["Close"] = json_message["k"]["c"]
        json_message["Volume"] = json_message["k"]["v"]
        json_message["High"] = json_message["k"]["h"]
        json_message["Low"] = json_message["k"]["l"]
        json_message["timestamp"] = json_message["k"]["T"]

        del json_message["k"]
        del json_message["E"]
        del json_message["e"]

        df = pd.DataFrame(json_message, index=[0])
        if type(data) is not type(None):
            data_dict = json.loads(data)
            data_df = pd.DataFrame.from_dict(data_dict)

            data_df = pd.concat([data_df, df], ignore_index=True)
            data_df.reindex()
            print("second")
            print(data_df)
        else:
            data_df = df

        display_df = data_df.copy(deep=True)
        display_df["timestamp"] = pd.to_datetime(
            display_df["timestamp"], unit='ms', utc=False)

        candles = go.Figure(data=[
            go.Candlestick(x=display_df.timestamp, open=display_df.Open, high=display_df.High, low=display_df.Low, close=display_df.Close)])
        candles.update_layout(uirevision="Don't change")
        return candles, json.dumps(pd.DataFrame.to_dict(data_df))
    else:
        raise PreventUpdate("No update")


if __name__ == '__main__':
    app.run_server(debug=True)
