from dash_extensions.enrich import DashProxy, html, dcc, Input, Output, State, Dash
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
import dash_bootstrap_components as dbc

app = DashProxy(prevent_initial_callbacks=True,
                external_stylesheets=[dbc.themes.CYBORG])


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


def predictionDf(currency, interval, model):
    data = {
        "currency": currency.lower(),
        "interval": interval,
        "model": model
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

def serve_layout():

    infoReq = requests.get('http://localhost:5000/getServerInfo')

    info = json.loads(infoReq.text)
    info = info['data']

    defaultSym = info['currency']
    defaultInterval = info['interval']
    defaultModel = info['model']

    startDf = mydb.getAll()

    startDf = convertTimeStamp(startDf, localize=True)

    predictDf = predictionDf(defaultValue.defaultSymbol,
                             defaultValue.intervalVal[defaultValue.defaultInterval], "XGBoost")
# predictDf = predictionDf(defaultValue.defaultSymbol,
#                          defaultValue.defaultInterval)
    return html.Div([
        html.H1(id="Header"),
        html.Div([
            html.Div([
                html.P(children="Coin symbol", style={
                    "margin": 0}),
                dcc.Dropdown(id='currency',
                             options=[{'label': 'BTC-USDT', 'value': 'BTCUSDT'},
                                      {'label': 'Etherium', 'value': 'ETHUSDT'}],
                             multi=False, value=defaultSym.upper(), style={"width": "200px", "display": "flex", "margin": "0 10px 0 10px", "jusifyContent": "center"}, clearable=False)
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            html.Div([
                html.P(children="Model", style={
                    "margin": 0}),
                dcc.Dropdown(id='model',
                             options=[{'label': 'XGBoost', 'value': 'XGBoost'},
                                      {'label': 'RNN', 'value': 'RNN'},
                                      {'label': 'LSTM', 'value': 'LSTM'}],
                             multi=False, value=defaultModel, style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            html.Div([
                html.P(children="Interval", style={
                    "margin": 0}),
                dcc.Dropdown(id='interval',
                             options=[{'label': '1m', 'value': '1m'},
                                      {'label': '5m', 'value': '5m'},
                                      {'label': '30m', 'value': '30m'},
                                      {'label': '1h', 'value': '1h'}],
                             multi=False, value=defaultInterval, style={"width": "200px", "display": "flex", "alignItems": "center", "margin": "0 10px 0 10px"}, clearable=False)
            ], style={"display": "flex", "alignItems": "center",  "marginBottom": "10px"}),
            html.Div([
                html.P(children="Target", style={
                    "margin": 0}),
                dcc.Dropdown(id='based',
                             options=[{'label': 'Close', 'value': 'C'},
                                      {'label': 'ROC', 'value': 'ROC'}],
                             multi=True, value=['C', 'ROC'], style={"width": "200px", "display": "flex", "alignItems": "center", "textAlign": "right", "margin": "0 10px 0 10px"}, clearable=False)
            ], style={"display": "flex", "alignItems": "center",  "marginBottom": "10px"})
        ], style={"display": "flex", "margin": "auto", "width": "100%", "alignItems": "center", "justifyContent": "space-evenly", "flexWrap": "wrap", "marginBottom": "20px"}),
        dcc.Graph(id="candles", figure=go.Figure(data=[
            go.Candlestick(x=startDf.timestamp, open=startDf.Open,
                           high=startDf.High, low=startDf.Low, close=startDf.Close, name="Readtime data"),
            go.Scatter(x=predictDf.timestamp, y=predictDf.Close, name="Model prediction")], layout=dict(template='plotly_dark')
        )),
        # dcc.Store(id='graphdata', data=startData),
        # WebSocket(
        #     id="ws", url=f'wss://stream.binance.us:9443/ws/{defaultValue.defaultSymbol.lower()}@kline_{defaultValue.defaultInterval}'),
        dcc.Interval(id="intervalCounter",
                     interval=defaultValue.intervalVal[defaultInterval])
    ])


@app.callback(Output("candles", "figure", allow_duplicate=True), Input("intervalCounter", "n_intervals"), Input("intervalCounter", "interval"), Input("currency", "value"), Input('model', 'value'))
def update_figure(n_intervals, interval, currency, model):
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

    predictDf = predictionDf(currency, interval, model)

    display_df = data_df.copy(deep=True)
    display_df = convertTimeStamp(display_df, localize=True, convert=False)

    candles = go.Figure(data=[
        go.Candlestick(x=display_df.timestamp, open=display_df.Open,
                       high=display_df.High, low=display_df.Low, close=display_df.Close, name="Realtime data"),
        go.Scatter(x=predictDf.timestamp, y=predictDf.Close, name="Model prediction")])
    candles.update_layout(uirevision="Don't change", template="plotly_dark")

    return candles


@app.callback(Output("intervalCounter", "interval"), Input("currency", "value"), Input("interval", "value"), Input('model', 'value'))
def update_firgure(currency, interval, model):
    data = {
        "currency": currency.lower(),
        "interval": interval,
        "model": model
    }

    updateSocket = requests.post('http://localhost:5000/updateSocket', json=data, headers={
        'Content-type': 'application/json',
        'Accept': 'application/json'
    })

    data_df = mydb.getAll()

    display_df = data_df.copy(deep=True)
    display_df = convertTimeStamp(display_df, localize=True, convert=False)

    # candles = go.Figure(data=[
    #     go.Candlestick(x=display_df.timestamp, open=display_df.Open,
    #                    high=display_df.High, low=display_df.Low, close=display_df.Close),
    # ])
    # candles.update_layout(uirevision="Don't change", template="plotly_dark")

    return defaultValue.intervalVal[f"{interval}"]


app.layout = serve_layout()
if __name__ == '__main__':
    app.run_server(debug=True)
