import sqlite3
import requests
import pandas as pd
import os
import defaultValue


def fetchFirstCandles(symbol, interval, limit):
    api = f'https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}'
    data = requests.get(api)
    data = data.json()
    formatedData = []

    print(data)
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
    return formatedData


def createDb():
    conn = sqlite3.connect('candle.db')
    print('opened db successfully')

    conn.execute('''CREATE TABLE CANDLE
         (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
         Open           TEXT    NOT NULL,
         High           TEXT    NOT NULL,
         Low           TEXT    NOT NULL,
         Close          TEXT    NOT NULL,
         Volume         TEXT    NOT NULL,
         s              TEXT    NOT NULL,
         timestamp      INT    NOT NULL);''')

    print("Table created successfully")

    conn.close()


def insertRow(item):
    conn = sqlite3.connect('./candle.db')
    insertQuery = f"INSERT INTO CANDLE (Open,High,Low,Close,Volume,s,timestamp) \
        VALUES (?,?,?,?,?,?,?)"

    conn.execute(insertQuery, (item['Open'], item['High'], item['Low'],
                 item['Close'], item['Volume'], item['s'], item['timestamp']))
    conn.commit()
    conn.close()


def getRecent(total):
    conn = sqlite3.connect('./candle.db')

    df = pd.read_sql_query(
        f'SELECT * FROM (SELECT * FROM CANDLE ORDER BY id DESC LIMIT {total}) ORDER BY id ASC', conn)

    conn.close()
    return df


def getAll():
    conn = sqlite3.connect('./candle.db')

    df = pd.read_sql_query(
        f'SELECT * FROM CANDLE', conn)
    conn.close()

    return df


def prepareDb(currency, interval, limit=1000, type="new"):
    if type == "new":
        if os.path.exists(f'./candle.db'):
            conn = sqlite3.connect('./candle.db')
            conn.execute('DELETE FROM CANDLE')
            conn.commit()
            conn.close()
        else:
            createDb()

    dataAsList = fetchFirstCandles(
        currency, interval, limit)

    for item in dataAsList:
        insertRow(item)


# createDb()
# formatedData, df = fetchCandlesData('BTCUSDT', '1m', 1000)
# for item in formatedData:
#     insertRow(item)
