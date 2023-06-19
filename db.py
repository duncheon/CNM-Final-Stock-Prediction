import sqlite3
import requests


def fetchFirstCandles(symbol):
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


def createDb():
    conn = sqlite3.connect('canlde.db')
    print('opened db successfully')

    conn.execute('''CREATE TABLE CANDLE
         (ID INT PRIMARY KEY     AUTOINCREMENT,
         OPEN           TEXT    NOT NULL,
         HIGH           TEXT    NOT NULL,
         LOW           TEXT    NOT NULL,
         CLOSE          TEXT    NOT NULL,
         VOLUME         TEXT    NOT NULL,
         S              TEXT    NOT NULL,
         TIMESTAMP      INT    NOT NULL);''')

    print("Table created successfully")

    conn.close()


def insertRow(item):
    conn = sqlite3.connect('candle.db')

    insertQuery = f"INSERT INTO CANDLE (OPEN,HIGH,LOW,CLOSE,VOLUME,S,TIMESTAMP) \
        VALUES ({item['Open']},{item['High']},{item['Low']},{item['Close']},{item['Volume']},{item['s']},{item['timestamp']})"

    conn.execute(insertQuery)
    conn.commit()
    conn.close()


createDb()
