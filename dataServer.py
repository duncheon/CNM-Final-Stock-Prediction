from flask import Flask, json, request, jsonify
import websocket
import threading
import atexit
import db
import defaultValue

currency = defaultValue.defaultSymbol
interval = defaultValue.defaultInterval

# create a restart flag
streamingFlag = False
isKilled = False


db.prepareDb(currency, interval, defaultValue.defaultLimit, "new")


def on_error(ws, error):
    print(error)


def on_close(close_msg):
    print("### closed ###" + close_msg)


def on_message(ws, message):

    global streamingFlag
    if streamingFlag == True:
        ws.close()
        streamingFlag = False
    else:

        json_message = json.loads(message)

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

            db.insertRow(json_message)


def streaming_thread():
    while True:
        if (isKilled):
            break
        try:
            websocket.enableTrace(False)
            socket = f'wss://stream.binance.com:9443/ws/{currency.lower()}@kline_{interval}'
            ws = websocket.WebSocketApp(socket,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
            print(f"{currency} websocket feed started.Interval {interval}.")
            ws.run_forever()
        except Exception as e:
            print("Hi")


thread = threading.Thread(target=streaming_thread, args=(), name="streaming")
thread.daemon = True
thread.start()


# def updateAPI(newCurrency, newInterval):
#     global currency
#     global interval
#     currency = newCurrency
#     interval = newInterval


# def changeSocket():
#     global ws
#     global currency
#     global interval
#     ws.close()
#     ws = streamKline(currency, interval)
#     print(ws)


# def startSocket():
#     global ws
#     ws.run_forever()


# socket_process = multiprocessing.Process(target=startSocket)


api = Flask(__name__)


@atexit.register
def killThread():
    global isKilled
    isKilled = True
    print("exit flask")


def start_flask_app():
    api.run()


@api.route('/updateSocket', methods=["POST"])
def update_socket():
    body = request.get_json()
    print(body)
    global currency
    global interval
    global streamingFlag

    if (currency == body["currency"] and interval == body["interval"]) is False:
        print(currency)
        currency = body["currency"]
        interval = body["interval"]

        db.prepareDb(currency, interval, 1000, "new")
        streamingFlag = True

        return jsonify({
            'msg': f'Success changed to {currency} + {interval}',
            'status': 201
        })
    else:
        return jsonify({
            'msg': f'Unchanged',
            'status': 201
        })


if __name__ == '__main__':
    api.run()
    thread.start()
