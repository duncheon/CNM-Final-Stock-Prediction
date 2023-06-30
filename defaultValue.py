currencyList = ["BTCUSDT", "ETHUSDT"]
defaultInterval = "1m"

intervalKey = ["1m", "5m", "30m", "1h"]
intervalVal = {
    "1m": 60000,
    "5m": 300000,
    "30m": 1800000,
    "1h": 3600000,
}


def getIntervalKey(val):
    match val:
        case 60000:
            return "1m"
        case 300000:
            return "5m"
        case 1800000:
            return "30m"
        case 3600000:
            return "1h"
    return "1m"


defaultLimit = 1000
defaultSymbol = currencyList[0]


trainCandlesSize = 1000  # 1 day
trainSplitSize = 750
testSplitSize = 250
