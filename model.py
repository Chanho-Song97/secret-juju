import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
#from tensorflow.python.keras.datasets import mnist
import pandas as pd
import numpy as np
import seaborn as sns
from pykrx import stock
import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmax



# 삼성전자
def price():
    data = stock.get_market_ohlcv_by_date("20050101","20221128","005930")
    #print(data)

    raw_df = pd.DataFrame(data).copy()

    def MinMaxScaler(data):
        """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    dfx = raw_df[['시가','고가','저가','거래량', '종가']]
    dfx = MinMaxScaler(dfx)
    dfy = dfx[['종가']]

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    data_x = []
    data_y = []
    window_size = 10

    for i in range(len(y) - window_size):
        _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
        _y = y[i + window_size]     # 다음 날 종가
        data_x.append(_x)
        data_y.append(_y)
    print(_x, "->", _y)

    train_size = int(len(data_y) * 0.7)
    train_x = np.array(data_x[0 : train_size])
    train_y = np.array(data_y[0 : train_size])

    test_size = len(data_y) - train_size
    test_x = np.array(data_x[train_size : len(data_x)])
    test_y = np.array(data_y[train_size : len(data_y)])

    # 모델 생성
    model = Sequential()
    model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, 5)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=60, batch_size=30)
    pred_y = model.predict(test_x)

    # raw_df.close[-1] : dfy.close[-1] = x : pred_y[-1]
    samsung = raw_df.종가[-1] * pred_y[-1] / dfy.종가[-1]
    print("Tomorrow's SEC price :", raw_df.종가[-1] * pred_y[-1] / dfy.종가[-1])
    return samsung