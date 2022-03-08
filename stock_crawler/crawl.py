import tushare as ts
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.optimizers import adam_v2


inputFile = ''  # excel输入
outputFile = ''  # excel输出
modelFile = ''  # 网络权重
feature = ['close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount']  # 输入的特征

class Crawl:



    def __init__(self, df):
        pass

    def access(self):
        ts.set_token('752ba2cd8f48df61d08385e43ba808cd0e6d0e8a8ceca52f86648f67')                # 设置token
        pro = ts.pro_api()                                                                      # 初始化pro接口
        df = pro.index_daily(ts_code='399001.SZ', start_date='20210101', end_date='20210320')   # 返回深证成份指数数据,df为DataFrame类型（表格）数据
        data = df[['close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount']]                 # 以前一日股票的开盘价、收盘价、最低价、最高价、成交量、成交额和涨跌幅为输入，以后一日的开盘价open为输出
        label = df[['open']]
        print(data)
        len_train = int(0.9 * len(data))
        train_data = data[0: len_train]       # 训练集
        label_data = data[len_train:]
        test_data = []                        # 测试集
        test_label_data = []
        return train_data, label_data, test_data, test_label_data

    # 数据预处理
    def pretreatment(self, train_data, label_data, test_data, test_label_data):
        min_data = train_data.min()
        max_data = train_data.max()
        min_label_data = label_data.min()
        max_label_data = label_data.max()
        min_test_data = test_data.min()
        max_test_data = test_data.max()
        min_test_lable_data = test_label_data.min()
        max_test_label_data = test_label_data.max()
        # 数据归一化(为什么标准化之后的范围不是在1与-1之间？)
        train_data = (train_data - min_data)/(max_data - min_data)
        label_data = (label_data - min_label_data)/(max_label_data - min_label_data)
        test_data = (test_data - min_test_data)/(max_test_data - min_test_data)
        test_label_data = (test_label_data - min_test_lable_data)/(max_test_label_data - min_test_lable_data)
        x_train = train_data.values
        y_train = label_data.values
        x_test = test_data.values
        y_test = test_label_data.values
        return x_train, y_train, x_test, y_test

    # 构建网络模型
    def network(self, x_train, y_train, x_test, y_test):
        model = Sequential()
        model.add(Dense(10, input_dim=7, activation='sigmoid'))
        model.add(Dense(5, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer=adam_v2.Adam(lr=0.1), metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20, batch_size=5, verbose=2)
        print(model.summary())
        result = model.predict(x_test)
        # 误差和准确率
        print('-------------result--------------')
        print(result)
        print('-------------y_test--------------')
        print(y_test)





if __name__ == '__main__':

    crawl = Crawl()
    (train_data, label_data, test_data, test_label_data) = crawl.access()
    (x_train, y_train, x_test, y_test) = crawl.pretreatment(train_data, label_data, test_data, test_label_data)
    # print('-------------x_train----------------')
    # print(x_train)
    # print('-------------y_train----------------')
    # print(y_train)
    crawl.network(x_train, y_train, x_test, y_test)

