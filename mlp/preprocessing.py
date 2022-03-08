
import pandas as pd
import numpy as np


class Processing:

    # 训练数据集规模应大于100
    def __init__(self, df, divide):
        data = df[['close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount', 'trade_date']]   # 输入属性包括开盘价、收盘价、最高价、最低价、涨跌幅、成交量、成交额
        # label = df[['open']]                            # 预测明天的开盘价
        self.len_train = int(divide * len(data))
        self.train_data = data[0: self.len_train]       # 训练集
        self.test_data = data[self.len_train: ]
        # print('-----------train----------')
        # print(self.train_data)
        # print('--------------test------------')
        # print(self.test_data)
        self.input_train = []
        self.output_train = []
        self.input_test = []
        self.output_test = []
        self.time = []

    def get_train_data(self, cycle):
        for i in range(len(self.train_data)-cycle-1):                        # 取10天为一周期进行训练，以后一天的开盘价为标签
            x = np.array([self.train_data.iloc[i: i + cycle,1]], np.float)  # 取下标为1的列为标签
            y = np.array([self.train_data.iloc[i + cycle + 1, 1]], np.float)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train)
        self.Y_train = np.array(self.output_train)

    def get_test_data(self, cycle):
        for i in range(len(self.test_data)-cycle-1):
            x = np.array([self.test_data.iloc[i: i + cycle, 1]], np.float)
            y = np.array([self.test_data.iloc[i + cycle + 1, 1]], np.float)
            t = np.array([self.train_data.iloc[i + cycle + 1, 7]])
            self.input_test.append(x)
            self.output_test.append(y)
            self.time.append(t)
        self.X_test = np.array(self.input_test)
        self.Y_test = np.array(self.output_test)
        self.time = np.array(self.time)
