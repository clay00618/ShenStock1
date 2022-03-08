import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
import tensorflow.python,keras.layers
from tensorflow.python.keras.optimizers import adam_v2
import mlp.get_data as source
import mlp.preprocessing as deal
from matplotlib import pyplot as plt


ts_code = '399001.SZ'    # 股票代码
start_date = '20000320'  # 开始日期
end_date = '20210320'    # 结束日期
divide = 0.8             # 数据划分
cycle = 10               # 预测周期


# 数据获取和预处理
df = source.Get_data(start_date, end_date, ts_code).df
process = deal.Processing(df, divide)
process.get_train_data(cycle)
process.get_test_data(cycle)

X_train = process.X_train
Y_train = process.Y_train
X_test = process.X_test
Y_test = process.Y_test

print(X_train.shape)
print(Y_train.shape)

# 训练集标准化
x_mean = X_train.mean(axis=0)
X_train -= x_mean
x_std = X_train.std(axis=0)
X_train /= x_std

y_mean = Y_train.mean(axis=0)
Y_train -= y_mean
y_std = Y_train.std(axis=0)
Y_train /= y_std

# 测试集标准化
x_test_mean = X_test.mean(axis=0)
X_test -= x_test_mean
x_test_std = X_test.std(axis=0)
X_test /= x_test_std

y_test_mean = Y_test.mean(axis=0)
Y_test -= y_test_mean
y_test_std = Y_test.std(axis=0)
Y_test /= y_test_std
date = process.time

# 绘制测试集数据图
# fig = plt.figure()
# ax = fig.add_subplot(211)
# plt.plot(date[0:8, 0], Y_test[0:8])
#
# plt.subplot(212)
# plt.plot(date[0:8, 0], process.Y_test[0:8])
#
# plt.show()


# 建立多层感知机模型
model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=adam_v2.Adam(lr=0.01), loss='mse', metrics=['mae'])
model.fit(X_train, Y_train, epochs=100, verbose=2, batch_size=10)
print(model.evaluate(X_test, Y_test))
print(model.summary())


result = model.predict(X_test)
for i in range(len(Y_test)):
    print('X=%s, prediceted=%s' % (Y_test[i], result[i]))


fig = plt.figure()

plt.plot(range(0, 40), result[0: 40, 0], color='green', label='predict_open')   # 绿色为预测值
plt.plot(range(0, 40), Y_test[0: 40, 0], color='red', label='real_open')   # 红色为真实值
plt.xlabel('index')
plt.ylabel('open')
plt.title('twenty-years')
plt.show()









