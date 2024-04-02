import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import pathlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import os

# 데이터 전처리
data_dir = pathlib.Path('D:\PycharmProjects\_Merged_Data')
filenames = glob.glob(str(data_dir/'*.csv')) #filenames = list
csv_list = []
for item in filenames:
    csv_list.append(pd.read_csv(item, names=["index","temp1","press1","temp2","press2","wave","gas"], index_col="index"))

def clip_data(df):
    df['temp1'] = df['temp1'].clip(0, 100) /100
    df['Press1'] = df['Press1'].clip(0, 3.25) /3.25
    df['temp2'] = df['temp2'].clip(0, 100) / 100
    df['Press2'] = df['Press2'].clip(0, 3.25) / 3.25
    df['accel'] = df['accel'].clip(0, 0.1) / 0.1
    df['gas'] = df['gas'].clip(0, 1007) / 1007
    return df

# CSV 파일을 읽어와 전처리하는 함수
def preprocess_csv(file_path):
    df = pd.read_csv(file_path, header=None, names=['index', 'temp1', 'Press1', 'temp2', 'Press2', 'accel', 'gas'])
    df = clip_data(df)
    return df

# 슬라이딩 윈도우를 사용하여 시퀀스 생성
def create_sequences(data, seq_length, pred_length):
    xs = []
    ys = []
    for i in range(len(data)-(seq_length+pred_length)-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length+pred_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 정규화
def normalization(csv_list) -> list:
    np_list = []
    for item in csv_list:
        item.loc[item["temp1"] > 100, "temp1"] = 100
        item.loc[item["temp1"] < 0, "temp1"] = 0
        item.loc[item["temp2"] > 100, "temp2"] = 100
        item.loc[item["temp2"] < 0, "temp2"] = 0
        item.loc[item["press1"] > 3.25, "press1"] = 3.25
        item.loc[item["press2"] > 3.25, "press2"] = 3.25
        item.loc[item["press1"] < 0, "press1"] = 0
        item.loc[item["press2"] < 0, "press2"] = 0
        item.loc[item["wave"] > 0.1, "wave"] = 0.1
        item.loc[item["wave"] < 0, "wave"] = 0
        item.loc[item["gas"] > 1007, "gas"] = 1007
        item.loc[item["gas"] < 0, "gas"] = 0
        numpy_array = item.values
        numpy_array[:,0] = numpy_array[:,0] / 100.0
        numpy_array[:,1] = numpy_array[:,1] / 3.25
        numpy_array[:,2] = numpy_array[:,2] / 100.0
        numpy_array[:,3] = numpy_array[:,3] / 3.25
        numpy_array[:,4] = numpy_array[:,4] / 0.1
        numpy_array[:,5] = numpy_array[:,5] / 1007.0
        np_list.append(numpy_array)
    return np_list


ds = normalization(csv_list)

#윈도우 슬라이싱
def slidingWindow(ds,window_size,step_size):
    x_data = []
    y_data = []
    for item in ds:
        for i in range(len(item)-step_size-window_size):
            x_data.append(item[i:i+window_size,:])
            #x_data 리스트 = [np.array(window_size,6),np.array(window_size,6),np.array(window_size,6)]
            y_data.append(item[i+window_size+step_size])
            #y_data 리스트 = [np.array(1),np.array(1),np.array(1),np.array(1)]


    x_data_np = np.reshape(x_data, newshape=(-1, window_size, 6))
    y_data_np = np.reshape(y_data, newshape=(-1, 6))
    return x_data_np, y_data_np


window_size= 10
step_size=1200  #30초
x_data, y_data = slidingWindow(ds, window_size, step_size)

x_training_data = x_data[:int(len(x_data)*0.7)]
y_training_data = y_data[:int(len(y_data)*0.7)]
x_test_data = x_data[int(len(x_data)*0.7):]
y_test_data = y_data[int(len(y_data)*0.7):]


input_shape=(window_size,6)
learning_rate = 0.001

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'predict_Time{step_size/10}sec.h5',
    save_best_only=True,  # 최적의 모델만 저장
    monitor='val_loss',   # 검증 손실을 기준으로 선택
    mode='min',           # 최소화
    verbose=1
)


inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.Dense(256)(x)
outputs = tf.keras.layers.Dense(6, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
loss = tf.keras.losses.mean_squared_error
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=loss, optimizer=optimizer, metrics= ['mean_squared_error'])
model.summary()




history = model.fit(x_training_data,y_training_data, epochs=10, batch_size=512, validation_data=(x_test_data,y_test_data), shuffle="True", callbacks=[model_checkpoint_callback])
# 훈련 데이터에 대한 손실과 평균 제곱 오차
plt.plot(history.epoch, history.history['loss'], 'r', label='Training Loss')
plt.plot(history.epoch, history.history['mean_squared_error'], 'b', label='Training Mean Squared Error')

# 검증 데이터에 대한 손실과 평균 제곱 오차
plt.plot(history.epoch, history.history['val_loss'], 'r--', label='Validation Loss')
plt.plot(history.epoch, history.history['val_mean_squared_error'], 'b--', label='Validation Mean Squared Error')
plt.legend()
plt.savefig(f"predict_Time{step_size/10}sec.png")
plt.show()