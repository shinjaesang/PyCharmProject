import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

# csv 파일이 저장된 폴더 경로를 지정합니다.
data_dir = pathlib.Path('./PycharmProjects/_Merged_Data')

# 지정된 폴더 내의 모든 csv 파일의 경로를 가져옵니다.
filenames = glob.glob(str(data_dir/'*.csv'))

# 각 csv 파일을 읽어서 DataFrame으로 변환하고, 이를 리스트에 저장합니다.
dataframes = []
for filename in filenames:
    df = pd.read_csv(filename)
    dataframes.append(df)

# 모든 DataFrame을 하나로 합칩니다.
# 이때, ignore_index=True 옵션을 사용하여 인덱스를 재설정할 수 있습니다.
train_label = pd.concat(dataframes, ignore_index=True)

# 데이터 정규화
train_label = (train_label - train_label.min()) / (train_label.max() - train_label.min())

seq_length = 10  # 시퀀스 길이 정의

def create_sequences(dataset: tf.data.Dataset, seq_length: int) -> tf.data.Dataset:
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels = sequences[-1]
        return inputs, labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

notes_ds = tf.data.Dataset.from_tensor_slices(train_label.values)
seq_ds = create_sequences(notes_ds, seq_length)

# 모델 구성
input_shape = (seq_length, train_label.shape[1])
learning_rate = 0.01

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)
outputs = tf.keras.layers.Dense(train_label.shape[1])(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate))
model.summary()

# 학습
epochs = 50
history = model.fit(seq_ds.batch(64), epochs=epochs)

plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()

def predict_future_values(
        values: np.ndarray,
        keras_model: tf.keras.Model,
        future_steps: int) -> np.ndarray:
    input_values = values.copy()
    for _ in range(future_steps):
        input_values = np.append(input_values, model.predict(input_values[-seq_length:][np.newaxis])[0])
    return input_values

def calculate_error_rate(original: np.ndarray, prediction: np.ndarray) -> float:
    return np.mean(np.abs((original - prediction) / original)) * 100

pred_1500 = predict_future_values(train_label.values[:1500], model, 30)
error_rate = calculate_error_rate(train_label.values[:1500 + 30], pred_1500)
print("1500번 데이터에 대한 오차율:", error_rate)