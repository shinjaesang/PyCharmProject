import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터 불러오기
files = glob.glob('_Merged_Data/*.csv')

# 데이터 전처리
all_data = []
min_values = [100, 3.25, 100, 3.25, 0.1, 1007, 0]

for file in files:
    data = pd.read_csv(file)
    data = np.minimum(data, np.tile(min_values, (len(data), 1)))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    sequence_length = 10
    X, y = [], []

    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:i+sequence_length, :])
        y.append(data_normalized[i+sequence_length, :])

    all_data.append((np.array(X), np.array(y)))

X = np.concatenate([data[0] for data in all_data])
y = np.concatenate([data[1] for data in all_data])

# 모델 설계
input_shape = (sequence_length, 7)  # 특성 수 변경
inputs = tf.keras.Input(input_shape)
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = [Dense(1)(x) for _ in range(7)]  # 특성 수 변경
model = tf.keras.Model(inputs, outputs)

# 모델 훈련
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.compile(optimizer='adam', loss='mse')

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# 손실 그래프
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 예측
test_file = 'datasets/_Merged_Data_data_set_01500.csv'
test_data = pd.read_csv(test_file)
test_data = np.minimum(test_data, np.tile(min_values, (len(test_data), 1)))
test_data = test_data / np.tile(min_values, (len(test_data), 1))
test_sequence = []

for i in range(len(test_data) - sequence_length):
    test_sequence.append(test_data.iloc[i:i+sequence_length, :].values)

test_sequence = np.array(test_sequence)
predictions = model.predict(test_sequence)


# 예측 그래프 그리기
plt.figure(figsize=(10, 6))

features = ['인덱스', '온도1', '압력1', '온도2', '압력2', '진동', '가스']

for i in range(6):  # 6은 예측하려는 특성의 수
    plt.plot(test_data.values[:, i], label=f'진짜 {features[i]}')  # 수정: test_data는 DataFrame이므로 .value를 사용해야 합니다.
    plt.plot(predictions[:, i], label=f'예측 {features[i]}', linestyle='dashed')

# 원본 데이터 그리기
for i in range(6):
    # 수정: 원본 데이터가 test_data에 포함되어 있다는 가정이 명확하지 않으므로, 원본 데이터를 따로 불러와야 합니다.
    original_data = pd.read_csv('your_original_data_path.csv')
    original_data = np.minimum(original_data, np.tile(min_values, (len(original_data), 1)))
    original_data = original_data / np.tile(min_values, (len(original_data), 1))
    plt.plot(original_data.iloc[sequence_length:, i].values, label=f'원본 {features[i]}', linestyle='dotted')

plt.title("진짜 vs 예측 vs 원본 데이터")
plt.legend()
plt.tight_layout()
plt.show()
