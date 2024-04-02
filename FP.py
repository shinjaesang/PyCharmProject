import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import tensorflow as tf
import glob
import matplotlib.pyplot as plt

# Dense 모델 생성을 위한 함수
def create_model(input_shape, num_features):
    inputs = Input(input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = [Dense(1)(x) for _ in range(num_features)]
    model = Model(inputs, outputs)
    return model

# 학습률 스케줄러 함수
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# CSV 파일 로드
csv_files = glob.glob("/_Merged_Data/*.csv")  # CSV 파일 리스트
all_data = []  # 데이터 저장 리스트
min_values = ['temp1', 'press1', 'temp2', 'press3', 'Accel', 'GasLeak']  # 실제 데이터 프레임의 열 이름을 사용
scaler = MinMaxScaler(feature_range=(0, 1))  # 정규화를 위한 스케일러

# 각 CSV 파일을 처리하고 리스트에 추가
for file in csv_files:
    data = pd.read_csv(file, names=min_values)
    max_values = data.max().values  # 각 열의 최대값 계산
    data = np.minimum(data, np.tile(max_values, (len(data), 1)))  # 각 열의 값이 최대값을 초과하면 최대값으로 대체
    data_normalized = scaler.fit_transform(data)  # 정규화
    sequence_length = 10
    X, y = [], []

    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:i+sequence_length, :])
        y.append(data_normalized[i+sequence_length, :])

    all_data.append((np.array(X), np.array(y)))

X = np.concatenate([data[0] for data in all_data])
y = np.concatenate([data[1] for data in all_data])

# 훈련 데이터와 검증 데이터로 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dense 모델 생성
model = create_model((sequence_length, len(min_values)), len(min_values))

# 모델 컴파일
model.compile(optimizer=Adam(), loss='mse')

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
lr_scheduler = LearningRateScheduler(scheduler)

# 모델 훈련
history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])], epochs=10, batch_size=32, validation_data=(X_val, [y_val[:, i] for i in range(y_val.shape[1])]), callbacks=[early_stopping, model_checkpoint, lr_scheduler])  # 각 특성별로 분리하여 학습

# 테스트 데이터에 대한 예측
test_file = '_Merged_Data_data_set_01500.csv'
test_data = pd.read_csv(test_file, names=min_values)
test_data = np.minimum(test_data, np.tile(min_values, (len(test_data), 1)))
test_data_normalized = scaler.transform(test_data)  # 정규화
test_sequence = []

for i in range(len(test_data_normalized) - sequence_length):
    test_sequence.append(test_data_normalized[i:i+sequence_length, :])

test_sequence = np.array(test_sequence)
predictions = model.predict(test_sequence)

# 예측 그래프 그리기
plt.figure(figsize=(10, 6))

features = ['temp1', 'press1', 'temp2', 'press3', 'Accel', 'GasLeak']  # 실제 데이터 프레임의 열 이름을 사용

for i in range(6):  # 6은 예측하려는 특성의 수
    plt.plot(test_data[sequence_length:, i], label=f'진짜 {features[i]}')
    plt.plot(predictions[:, i], label=f'예측 {features[i]}', linestyle='dashed')
    plt.plot(test_data.iloc[sequence_length:, i].values, label=f'원본 {features[i]}', linestyle='dotted')

plt.title("진짜 vs 예측 vs 원본 데이터")
plt.legend()
plt.tight_layout()
plt.show()