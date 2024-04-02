import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 데이터 폴더 지정
data_dir = pathlib.Path('D:/PycharmProjects/_Merged_Data')

# csv 파일 경로 리스트 생성
csv_files = list(data_dir.glob('*.csv'))

# 예측할 시간 간격 설정
forecast_steps = [30, 60, 100, 300, 600, 1200]

# 각각의 csv 파일에 대해 모델 생성 및 예측
for csv_file in csv_files:
    dataframe = pd.read_csv(csv_file)

    # 각각의 시계열 데이터에 대해 모델 생성 및 예측
    for column in dataframe.columns:
        data = dataframe[column].values

        # 모델을 학습시킬 데이터와 레이블을 생성합니다.
        X = []
        y = []
        for i in range(len(data) - max(forecast_steps) - 1):
            X.append(data[i: i + max(forecast_steps)])
            y.append([data[i + step] for step in forecast_steps])
        X = np.array(X)
        y = np.array(y)

        # 모델 구조를 정의합니다.
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        model.add(Dense(len(forecast_steps)))
        model.compile(optimizer='adam', loss='mse')

        # 모델을 학습시킵니다.
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model.fit(X, y, epochs=200, verbose=0)

        # 예측을 수행합니다.
        x_input = np.array(data[-max(forecast_steps):])
        x_input = x_input.reshape((1, len(x_input), 1))
        yhat = model.predict(x_input, verbose=0)
        print(f"Predictions for {column} in {csv_file.name}: {yhat}")

        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.plot(range(len(data) - 1, len(data) - 1 + len(yhat[0])), yhat[0])
        plt.title(f"{column} in {csv_file.name}")
        plt.show()