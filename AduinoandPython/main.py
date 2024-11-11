import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
import torch
import torch.nn.functional as F
from cnn_transformer_model import CNNTransformerModel  # CNN+Transformer 모델을 가져옴
import serial
import threading

# 모델 설정
model = CNNTransformerModel()
model.load_state_dict(torch.load("path_to_trained_model.pth"))  # 모델의 학습된 가중치 경로를 지정
model.eval()  # 모델을 평가 모드로 전환

# 아두이노 포트 설정
ser = serial.Serial('COM3', 9600)  # COM 포트는 환경에 맞게 설정

# 그래프 초기화 설정
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)  # y축 범위 설정
ax.set_xlim(0, 100)  # x축 범위 설정

stop_flag = False  # 종료 플래그

def input_listener():
    """종료를 위한 사용자 입력 감지 함수"""
    global stop_flag
    while True:
        user_input = input("Type 'stop' to end: ")
        if user_input.lower() == 'stop':
            stop_flag = True
            break

def init():
    line.set_data(x_data, y_data)
    return line,

def update(frame):
    global stop_flag
    if stop_flag:
        plt.close(fig)
        ser.close()
        return line,

    if ser.in_waiting:
        try:
            # 아두이노에서 데이터 읽기 및 그래프에 추가
            raw_data = ser.readline().decode().strip()
            audio_sample = float(raw_data)
            x_data.append(time.time())
            y_data.append(audio_sample)

            # 푸리에 변환
            if len(y_data) > 100:
                y_data.pop(0)
                x_data.pop(0)

            freq_data = np.abs(fft(y_data))[:len(y_data)//2]
            freq_data_tensor = torch.tensor(freq_data, dtype=torch.float32).unsqueeze(0)

            # 모델로 예측
            with torch.no_grad():
                prediction = model(freq_data_tensor)
                prob = F.softmax(prediction, dim=1)[0, 1].item()

            # 예측 결과 표시
            if prob > 0.5:
                print("비명 소리 감지!")
            else:
                print("비명 소리 없음")

            # 그래프 업데이트
            line.set_data(x_data, y_data)
            ax.set_xlim(x_data[0], x_data[-1])

        except Exception as e:
            print("Error:", e)

    return line,

# 종료 스레드 실행
threading.Thread(target=input_listener, daemon=True).start()

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
plt.show()
