# 📋 Arduino-Python Real-Time Audio Analysis Requirements

## 1. **Project Overview**
- **Objective**: Utilize real-time audio data from Arduino to detect **scream sounds** using a pre-trained CNN+Transformer model and identify the word **'help'** using a Speech Recognition API.
- **Hardware Requirements**:
  - Arduino with a microphone module (e.g., MAX9814, KY-038)
  - USB connection for serial communication with Python
- **Software Requirements**:
  - Scream sound detection using CNN+Transformer model in Python
  - 'Help' word detection using Speech Recognition API

## 2. **Arduino Requirements**
### 2.1 **Microphone Module Setup**
- Connect the microphone module to Arduino to capture real-time audio data.
- Sample the audio data at **16 kHz**.
- Send the audio data in **0.5-second intervals** to Python for processing.

### 2.2 **Data Transmission**
- Initialize serial communication with `Serial.begin(115200)`.
- Transmit the audio data as an integer array or binary format using `Serial.write()`.
- Ensure Python can read the data correctly by checking `Serial.available()` before transmission.

### 2.3 **Error Handling**
- If an error occurs on the Python side, stop data transmission and enter **standby mode**.
- Retry sending data after confirming Python is ready using `Serial.available()`.

## 3. **Python Requirements**
### 3.1 **Serial Communication Setup**
- Use `pyserial` in Python to connect with Arduino (`serial.Serial()`).
- Set the port (`COM3`, `COM4`, etc.) and `baudrate` to **115200**.

### 3.2 **Audio Data Processing**
- Buffer the received audio data in **0.5-second chunks**.
- Perform **Fourier Transform** on the buffered data for feature extraction before passing it to the CNN+Transformer model.

### 3.3 **Model Inference and Output**
- Pass the processed data to the CNN+Transformer model for scream detection.
- If a scream is detected, print `Scream detected!`.
- Use the Speech Recognition API to detect the word 'help'.
  - If 'help' is detected, print `Help detected!` and trigger a notification.

### 3.4 **Exit Condition**
- The analysis loop will run indefinitely until the user inputs **'stop'**.
- Close the serial port and exit the program safely when 'stop' is entered.

## 4. **Expected Challenges and Solutions**
### 4.1 **Data Transmission Delay**
- Delays may occur during serial communication. Adjust the **buffer size** and use asynchronous data reading to reduce latency.

### 4.2 **Noise Interference**
- Background noise may reduce the accuracy of the model. Implement **noise filtering** before feature extraction.

### 4.3 **High CPU Usage**
- Real-time processing may cause high CPU usage. Use **multithreading** or **asynchronous processing** to handle data more efficiently.

## 5. **Additional Resources**
- **Python Libraries**:
  - `pyserial`: For Arduino-Python serial communication
  - `numpy`, `scipy`: For Fourier Transform and data processing
  - `matplotlib`: For real-time plotting of the frequency spectrum
- **Arduino Libraries**:
  - `Serial`: For serial communication
  - `ADC`: For analog data reading from the microphone module

---

### Next Steps:
- Review the requirements and confirm the setup.
- Start coding the Arduino sketch for audio data collection.
- Implement Python script for real-time analysis and model inference.


# 📋 아두이노-파이썬 실시간 음성 분석 요구 사항

## 1. **프로젝트 개요**
- **목표**: 아두이노에서 실시간 음성 데이터를 수집하고, 사전 학습된 CNN+Transformer 모델을 사용해 **비명 소리**를 탐지하며, Speech Recognition API를 통해 **'help'**라는 단어를 감지합니다.
- **하드웨어 요구 사항**:
  - 아두이노 및 마이크 모듈 (예: MAX9814, KY-038)
  - USB 연결을 통한 파이썬과의 시리얼 통신
- **소프트웨어 요구 사항**:
  - Python에서 CNN+Transformer 모델을 사용해 비명 소리 탐지
  - Speech Recognition API를 이용한 'help' 단어 탐지

## 2. **아두이노 요구 사항**
### 2.1 **마이크 모듈 설정**
- 마이크 모듈을 아두이노에 연결하여 실시간 음성 데이터를 수집합니다.
- **16 kHz**로 샘플링합니다.
- **0.5초 간격**으로 오디오 데이터를 Python으로 전송합니다.

### 2.2 **데이터 전송**
- `Serial.begin(115200)`을 사용해 시리얼 통신을 초기화합니다.
- 오디오 데이터를 **정수 배열** 또는 **이진 형식**으로 `Serial.write()`를 사용해 전송합니다.
- 전송 전에 `Serial.available()`를 확인하여 Python 측에서 데이터를 받을 준비가 되었는지 체크합니다.

### 2.3 **오류 처리**
- Python 측에서 오류가 발생하면 데이터 전송을 멈추고 **대기 모드**로 전환합니다.
- Python이 준비되면 `Serial.available()` 확인 후 데이터를 재전송합니다.

## 3. **파이썬 요구 사항**
### 3.1 **시리얼 통신 설정**
- Python에서는 `pyserial` 라이브러리를 사용해 아두이노와 연결합니다 (`serial.Serial()`).
- 포트 (`COM3`, `COM4` 등)와 `baudrate`는 **115200**으로 설정합니다.

### 3.2 **오디오 데이터 처리**
- 수신된 오디오 데이터를 **0.5초 단위**로 버퍼링합니다.
- 버퍼링된 데이터를 기반으로 **푸리에 변환**을 수행해 특징을 추출하고, CNN+Transformer 모델에 전달합니다.

### 3.3 **모델 예측 및 출력**
- 추출된 특징을 CNN+Transformer 모델에 전달해 비명 소리를 감지합니다.
  - 비명 소리가 감지되면 `비명 소리 탐지!` 출력
- Speech Recognition API를 사용해 'help' 단어를 감지합니다.
  - 'help'가 감지되면 `도움 요청 탐지!` 출력 및 알림 트리거

### 3.4 **종료 조건**
- 분석 루프는 **무한 루프**로 동작하며, 사용자가 **'stop'**을 입력하면 종료합니다.
- 시리얼 포트를 안전하게 닫고 프로그램을 종료합니다.

## 4. **예상되는 문제 및 해결 방안**
### 4.1 **데이터 전송 지연**
- 시리얼 통신 중 지연이 발생할 수 있습니다. **버퍼 크기**를 조정하고 비동기 데이터 수신을 통해 지연을 줄입니다.

### 4.2 **잡음 간섭**
- 배경 소음이 모델의 정확도를 낮출 수 있습니다. **노이즈 필터링**을 통해 특징 추출 전 잡음을 제거합니다.

### 4.3 **높은 CPU 사용량**
- 실시간 처리가 CPU 사용량을 높일 수 있습니다. **멀티스레딩** 또는 **비동기 처리**를 사용해 성능을 개선합니다.

## 5. **추가 리소스**
- **Python 라이브러리**:
  - `pyserial`: 아두이노와의 시리얼 통신
  - `numpy`, `scipy`: 푸리에 변환 및 데이터 처리
  - `matplotlib`: 실시간 주파수 스펙트럼 그래프 표시
- **Arduino 라이브러리**:
  - `Serial`: 시리얼 통신
  - `ADC`: 마이크 모듈의 아날로그 데이터 읽기

---

### 다음 단계:
- 요구 사항을 검토하고 설정을 확인합니다.
- 아두이노에서 오디오 데이터를 수집하기 위한 코딩을 시작합니다.
- Python 스크립트를 구현해 실시간 분석 및 모델 예측을 진행합니다.







