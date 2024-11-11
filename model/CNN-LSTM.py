import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 데이터 디렉터리 설정
data_dir = "./data"

# 데이터 로드 및 자동 라벨링 함수 정의
def load_data(directory, max_allowed_length=10000):
    data = []
    labels = []

    print(f"Loading data from: {directory}")
    file_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                label = 1 if 'scream' in filename.lower() else 0
                file_count += 1

                try:
                    numerical_data = np.loadtxt(file_path, dtype=np.float32)

                    if len(numerical_data) > max_allowed_length:
                        numerical_data = numerical_data[:max_allowed_length]

                    data.append(numerical_data)
                    labels.append(label)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    if file_count == 0:
        print("No .txt files found in the directory.")
        return np.array([]), np.array([])

    if len(data) == 0:
        print("No valid data found.")
        return np.array([]), np.array([])

    max_length = min(max(len(x) for x in data), max_allowed_length)
    data = [np.pad(x, (0, max_length - len(x)), 'constant') for x in data]

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

# 데이터 로드
all_data, all_labels = load_data(data_dir)

# 언더샘플링 적용
positive_data = all_data[all_labels == 1]
negative_data = all_data[all_labels == 0]

np.random.seed(42)
undersampled_negative_data = negative_data[np.random.choice(len(negative_data), len(positive_data), replace=False)]

balanced_data = np.concatenate([positive_data, undersampled_negative_data])
balanced_labels = np.concatenate([np.ones(len(positive_data)), np.zeros(len(undersampled_negative_data))])

indices = np.random.permutation(len(balanced_data))
balanced_data = balanced_data[indices]
balanced_labels = balanced_labels[indices]

# TensorDataset 및 DataLoader 설정
batch_size = 32
dataset = TensorDataset(torch.tensor(balanced_data), torch.tensor(balanced_labels))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# CNN + LSTM 모델 정의
class CNN_LSTM(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(64 * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :]

        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 모델 초기화
input_size = balanced_data.shape[1]
model = CNN_LSTM(input_size=input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 학습 함수 정의
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(data).squeeze().float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 모델 학습
num_epochs = 10
train(model, train_loader, criterion, optimizer, num_epochs)

# 모델 평가 함수 정의
def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)
        outputs = model(data).squeeze()
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == labels).float().mean().item()
        print(f"Accuracy: {accuracy:.4f}")

# 모델 평가
evaluate(model, balanced_data, balanced_labels)
