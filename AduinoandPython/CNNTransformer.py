import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

## 데이터 로드 및 균형 조정
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data, self.labels = self.load_and_balance_data(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    ## 데이터 로드 및 다운샘플링
    def load_and_balance_data(self, data_dir):
        data = []
        labels = []

        ## 재귀적으로 모든 .txt 파일 로드
        for file_path in glob.glob(f"{data_dir}/**/*.txt", recursive=True):
            try:
                array = np.loadtxt(file_path)
                label = 1 if "scream" in file_path else 0
                data.append(array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

        ## 패딩 및 길이 조정
        max_length = 5000
        data = [np.pad(x, (0, max_length - len(x)), 'constant')[:max_length] for x in data]
        data = np.array(data)
        labels = np.array(labels)

        ## 데이터 불균형 확인
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count
        print(f"Positive samples (scream): {pos_count}, Negative samples (non-scream): {neg_count}")

        ## 다운샘플링 적용 (음성 비명 데이터 기준으로 맞춤)
        if pos_count < neg_count:
            pos_indices = np.where(labels == 1)[0]
            neg_indices = np.where(labels == 0)[0]
            sampled_neg_indices = np.random.choice(neg_indices, size=pos_count, replace=False)
            balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
        else:
            balanced_indices = np.arange(len(labels))

        ## 다운샘플링된 데이터 반환
        balanced_data = data[balanced_indices]
        balanced_labels = labels[balanced_indices]

        print(f"Balanced data size: {len(balanced_labels)}")
        return balanced_data, balanced_labels

## CNN + Transformer 모델 정의
class CNNTransformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch, feature)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, feature)
        x = self.fc(x)
        return x

## 학습 함수
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in dataloader:
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

## 평가 함수
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

## 데이터 로더 준비
data_dir = './dataset'
batch_size = 32

dataset = AudioDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

## 모델 초기화 및 하이퍼파라미터 설정
input_size = 5000
num_classes = 2
num_epochs = 10
learning_rate = 0.001

model = CNNTransformer(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## 모델 학습
print("Training the model...")
train(model, dataloader, criterion, optimizer, num_epochs)

## 모델 평가
print("Evaluating the model...")
evaluate(model, dataloader)

## 모델 저장
torch.save(model.state_dict(), "cnn_transformer_model.pth")
print("Model saved as cnn_transformer_model.pth")