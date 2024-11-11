import os
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train data')
TEST_DIR = os.path.join(BASE_DIR, 'test data')
RAWTRAIN_DIR = os.path.join(BASE_DIR, 'rawtrain')
RAWTEST_DIR = os.path.join(BASE_DIR, 'rawtest')

# 모든 하위 디렉터리까지 파일 탐색
def find_audio_files(input_dir):
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
                print(f"Audio file found: {file_path}")
    return audio_files

# Fourier Transform 함수
def perform_fourier_transform(file_path):
    try:
        print(f"Loading audio file: {file_path}")
        data, sample_rate = librosa.load(file_path, sr=None)
        if data is None or len(data) == 0:
            print(f"Failed to load audio (empty data): {file_path}")
            return None
        print(f"Audio loaded successfully: {file_path}, Sample rate: {sample_rate}, Data length: {len(data)}")

        # Fourier Transform
        freq_data = np.fft.fft(data)
        magnitude = np.abs(freq_data)
        return magnitude
    except Exception as e:
        print(f"Error during file processing: {file_path}, Error message: {str(e)}")
        return None

# 데이터 저장 함수
def save_to_txt(data, output_path):
    try:
        if data is None or len(data) == 0:
            print(f"No data to save: {output_path}")
            return
        print(f"Saving file: {output_path}, Data length: {len(data)}")
        np.savetxt(output_path, data, fmt='%.6f')
        print(f"File saved successfully: {output_path}")
    except Exception as e:
        print(f"Failed to save file: {output_path}, Error message: {str(e)}")

# 오디오 파일 처리 함수
def process_audio_files(input_dir, output_dir):
    audio_files = find_audio_files(input_dir)
    for file_path in audio_files:
        relative_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, relative_path).replace('.wav', '.txt').replace('.mp3', '.txt').replace('.ogg', '.txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        magnitude = perform_fourier_transform(file_path)
        if magnitude is not None:
            save_to_txt(magnitude, output_path)

# 메인 함수
def main():
    os.makedirs(RAWTRAIN_DIR, exist_ok=True)
    os.makedirs(RAWTEST_DIR, exist_ok=True)

    print("Processing train data...")
    process_audio_files(TRAIN_DIR, RAWTRAIN_DIR)

    print("Processing test data...")
    process_audio_files(TEST_DIR, RAWTEST_DIR)

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
