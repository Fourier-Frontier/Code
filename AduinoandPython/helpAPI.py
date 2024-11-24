import os
import sounddevice as sd
import librosa
import speech_recognition as sr

# 🟢 탐지할 키워드 정의
keywords = {
    "en": ["help", "please help", "help me"],
    "ko": ["도와줘", "도와주세요"],
    "es": ["ayuda", "ayúdame"],
    "fr": ["aidez-moi", "au secours"],
    "de": ["hilfe", "bitte hilfe"]
}

# 🟢 Google Speech API를 통한 다국어 지원 탐지
def detect_help(audio_data, sample_rate, lang="en"):
    recognizer = sr.Recognizer()
    try:
        # Audio 데이터를 WAV 파일로 저장 후 처리
        librosa.output.write_wav("temp.wav", audio_data, sample_rate)
        with sr.AudioFile("temp.wav") as source:
            audio = recognizer.record(source)
            # Google Speech API 호출 (언어 설정)
            text = recognizer.recognize_google(audio, language=lang)
            print(f"[{lang}] Transcription: {text}")

            # 탐지된 텍스트와 키워드 비교
            for keyword in keywords.get(lang, []):
                if keyword.lower() in text.lower():
                    return True
        return False
    except sr.UnknownValueError:
        print(f"Could not understand audio for language: {lang}")
        return False
    except Exception as e:
        print(f"Error detecting help for language {lang}: {e}")
        return False
    finally:
        # 임시 파일 삭제
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

# 🟢 실시간 음성 처리
def record_and_detect_help(duration=5, sample_rate=16000, languages=["en"]):
    print("Listening for audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 녹음 완료 대기
    audio_data = audio_data.flatten()  # 1D로 변환

    # 언어별로 `help` 단어 탐지
    for lang in languages:
        help_detected = detect_help(audio_data, sample_rate, lang=lang)
        if help_detected:
            print(f"Help detected in language: {lang}")
            return True

    print("No help detected in any language.")
    return False

# 🟢 실행 루프
def main():
    languages = ["en", "ko", "es", "fr", "de"]  # 지원 언어 목록
    while True:
        try:
            user_input = input("Type 'stop' to end or press Enter to record: ").strip().lower()
            if user_input == 'stop':
                print("Stopping the program.")
                break
            record_and_detect_help(languages=languages)
        except KeyboardInterrupt:
            print("Program interrupted.")
            break

if __name__ == "__main__":
    main()
