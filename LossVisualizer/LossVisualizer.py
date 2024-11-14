import re
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import scrolledtext

def plot_loss():
    data = text_area.get("1.0", tk.END)  
    loss_values = re.findall(r'Loss:\s([0-9.]+)', data)
    loss_values = [float(loss) for loss in loss_values]
    epochs = list(range(1, len(loss_values) + 1))
    
    # 새로운 Figure 객체 생성
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_values, marker='o')
    ax.set_title('Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(epochs)
    ax.grid(True)
    
    plt.show()

# Tkinter 윈도우 설정
window = tk.Tk()
window.title("Loss Visualizer")
window.geometry("350x280")

# 설명 레이블
label = tk.Label(window, text="Training log data (copy & paste here):")
label.pack(pady=5)

# 텍스트 입력 영역
text_area = scrolledtext.ScrolledText(window, width=40, height=10)
text_area.pack(pady=10)

# 버튼
button = tk.Button(window, text="Plot Loss", command=plot_loss, width=30, height= 3)
button.pack(pady=10)

# GUI 실행
window.mainloop()