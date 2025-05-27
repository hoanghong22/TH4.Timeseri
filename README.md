# TH4.Timeseri
# Thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numpy.fft import fft, fftfreq

# Hàm đọc dữ liệu với kiểm tra cột
def read_signal(file_path, column_name=None):
    df = pd.read_csv(file_path)
    print(f"Columns in {file_path}:", df.columns.tolist())  
    if column_name is None:
      
        signal = df.iloc[:, 0].values
    else:
        signal = df[column_name].values
    return signal

# chạy thử
inner_file = 'XYZ_IR(991).csv'
normal_file = 'XYZ_N(986).csv'
outer_file = 'XYZ_OR(99).csv'

# chạy từng file để xem tên cột 
inner_signal = read_signal(inner_file)    # chưa biết cột, lấy tạm cột đầu
normal_signal = read_signal(normal_file)
outer_signal = read_signal(outer_file)

def read_signal(file_path, col_idx=0):
    df = pd.read_csv(file_path)
    signal = df.iloc[:, col_idx].values
    return signal

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot_signals(original, filtered, title):
    plt.figure(figsize=(12,5))
    plt.plot(original, label='Original')
    plt.plot(filtered, label='Filtered', linewidth=2)
    plt.title(f'Signal Filtering: {title}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def plot_fft(signal, fs, title):
    N = len(signal)
    T = 1.0 / fs
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]

    plt.figure(figsize=(12,5))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(f'Frequency Spectrum: {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Đường dẫn file
    inner_file = 'XYZ_IR(991).csv'
    normal_file = 'XYZ_N(986).csv'
    outer_file = 'XYZ_OR(99).csv'

    fs = 1000  # Tần số lấy mẫu (Hz), bạn chỉnh đúng nếu khác
    cutoff = 50  # Tần số cắt lọc (Hz)

    # Đọc tín hiệu cột đầu tiên
    inner_signal = read_signal(inner_file, col_idx=0)
    normal_signal = read_signal(normal_file, col_idx=0)
    outer_signal = read_signal(outer_file, col_idx=0)

    # Lọc tín hiệu normal và outer dùng bộ lọc thiết kế
    normal_filtered = butter_lowpass_filter(normal_signal, cutoff, fs)
    outer_filtered = butter_lowpass_filter(outer_signal, cutoff, fs)

    # Vẽ biểu đồ
    plot_signals(normal_signal, normal_filtered, 'Normal Signal')
    plot_fft(normal_signal, fs, 'Normal Signal - Original')
    plot_fft(normal_filtered, fs, 'Normal Signal - Filtered')

    plot_signals(outer_signal, outer_filtered, 'Outer Signal')
    plot_fft(outer_signal, fs, 'Outer Signal - Original')
    plot_fft(outer_filtered, fs, 'Outer Signal - Filtered')

    print("Xử lý lọc tín hiệu hoàn tất.")
