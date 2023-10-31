import librosa
import matplotlib.pyplot as mpl
import librosa.display as ld
import numpy as np

Sample_Rate = 22050
path = 'ru_test.wav'


def get_signal(path, SR):
    signal, sr = librosa.load(path, sr=SR)
    mpl.figure(figsize=(12, 4))
    return signal, sr


def AmplitudeOnTime():
    global path
    global Sample_Rate
    signal, sr = get_signal(path, Sample_Rate)
    # ld.waveshow(signal, sr=22050)  # Амплитуда от времени; конфликт версий
    mpl.plot(signal)
    mpl.xlabel('Samples')
    mpl.ylabel('Amplitude')
    mpl.show()


def AmplitudeOnFreq():
    global path
    global Sample_Rate
    signal, sr = get_signal(path, Sample_Rate)
    n_fft = 2048  # длина оконного сигнала после заполнения нулями, hop_length размер быстрого преобразования Фурье
    ft = np.abs(librosa.stft(signal[:n_fft], hop_length=n_fft + 1))
    mpl.plot(ft)  # Амплитуда от частоты
    mpl.title('Spectrum')
    mpl.xlabel('Frequency Bin')
    mpl.ylabel('Amplitude')
    mpl.show()


def Spectrogram():
    global path
    global Sample_Rate
    signal, sr = get_signal(path, Sample_Rate)
    x = librosa.stft(signal)  # Быстрое преобразование Фурье
    s = librosa.amplitude_to_db(abs(x))
    ld.specshow(s, sr=sr, x_axis='time', y_axis='linear')
    mpl.colorbar()
    mpl.show()


def get_mfcc():
    global path
    global Sample_Rate
    signal, sr = get_signal(path, Sample_Rate)
    return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40,
                                hop_length=512)  # n_mfcc - еол-во коэфф | hop_length - размер кадра


def mfcc_spectro():
    global path
    global Sample_Rate
    signal, sr = get_signal(path, Sample_Rate)
    mls = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_mels=40)
    ld.specshow(librosa.power_to_db(mls, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    mpl.colorbar(format='%+2.0f dB')
    mpl.title('Mel spectrogram')
    mpl.show()


signal, sr = get_signal(path, Sample_Rate)
print(signal)
print(get_mfcc())
AmplitudeOnTime()
AmplitudeOnFreq()
Spectrogram()
mfcc_spectro()
