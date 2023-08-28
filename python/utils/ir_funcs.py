import numpy as np
import scipy.signal as signal


def get_freq_resp(x, fs=44100, win=None):
    N = len(x)  # Length of input sequence
    if win is None:
        win = np.ones(x.shape)
    sp = np.fft.rfft(x)
    w = np.arange((N / 2) + 1) / (float(N) / fs)
    mag = np.abs(sp) * 2 / np.sum(win)
    h = 20 * np.log10(mag)
    return w, h


def get_ir(input_ess, target_ess, fs=44100, f1=10, f2=22050, length_sec=5):
    t = np.arange(0, length_sec * fs) / fs
    R = np.log(f2 / f1)
    k = np.exp(t * R / length_sec)
    f = target_ess[::-1] / k
    return signal.convolve(input_ess, f, mode='same')
