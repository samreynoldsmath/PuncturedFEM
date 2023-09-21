import numpy as np


def fft_derivative(f: np.ndarray, interval_length: float) -> np.ndarray:
    N = len(f)
    omega = np.fft.fft(f)
    omega *= 1j * N * np.fft.fftfreq(N)
    omega *= 2 * np.pi / interval_length
    return np.real(np.fft.ifft(omega))


def fft_antiderivative(df: np.ndarray, interval_length: float) -> np.ndarray:
    N = len(df)
    omega = np.fft.fft(df)
    fft_idx = np.fft.fftfreq(len(df))
    fft_idx[0] = 1
    omega *= -1j / (N * fft_idx)
    omega *= 0.5 * interval_length / np.pi
    omega[0] = 0
    return np.real(np.fft.ifft(omega))
