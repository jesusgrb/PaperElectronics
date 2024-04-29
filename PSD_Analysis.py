import numpy as np
import matplotlib.pyplot as plt

def power_spectral_density_fft(signal, fs):
    """
    Calculate the Power Spectral Density (PSD) of a signal using the Fourier Transform.

    Args:
    signal (array): Input signal.
    fs (float): Sampling frequency of the signal.

    Returns:
    psd (array): Power Spectral Density.
    freqs (array): Vector of frequencies corresponding to the PSD.
    """

    # Calculate the Fourier Transform of the signal
    fft_signal = np.fft.fft(signal)

    # Calculate the Power Spectral Density (PSD)
    psd = np.abs(fft_signal) ** 2 / len(signal)

    # Calculate the corresponding frequency vector
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Only take half of the values since the FFT is symmetric
    psd = psd[:len(psd)//2]
    freqs = freqs[:len(freqs)//2]
    
    psd = psd / np.max(psd)


    return psd, freqs

def power_spectral_density_autocorr(signal, fs):
    """
    Calculate the Power Spectral Density (PSD) of a signal using autocorrelation.

    Args:
    signal (array): Input signal.
    fs (float): Sampling frequency of the signal.

    Returns:
    psd (array): Power Spectral Density.
    freqs (array): Vector of frequencies corresponding to the PSD.
    """

    # Calculate the autocorrelation of the signal
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take only half of the values

    # Calculate the Fourier Transform of the autocorrelation
    fft_autocorr = np.fft.fft(autocorr)

    # Calculate the Power Spectral Density (PSD)
    psd = np.abs(fft_autocorr) ** 2 / (fs * len(signal))

    # Calculate the corresponding frequency vector
    freqs = np.fft.fftfreq(len(autocorr), 1/fs)

    # Only take half of the values since the FFT is symmetric
    psd = psd[:len(psd)//2]
    freqs = freqs[:len(freqs)//2]
    
    psd = psd / np.max(psd)


    return psd, freqs



# Calcular PSD cruzada utilizando la función personalizada
def psd_cruzada(signal1, signal2, fs, nperseg):
    # Ajustar la longitud de la señal
    n = min(len(signal1), len(signal2), nperseg)
    fft_signal1 = np.fft.fft(signal1[:n], n=n)
    fft_signal2 = np.fft.fft(signal2[:n], n=n)
    fft_signal2_conjugate = np.conj(fft_signal2)
    psd_cross = fft_signal1 * fft_signal2_conjugate
    psd_cross /= (fs * n)  # Normalizar por la longitud de la señal y la frecuencia de muestreo
    frequencies = np.fft.fftfreq(n, 1/fs)[:n//2]
    return psd_cross[:n//2], frequencies