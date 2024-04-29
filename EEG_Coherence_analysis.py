import os
from scipy import signal
from PSD_Analysis import *
from Filters import *
import numpy as np
import matplotlib.pyplot as plt

# Games: cubism, puzzling places, tetris

# Simulation of example EEG signal (replace with your actual data)
fs = 500  # Sampling frequency in Hz
# Define cutoff frequencies for the beta band
# Beta: 12-30 Hz
# Gamma: 30-40 Hz
# Alpha: 8-12 Hz
lowcut = 8  # Lower cutoff frequency of the beta band in Hz
highcut = 12  # Upper cutoff frequency of the beta band in Hz
#****** columns ****** Beta
    # AF7 : 14
    # AF8 : 15
    # TP9 : 13
    # TP10 : 16
#****** columns ****** Gamma
    # AF7 : 18
    # AF8 : 19
    # TP9 : 17
    # TP10 : 20    
#****** columns ****** Alpha
    # AF7 : 10
    # AF8 : 11
    # TP9 : 9
    # TP10 : 12      
    
col1 = 9
col2 = 12

# EEG data for 30 people
n_people = 30
n_sample = 1024
FFTsize = 512

ms_coherence_cubism = []
ms_coherence_puzzling = []
ms_coherence_tetris = []

# Path to the folder containing the CSV file
#folder = '/Users/albertodel-puerto/Documents/Draft-Paper02-Game/EEG Data/Cubism'
folder = '/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Cubism'
# Subfolder name (if any)
subfolder = 'Con Audio'

# CSV file name
for person in range(n_people):
    # Construct the full path to the CSV file
    full_path = os.path.join(folder, subfolder, f'game1_{person}.csv')

    # Read data from the CSV file
    eeg_signal_data_1 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col1)
    eeg_signal_data_1 = np.nan_to_num(eeg_signal_data_1, nan=0)
    eeg_signal_data_2 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col2)
    eeg_signal_data_2 = np.nan_to_num(eeg_signal_data_2, nan=0)

    # Handle data length
    if len(eeg_signal_data_1) <= n_sample:
        aux = np.zeros(n_sample)
        aux[0:len(eeg_signal_data_1)] = eeg_signal_data_1
        eeg_signal_data_1 = aux
    else:
        eeg_signal_data_1 = eeg_signal_data_1[0:n_sample]
    
    if len(eeg_signal_data_2) <= n_sample:
        aux = np.zeros(n_sample)
        aux[0:len(eeg_signal_data_2)] = eeg_signal_data_2
        eeg_signal_data_2 = aux
    else:    
        eeg_signal_data_2 = eeg_signal_data_2[0:n_sample]
    

    # Apply filter to the EEG signal
    eeg_beta_filtered_1 = butter_bandpass_filter(eeg_signal_data_1, lowcut, highcut, fs, order=4)
    eeg_beta_filtered_2 = butter_bandpass_filter(eeg_signal_data_2, lowcut, highcut, fs, order=4)

    # Calculate PSD and CSD of the filtered signals
    frequency, psd_xx = signal.welch(eeg_beta_filtered_1, fs=fs, nperseg=FFTsize)
    frequency, psd_yy = signal.welch(eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)
    frequency, psd_xy = signal.csd(eeg_beta_filtered_1, eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)

    print("psd cruzada ",psd_xy)

    msc = np.abs(psd_xy)**2 / (psd_xx * psd_yy)
    # Calculate ms_coherence and append to corresponding list
    if not np.isnan(msc).any():
        ms_coherence_cubism.append(msc)

# Calculate median by columns of ms_coherence_cubism
ms_coherence_cubism_median = np.median(np.array(ms_coherence_cubism), axis=0)
print("ms coherence cubism ", ms_coherence_cubism_median)


# Path to the folder containing the CSV file
#folder = '/Users/albertodel-puerto/Documents/Draft-Paper02-Game/EEG Data/Puzzling'
folder = '/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Puzzling'

# Subfolder name (if any)
subfolder = 'Con Audio'

# CSV file name
for person in range(n_people):
    # Construct the full path to the CSV file
    full_path = os.path.join(folder, subfolder, f'game2_{person}.csv')

    # Read data from the CSV file
    eeg_signal_data_1 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col1)
    eeg_signal_data_1 = np.nan_to_num(eeg_signal_data_1, nan=0)
    eeg_signal_data_2 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col2)
    eeg_signal_data_2 = np.nan_to_num(eeg_signal_data_2, nan=0)

    # Handle data length
    if len(eeg_signal_data_1) <= n_sample:
        eeg_signal_data_1 = np.pad(eeg_signal_data_1, (0, n_sample - len(eeg_signal_data_1)), mode='constant')
    else:
        eeg_signal_data_1 = eeg_signal_data_1[:n_sample]

    if len(eeg_signal_data_2) <= n_sample:
        eeg_signal_data_2 = np.pad(eeg_signal_data_2, (0, n_sample - len(eeg_signal_data_2)), mode='constant')
    else:
        eeg_signal_data_2 = eeg_signal_data_2[:n_sample]

    # Apply filter to the EEG signal
    eeg_beta_filtered_1 = butter_bandpass_filter(eeg_signal_data_1, lowcut, highcut, fs, order=4)
    eeg_beta_filtered_2 = butter_bandpass_filter(eeg_signal_data_2, lowcut, highcut, fs, order=4)

    # Calculate PSD and CSD of the filtered signals
    frequency, psd_xx = signal.welch(eeg_beta_filtered_1, fs=fs, nperseg=FFTsize)
    frequency, psd_yy = signal.welch(eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)
    frequency, psd_xy = signal.csd(eeg_beta_filtered_1, eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)

    # Calculate ms_coherence and append to corresponding list
    msc = np.abs(psd_xy)**2 / (psd_xx * psd_yy)
    # Calculate ms_coherence and append to corresponding list
    if not np.isnan(msc).any():
        ms_coherence_puzzling.append(msc)

# Calculate median by columns of ms_coherence_puzzling
ms_coherence_puzzling_median = np.median(np.array(ms_coherence_puzzling), axis=0)

# Path to the folder containing the CSV file
#folder = '/Users/albertodel-puerto/Documents/Draft-Paper02-Game/EEG Data/Tetris'
folder = '/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Tetris'

# Subfolder name (if any)
subfolder = 'Con Audio'

# CSV file name
for person in range(n_people):
    # Construct the full path to the CSV file
    full_path = os.path.join(folder, subfolder, f'game3_{person}.csv')

    # Read data from the CSV file
    eeg_signal_data_1 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col1)
    eeg_signal_data_1 = np.nan_to_num(eeg_signal_data_1, nan=0)
    eeg_signal_data_2 = np.genfromtxt(full_path, delimiter=',', skip_header=1, usecols=col2)
    eeg_signal_data_2 = np.nan_to_num(eeg_signal_data_2, nan=0)

    # Handle data length
    if len(eeg_signal_data_1) <= n_sample:
        eeg_signal_data_1 = np.pad(eeg_signal_data_1, (0, n_sample - len(eeg_signal_data_1)), mode='constant')
    else:
        eeg_signal_data_1 = eeg_signal_data_1[:n_sample]

    if len(eeg_signal_data_2) <= n_sample:
        eeg_signal_data_2 = np.pad(eeg_signal_data_2, (0, n_sample - len(eeg_signal_data_2)), mode='constant')
    else:
        eeg_signal_data_2 = eeg_signal_data_2[:n_sample]

    # Apply filter to the EEG signal
    eeg_beta_filtered_1 = butter_bandpass_filter(eeg_signal_data_1, lowcut, highcut, fs, order=4)
    eeg_beta_filtered_2 = butter_bandpass_filter(eeg_signal_data_2, lowcut, highcut, fs, order=4)

    # Calculate PSD and CSD of the filtered signals
    frequency, psd_xx = signal.welch(eeg_beta_filtered_1, fs=fs, nperseg=FFTsize)
    frequencye, psd_yy = signal.welch(eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)
    frequency, psd_xy = signal.csd(eeg_beta_filtered_1, eeg_beta_filtered_2, fs=fs, nperseg=FFTsize)

    # Calculate ms_coherence and append to corresponding list
    msc = np.abs(psd_xy)**2 / (psd_xx * psd_yy)
    # Calculate ms_coherence and append to corresponding list
    if not np.isnan(msc).any():
        ms_coherence_tetris.append(msc)

# Calculate median by columns of ms_coherence_tetris
ms_coherence_tetris_median = np.median(np.array(ms_coherence_tetris), axis=0)

# Plot both signals on the same graph
# plt.plot(frequency, ms_coherence_cubism_median, color='b', linewidth=1.5, label='Cubism, Pair AF7-AF8')
# plt.plot(frequency, ms_coherence_puzzling_median, color='k', linewidth=1.5, linestyle='-', label='Puzzling, Pair AF7-AF8')
# plt.plot(frequency, ms_coherence_tetris_median, color='r', linewidth=1.5, linestyle='-', label='Tetris, Pair AF7-AF8')
plt.plot(frequency, ms_coherence_cubism_median, color='b', linewidth=1.5, label='Cubism, Pair TP9-TP10')
plt.plot(frequency, ms_coherence_puzzling_median, color='k', linewidth=1.5, linestyle='-', label='Puzzling, Pair TP9-TP10')
plt.plot(frequency, ms_coherence_tetris_median, color='r', linewidth=1.5, linestyle='-', label='Tetris, Pair TP9-TP10')

plt.xlabel('Frequency (Hz)')
plt.ylabel('$|C_{xy}(f)|^2$')
plt.legend()  # Add legend
plt.grid(True, linestyle='--')
plt.xlim(0, 250)
plt.legend(loc='lower right')
plt.show()
