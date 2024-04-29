import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
from PSD_Analysis import *
from Filters import *
from scipy import signal

#Games: cubism, puzzling places, tetris

# Simulation of an example EEG signal (replace with your actual data)
fs = 500  # Sampling frequency in Hz
# Define the cutoff frequencies for the beta band
# Beta: 12-30 Hz
# Gamma: 30-40 Hz
# Alpha: 8-12 Hz
lowcut = 30  # Lower cutoff frequency of the beta band in Hz
highcut = 100  # Upper cutoff frequency of the beta band in Hz
#****** columns ******
    # AF7 : 18
    # AF8 : 19
    # TP9 : 17
    # TP10: 20 
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

col1=20

# EEG data for 30 people
n_game = 3 
n_people = 30
n_sample = 1024
FFTsize=1024
spectral_entropy = np.zeros((n_game,n_people))
H = np.zeros(n_people)
spectral_entropy_puzzling = np.zeros(n_people)
spectral_entropy_tetris = np.zeros(n_people)

# Path to the folder where the CSV file is located
#folder = '/Users/albertodel-puerto/Documents/Draft-Paper02-Game/EEG Data/Cubism'  # Replace '/path/to/the/folder/where/the/subfolder/is' with the actual path
folder = ['/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Cubism/With Audio','/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Puzzling/With Audio','/Users/albert/Documents/Draft-Paper02-Game/EEG Data/Tetris/With Audio']
for i in range(n_game):
    for person in range(n_people):
    # Build the full path to the CSV file
        full_path = os.path.join(folder[i], f'game{i+1}_{person}.csv')
        eeg_signal_data_1 = np.genfromtxt(full_path, delimiter=',', skip_header=1,usecols=col1)
        eeg_signal_data_1 = np.nan_to_num(eeg_signal_data_1, nan=0)
        
        if len(eeg_signal_data_1) <= n_sample:
            aux = np.zeros(n_sample)
            aux[0:len(eeg_signal_data_1)] = eeg_signal_data_1
            eeg_signal_data_1 = aux
        else:
            eeg_signal_data_1 = eeg_signal_data_1[0:n_sample]
        
        # Apply the filter to the EEG signal
        eeg_beta_filtered_1 = butter_bandpass_filter(eeg_signal_data_1, lowcut, highcut, fs, order=4)


        ## calculation of the PSD of the filtered signal
        freq, psd = signal.welch(eeg_beta_filtered_1, fs=fs, nperseg=FFTsize)
        # Normalize the PSD
        psd_normalized = psd / np.sum(psd)

        # Calculate spectral entropy using Shannon entropy
        H_person= entropy(psd_normalized, base=2)
        
        spectral_entropy[i,person] = H_person  
print(spectral_entropy)

spectral_entropy_cubism = spectral_entropy[0,:]
spectral_entropy_puzzling = spectral_entropy[1,:]
spectral_entropy_tetris = spectral_entropy[2,:]

spectral_entropy_cubism = spectral_entropy_cubism[~np.isnan(spectral_entropy_cubism)]
spectral_entropy_puzzling = spectral_entropy_puzzling[~np.isnan(spectral_entropy_puzzling)]
spectral_entropy_tetris = spectral_entropy_tetris[~np.isnan(spectral_entropy_tetris)]

print('Final Results')
print("spectral entropy cubism",spectral_entropy_cubism)
print('spectral entropy puzzling',spectral_entropy_puzzling)
print('spectral entropy tetris', spectral_entropy_tetris)

H_cubism_mean = np.mean(spectral_entropy_cubism)
H_puzzling_mean = np.mean(spectral_entropy_puzzling)
H_tetris_mean = np.mean(spectral_entropy_tetris)

print('C_xy_cubis_mean: ',H_cubism_mean)
print('C_xy_puzzling_mean: ',H_puzzling_mean)
print('C_xy_tetris_mean', H_tetris_mean)

H_data = [spectral_entropy_cubism, spectral_entropy_puzzling, spectral_entropy_tetris]
labels = ['Cubism', 'Puzzling', 'Tetris']

# Plot the vectors in a box plot
box = plt.boxplot(H_data, positions=np.arange(1, len(labels) * 2, 2), labels=labels, widths=0.2, patch_artist=True, medianprops={'color':'red'})

# Calculate positions for the points
jittered_x_positions = np.arange(0.8, len(labels) * 2, 2)

# Add distribution points
for i, data in enumerate(H_data):
    y_positions = data
    x_positions = [jittered_x_positions[i]] * len(y_positions)
    plt.scatter(x_positions, y_positions, color='gray', alpha=0.5)

#plt.title('PSD Analysis')
#plt.xlabel('Games')
plt.rc('text', usetex=True)
plt.ylabel('$SpEn$')
plt.grid(True, linestyle='dotted')  # Add dotted grid to the plot
plt.show()
