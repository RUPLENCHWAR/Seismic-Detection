import csv
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Step 1: Load Data (CSV reading without pandas)
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if there is one
        for row in reader:
            # Assuming the first column is the seismic signal and the second is the label (0 or 1)
            data.append(float(row[0]))  # Seismic signal
            labels.append(int(row[1]))  # Quake label (1) or noise (0)
    return data, labels

# Step 2: Signal Preprocessing using Bandpass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Example: Preprocess seismic signal
def preprocess_signal(signal, fs=20.0, lowcut=0.5, highcut=10.0):
    filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)
    return filtered_signal

# Step 3: Simple Event Detection Algorithm (Threshold-Based)
def detect_events(signal, threshold=0.5):
    events = []
    for i, val in enumerate(signal):
        if abs(val) > threshold:  # If the signal exceeds a certain threshold, consider it a potential quake
            events.append(i)
    return events

# Load and preprocess example data
file_path = 'train_seismic.csv'  # Replace with your actual file path
signal_data, quake_labels = load_data(file_path)

# Filter the signal to remove noise
filtered_signal = preprocess_signal(signal_data)

# Detect seismic events using a simple threshold-based method
detected_events = detect_events(filtered_signal, threshold=0.5)

# Plot the filtered signal and mark detected events
plt.plot(filtered_signal, label='Filtered Signal')
for event in detected_events:
    plt.axvline(x=event, color='red', linestyle='--', label='Detected Event' if event == detected_events[0] else "")

plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.title('Seismic Signal with Detected Events')
plt.legend()
plt.show()

# Simple accuracy check (compare detected events with actual quake labels)
correct_detections = sum(1 for i in detected_events if quake_labels[i] == 1)
print(f"Correct detections: {correct_detections}/{len(detected_events)}")
