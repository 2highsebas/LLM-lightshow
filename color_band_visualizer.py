import numpy as np
import librosa, sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

SR = 16000
HOP = 512
NUM_BANDS = 8
N_FFT = 1024
COLORS = plt.cm.plasma(np.linspace(0, 1, NUM_BANDS))

def compute_bands(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    edges = np.geomspace(20, 16000, num=NUM_BANDS + 1)
    bands = []
    for i in range(NUM_BANDS):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        bands.append(np.mean(S[idx, :], axis=0) if len(idx) else np.zeros(S.shape[1]))
    B = np.vstack(bands)
    B /= np.max(B) + 1e-9
    return B

# === load and preprocess song ===
y, _ = librosa.load("songs/002.wav", sr=SR, mono=True)
y = librosa.util.normalize(y)
bands = compute_bands(y)
bands /= np.max(bands)

frames = bands.shape[1]
frame_dur = HOP / SR  # ≈ 0.032 s per frame

fig, ax = plt.subplots()
bars = ax.bar(range(NUM_BANDS), bands[:, 0], color=COLORS)
ax.set_ylim(0, 1)
ax.set_xticks(range(NUM_BANDS))
ax.set_xticklabels([f"B{i}" for i in range(NUM_BANDS)])
ax.set_ylabel("Amplitude")
ax.set_title("Perfectly Synced 8-Band Visualizer")

# track start time to sync with sounddevice playback
start_time = [None]

def update(_):
    if start_time[0] is None:
        return bars  # not started yet

    # get elapsed playback time
    elapsed = time.perf_counter() - start_time[0]
    current_frame = int(elapsed / frame_dur)
    if current_frame >= frames:
        plt.close(fig)
        return bars

    amps = bands[:, current_frame]
    for b, val in zip(bars, amps):
        b.set_height(val)
        b.set_color(plt.cm.plasma(val))
    return bars

def start_playback():
    start_time[0] = time.perf_counter()
    sd.play(y, SR)

ani = FuncAnimation(fig, update, interval=frame_dur * 1000, blit=False)
start_playback()

plt.show()
sd.wait()
