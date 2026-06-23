import os
import librosa
import matplotlib.pyplot as plt

from fingerprint import compute_spectrogram

audio_path = "samples/sample2.wav"  

signal, sr = librosa.load(audio_path, sr=None, mono=True)

window_lengths = [0.005, 0.025, 0.1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, w in zip(axes, window_lengths):
    f, t, Sxx_db = compute_spectrogram(
        signal,
        sr,
        window_seconds=w
    )

    ax.imshow(
        Sxx_db,
        aspect="auto",
        origin="lower",
        extent=[t.min(), t.max(), f.min(), f.max()]
    )

    ax.set_title(f"Window = {int(w*1000)} ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

plt.tight_layout()

os.makedirs("figures", exist_ok=True)

plt.savefig(
    "figures/window_length_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()