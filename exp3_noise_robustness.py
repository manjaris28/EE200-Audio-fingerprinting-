import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa

from fingerprint import fingerprint_signal


QUERY_FILE = "samples/sample2.wav"     
DB_FILE = "fingerprint_db.pkl"

# noise levels
SNR_LEVELS = [30, 25, 20, 15, 10, 5, 0, -5, -10, -15]

EXPECTED_SONG = "Don_t Stop Me Now"     

with open(DB_FILE, "rb") as fh:
    data = pickle.load(fh)

db = data["db"]

signal, sr = librosa.load(
    QUERY_FILE,
    sr=None,
    mono=True
)

def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)

    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        signal.shape
    )

    noisy_signal = signal + noise

    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

    return noisy_signal


vote_counts = []
correct_flags = []
margins = []

for snr in SNR_LEVELS:

    noisy_signal = add_noise(signal, snr)

    fp = fingerprint_signal(
        noisy_signal,
        sr,
        paired=True
    )

    result = db.identify(fp["hashes"])

    prediction = result["prediction"]
    score = result["top_score"]
    margin = result["margin"]

    vote_counts.append(score)
    margins.append(max(margin, 1e-2))

    correct_flags.append(
        int(prediction == EXPECTED_SONG)
    )

    print(
        f"SNR={snr:>3} dB | "
        f"Prediction={prediction} | "
        f"Votes={score} | "
        f"Margin={margin:.2f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]

ax1.plot(
    SNR_LEVELS,
    vote_counts,
    "o-",
    color="tab:blue",
    linewidth=2,
    label="Top match votes"
)

ax1.set_xlabel("SNR (dB)")
ax1.set_ylabel("Top match vote count", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(alpha=0.3)

ax1.invert_xaxis()

ax2 = ax1.twinx()

ax2.plot(
    SNR_LEVELS,
    correct_flags,
    "r--s",
    linewidth=2,
    label="Correct prediction"
)

ax2.set_ylabel("Prediction correct", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(-0.05, 1.05)

axes[1].plot(
    SNR_LEVELS,
    margins,
    "o-",
    color="tab:green",
    linewidth=2
)

axes[1].set_yscale("log")
axes[1].invert_xaxis()

axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("Confidence margin")

axes[1].set_title(
    "Match confidence collapses as noise grows"
)

axes[1].grid(alpha=0.3)

fig.suptitle(
    f"Noise-robustness sweep: query = '{EXPECTED_SONG}'",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    "figures/exp3_noise_robustness.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
