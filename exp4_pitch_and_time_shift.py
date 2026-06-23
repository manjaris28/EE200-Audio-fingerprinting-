import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa

from fingerprint import fingerprint_signal

QUERY_FILE = "samples/sample2.wav"  
DB_FILE = "fingerprint_db.pkl"

PITCH_STEPS = [0, 0.25, 0.5, 1, 2, 3, 4, 5]
STRETCH_FACTORS = [1.00, 1.01, 1.02, 1.05, 1.10, 1.20]

with open(DB_FILE, "rb") as fh:
    data = pickle.load(fh)

db = data["db"]

signal, sr = librosa.load(
    QUERY_FILE,
    sr=None,
    mono=True
)

pitch_votes = []
pitch_runner = []

for step in PITCH_STEPS:

    shifted = librosa.effects.pitch_shift(
        signal,
        sr=sr,
        n_steps=step
    )

    fp = fingerprint_signal(
        shifted,
        sr,
        paired=True
    )

    result = db.identify(fp["hashes"])

    ranked = result["ranked"]

    top_score = ranked[0][1] if len(ranked) > 0 else 0
    runner_score = ranked[1][1] if len(ranked) > 1 else 0

    pitch_votes.append(top_score)
    pitch_runner.append(runner_score)

    print(
        f"Pitch +{step:>4} semitones | "
        f"votes={top_score}"
    )


stretch_votes = []
stretch_runner = []

for rate in STRETCH_FACTORS:

    stretched = librosa.effects.time_stretch(
        signal,
        rate=rate
    )

    fp = fingerprint_signal(
        stretched,
        sr,
        paired=True
    )

    result = db.identify(fp["hashes"])

    ranked = result["ranked"]

    top_score = ranked[0][1] if len(ranked) > 0 else 0
    runner_score = ranked[1][1] if len(ranked) > 1 else 0

    stretch_votes.append(top_score)
    stretch_runner.append(runner_score)

    print(
        f"Stretch {rate:>4.2f}x | "
        f"votes={top_score}"
    )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))


axes[0].plot(
    PITCH_STEPS,
    pitch_votes,
    "o-",
    color="tab:blue",
    label="best-match votes"
)

axes[0].plot(
    PITCH_STEPS,
    pitch_runner,
    "s--",
    color="gray",
    label="runner-up votes"
)

axes[0].set_xlabel("pitch shift (semitones up)")
axes[0].set_ylabel("votes")

axes[0].set_title(
    "Pitch shift quickly destroys the match"
)

axes[0].legend()
axes[0].grid(alpha=0.3)


axes[1].plot(
    STRETCH_FACTORS,
    stretch_votes,
    "o-",
    color="tab:purple",
    label="best-match votes"
)

axes[1].plot(
    STRETCH_FACTORS,
    stretch_runner,
    "s--",
    color="gray",
    label="runner-up votes"
)

axes[1].set_xlabel(
    "time-stretch rate (1.0 = unchanged)"
)

axes[1].set_ylabel("votes")

axes[1].set_title(
    "Time stretch also destroys the match"
)

axes[1].legend()
axes[1].grid(alpha=0.3)


fig.suptitle(
    "Pitch/time-warp robustness",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    "figures/exp4_pitch_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()