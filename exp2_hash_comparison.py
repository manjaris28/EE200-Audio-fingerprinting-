import os
import pickle
import soundfile as sf
import matplotlib.pyplot as plt

from fingerprint import (
    fingerprint_signal,
    Database
)

QUERY_FILE = "samples/sample2.wav"

with open("fingerprint_db.pkl", "rb") as fh:
    obj = pickle.load(fh)

db_paired = obj["db"]

print("Building single-peak database...")

db_single = Database()

for fname in sorted(os.listdir("songs")):

    if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        continue

    song_name = os.path.splitext(fname)[0]

    sig, sr = sf.read(os.path.join("songs", fname))

    if len(sig.shape) > 1:
       sig = sig.mean(axis=1)

    print(fname, sig.shape, sr)

    result = fingerprint_signal(
        sig,
        sr,
        paired=False
    )

    db_single.add_song(
        song_name,
        result["hashes"]
    )

print("Single database ready.")

query_sig, query_sr = sf.read(QUERY_FILE)


paired_query = fingerprint_signal(
    query_sig,
    query_sr,
    paired=True
)

single_query = fingerprint_signal(
    query_sig,
    query_sr,
    paired=False
)

paired_result = db_paired.identify(
    paired_query["hashes"]
)

single_result = db_single.identify(
    single_query["hashes"]
)

print("\nPAIRED")
print("Prediction:", paired_result["prediction"])

print("\nSINGLE")
print("Prediction:", single_result["prediction"])

paired_offsets = paired_result["offsets_per_song"]
single_offsets = single_result["offsets_per_song"]


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


top_paired = dict(
    paired_result["ranked"][:5]
)

for song, offsets in paired_offsets.items():

    if song in top_paired and len(offsets) > 0:

        axes[0].hist(
            offsets,
            bins=40,
            alpha=0.6,
            label=song
        )

axes[0].set_title("Paired Hashes")
axes[0].set_xlabel("Candidate Offset")
axes[0].set_ylabel("Votes")
axes[0].legend()


top_single = dict(
    single_result["ranked"][:5]
)

for song, offsets in single_offsets.items():

    if song in top_single and len(offsets) > 0:

        axes[1].hist(
            offsets,
            bins=40,
            alpha=0.6,
            label=song
        )

axes[1].set_title("Single Peaks")
axes[1].set_xlabel("Candidate Offset")
axes[1].set_ylabel("Votes")
axes[1].legend()

plt.tight_layout()

os.makedirs("figures", exist_ok=True)

plt.savefig(
    "figures/exp2_single_vs_paired.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print("figures/exp2_single_vs_paired.png")