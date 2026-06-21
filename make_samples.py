

import os
import random
import soundfile as sf
import numpy as np

SONGS_DIR = "songs"
SAMPLES_DIR = "samples"
N_SAMPLES = 5
CLIP_SECONDS = 12

random.seed(7)


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for f in os.listdir(SAMPLES_DIR):
        if f.startswith("sample") and f.endswith(".wav"):
            os.remove(os.path.join(SAMPLES_DIR, f))

    song_files = sorted(
        f for f in os.listdir(SONGS_DIR)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
    )
    if not song_files:
        print(f"No songs found in {SONGS_DIR}/ -- nothing to sample from.")
        return

    chosen = random.sample(song_files, k=min(N_SAMPLES, len(song_files)))
    manifest = []

    for i, fname in enumerate(chosen, start=1):
        sig, sr = sf.read(os.path.join(SONGS_DIR, fname), always_2d=False)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        duration = len(sig) / sr
        clip_len = min(CLIP_SECONDS, max(duration - 0.5, 1.0))
        start_max = max(duration - clip_len, 0)
        start = random.uniform(0, start_max) if start_max > 0 else 0
        clip = sig[int(start * sr): int((start + clip_len) * sr)]

        out_name = f"sample{i}.wav"
        sf.write(os.path.join(SAMPLES_DIR, out_name), clip, sr)
        true_song = os.path.splitext(fname)[0]
        manifest.append((out_name, true_song, round(start, 1), round(clip_len, 1)))
        print(f"{out_name}  <-  {fname}  (from {start:.1f}s, {clip_len:.1f}s long)")

    with open(os.path.join(SAMPLES_DIR, "_manifest.txt"), "w") as fh:
        fh.write("# sample_file, true_song, start_s, length_s -- for your own reference only,\n")
        fh.write("# not read by the app.\n")
        for row in manifest:
            fh.write(",".join(str(x) for x in row) + "\n")

    print(f"\n{len(chosen)} sample clips written to {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
