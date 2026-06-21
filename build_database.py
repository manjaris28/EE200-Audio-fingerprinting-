

import os
import pickle
import soundfile as sf
import numpy as np
from fingerprint import fingerprint_signal, Database

SONGS_DIR = "songs"
DB_PATH = "fingerprint_db.pkl"


def load_mono(path, target_sr=22050):
    sig, sr = sf.read(path, always_2d=False)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != target_sr:
        
        import librosa
        sig = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return sig.astype(np.float32), sr


def build():
    db = Database()
    sr_used = None
    for fname in sorted(os.listdir(SONGS_DIR)):
        if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
            continue
        song_name = os.path.splitext(fname)[0]  
        sig, sr = load_mono(os.path.join(SONGS_DIR, fname))
        sr_used = sr
        result = fingerprint_signal(sig, sr, paired=True)
        duration_s = len(sig) / sr
        db.add_song(song_name, result["hashes"], constellation=result["constellation"],
                    duration=duration_s)
        print(f"indexed {song_name}: {len(result['constellation'])} peaks, "
              f"{len(result['hashes'])} hashes, {duration_s:.1f}s")

    with open(DB_PATH, "wb") as fh:
        pickle.dump({"db": db, "sr": sr_used}, fh)
    print(f"\nSaved database with {len(db.song_names)} songs -> {DB_PATH}")


if __name__ == "__main__":
    build()
