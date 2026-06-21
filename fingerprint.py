

import re
import time
import numpy as np
from scipy.signal import spectrogram as _sp_spectrogram
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure


def pretty_title(name):
    s = name.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)  # camelCase -> camel Case
    s = re.sub(r"\s+", " ", s).strip()
    return s.title() if s == s.lower() or s == s.upper() else s


# --------------------------------------------------------------------------
# 1. Spectrogram
# --------------------------------------------------------------------------
def compute_spectrogram(signal, sr, window_seconds=0.025, overlap_ratio=0.5):
    nperseg = int(window_seconds * sr)
    nperseg = max(nperseg, 16)
    noverlap = int(nperseg * overlap_ratio)
    f, t, Sxx = _sp_spectrogram(
        signal, fs=sr, window="hann", nperseg=nperseg,
        noverlap=noverlap, mode="magnitude"
    )
    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db



def find_peaks_2d(Sxx_db, amp_min_db=-40, neighborhood=20):
    struct = generate_binary_structure(2, 1)
    neighborhood_mask = iterate_structure(struct, neighborhood)
    local_max = maximum_filter(Sxx_db, footprint=neighborhood_mask) == Sxx_db
    above_floor = Sxx_db > amp_min_db
    peaks_mask = local_max & above_floor
    freq_idx, time_idx = np.where(peaks_mask)
    peaks = list(zip(freq_idx, time_idx))
    return peaks


def peaks_to_constellation(peaks, f, t):
    return [(f[fi], t[ti]) for fi, ti in peaks]



def generate_paired_hashes(constellation, fan_value=8, min_dt=0.0, max_dt=4.0):
    constellation = sorted(constellation, key=lambda p: p[1])  # sort by time
    hashes = []
    n = len(constellation)
    for i in range(n):
        f1, t1 = constellation[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = constellation[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            h = (round(f1), round(f2), round(dt * 100))  # dt quantized to 10ms
            hashes.append((h, t1))
            count += 1
            if count >= fan_value:
                break
    return hashes


def generate_single_peak_hashes(constellation):
    
    hashes = []
    for f1, t1 in constellation:
        h = (round(f1),)
        hashes.append((h, t1))
    return hashes



class Database:
    MIN_VOTES = 3
    MIN_MARGIN = 1.5

    def __init__(self):
        self.index = {}        
        self.song_names = []
        self.song_meta = {}   

    def add_song(self, song_name, hash_list, constellation=None, duration=None):
        self.song_names.append(song_name)
        for h, t1 in hash_list:
            self.index.setdefault(h, []).append((song_name, t1))

        meta = {"hash_count": len(hash_list)}
        if duration is not None:
            meta["duration_s"] = duration
        if constellation is not None:
            meta["peak_count"] = len(constellation)
            thumb = constellation
            if len(thumb) > 1500:
                idx = np.linspace(0, len(thumb) - 1, 1500).astype(int)
                thumb = [thumb[i] for i in idx]
            meta["constellation_thumb"] = thumb
        self.song_meta[song_name] = meta

    def match(self, query_hash_list, top_k=3, return_timing=False):
        t_lookup_start = time.perf_counter()
        offsets_per_song = {}
        for h, t_query in query_hash_list:
            if h not in self.index:
                continue
            for song_name, t_db in self.index[h]:
                offset = round(t_db - t_query, 2)
                offsets_per_song.setdefault(song_name, []).append(offset)
        t_lookup_end = time.perf_counter()

        scores = {}
        best_offset = {}
        for song_name, offsets in offsets_per_song.items():
            offsets_arr = np.array(offsets)
            if len(offsets_arr) == 0:
                continue
            lo, hi = offsets_arr.min(), offsets_arr.max() + 0.05
            bins = np.arange(lo, hi + 0.05, 0.05)
            hist, edges = np.histogram(offsets_arr, bins=bins)
            peak_idx = np.argmax(hist)
            scores[song_name] = int(hist[peak_idx])
            best_offset[song_name] = edges[peak_idx]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_song = ranked[0][0] if ranked else None
        t_scoring_end = time.perf_counter()

        if return_timing:
            timing = {
                "lookup_ms": (t_lookup_end - t_lookup_start) * 1000,
                "scoring_ms": (t_scoring_end - t_lookup_end) * 1000,
            }
            return best_song, ranked[:top_k], offsets_per_song, best_offset, timing
        return best_song, ranked[:top_k], offsets_per_song, best_offset

    def identify(self, query_hash_list, top_k=5, min_votes=None, min_margin=None):
        """
        High-level match with a confidence threshold applied. Returns a dict:
          prediction   : song name, or None if no candidate is confident enough
          confident    : bool
          top_score    : winning vote count (0 if no candidates at all)
          margin       : top_score / runner_up_score (float('inf') if no runner-up)
          ranked       : top_k (song, score) list
          offsets_per_song, best_offset : for plotting
          timing       : {"lookup_ms":..., "scoring_ms":...}
        """
        min_votes = self.MIN_VOTES if min_votes is None else min_votes
        min_margin = self.MIN_MARGIN if min_margin is None else min_margin

        best_song, ranked, offsets_per_song, best_offset, timing = self.match(
            query_hash_list, top_k=top_k, return_timing=True
        )
        top_score = ranked[0][1] if ranked else 0
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        margin = (top_score / runner_up) if runner_up > 0 else float("inf")

        confident = (best_song is not None) and (top_score >= min_votes) and (margin >= min_margin)
        return {
            "prediction": best_song if confident else None,
            "confident": confident,
            "top_score": top_score,
            "margin": margin,
            "ranked": ranked,
            "offsets_per_song": offsets_per_song,
            "best_offset": best_offset,
            "timing": timing,
        }



def fingerprint_signal_timed(signal, sr, window_seconds=0.025, amp_min_db=-40,
                              neighborhood=20, fan_value=8, max_dt=4.0):
    """Same as fingerprint_signal(paired=True) but also returns a timing
    breakdown (ms) per stage, for the app's pipeline-timing display."""
    t0 = time.perf_counter()
    f, t, Sxx_db = compute_spectrogram(signal, sr, window_seconds=window_seconds)
    t1 = time.perf_counter()
    peaks_idx = find_peaks_2d(Sxx_db, amp_min_db=amp_min_db, neighborhood=neighborhood)
    constellation = peaks_to_constellation(peaks_idx, f, t)
    t2 = time.perf_counter()
    hashes = generate_paired_hashes(constellation, fan_value=fan_value, max_dt=max_dt)
    t3 = time.perf_counter()
    result = {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "peaks_idx": peaks_idx, "constellation": constellation,
        "hashes": hashes,
        "timing": {
            "spectrogram_ms": (t1 - t0) * 1000,
            "constellation_ms": (t2 - t1) * 1000,
            "hashing_ms": (t3 - t2) * 1000,
        },
        "shape": Sxx_db.shape,
    }
    return result


def fingerprint_signal(signal, sr, window_seconds=0.025, amp_min_db=-40,
                        neighborhood=20, fan_value=8, max_dt=4.0, paired=True):
    """Run spectrogram -> constellation -> hashes in one shot."""
    f, t, Sxx_db = compute_spectrogram(signal, sr, window_seconds=window_seconds)
    peaks_idx = find_peaks_2d(Sxx_db, amp_min_db=amp_min_db, neighborhood=neighborhood)
    constellation = peaks_to_constellation(peaks_idx, f, t)
    if paired:
        hashes = generate_paired_hashes(constellation, fan_value=fan_value, max_dt=max_dt)
    else:
        hashes = generate_single_peak_hashes(constellation)
    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "peaks_idx": peaks_idx, "constellation": constellation,
        "hashes": hashes,
    }
