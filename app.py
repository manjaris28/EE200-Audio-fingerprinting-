

import os
import io
import time
import pickle
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint import (
    compute_spectrogram, find_peaks_2d, peaks_to_constellation,
    fingerprint_signal_timed, Database, pretty_title,
)

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

DB_PATH = "fingerprint_db.pkl"
SONGS_DIR = "songs"
SAMPLES_DIR = "samples"


st.markdown("""
<style>
.stApp { background-color: #0b0f10; }
div[data-testid="stMetricValue"] { color: #2dd4bf; }
.eyebrow { letter-spacing: .12em; text-transform: uppercase; color: #6b7a78;
           font-size: 0.72rem; font-weight: 600; margin-bottom: 2px;}
.match-found { border: 1px solid #1f3d38; background: rgba(45,212,191,0.07);
               border-radius: 10px; padding: 18px 22px; margin: 10px 0 18px 0;}
.match-none { border: 1px solid #4a2a2a; background: rgba(220,80,80,0.07);
              border-radius: 10px; padding: 18px 22px; margin: 10px 0 18px 0;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
@st.cache_resource
def load_database():
    if not os.path.exists(DB_PATH):
        st.error(
            f"`{DB_PATH}` not found. Run `python build_database.py` first "
            f"(after placing the song library in `{SONGS_DIR}/`) and make "
            f"sure the .pkl file ships with the deployed app."
        )
        st.stop()
    with open(DB_PATH, "rb") as fh:
        obj = pickle.load(fh)
    return obj["db"], obj["sr"]


def load_audio_bytes(file_bytes, target_sr):
   
    sig, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)
    if sr != target_sr:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
    return sig


def list_sample_files():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav"))



db, db_sr = load_database()

st.markdown('<div class="eyebrow">SIGNALS, SYSTEMS &amp; NETWORKS</div>',
            unsafe_allow_html=True)
st.title("🎧 EE200: Audio Fingerprinting")
st.caption("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

tab_library, tab_identify, tab_batch = st.tabs(["📚 Library", "🔍 Identify", "📋 Batch"])


# TAB 1 — LIBRARY

with tab_library:
    st.markdown('<div class="eyebrow">Library</div>', unsafe_allow_html=True)
    st.info(
        "Song indexing is managed offline (`build_database.py`) — this tab is "
        "read-only. Drop a clip in the **Identify** tab to test the library.",
        icon="ℹ️",
    )
    st.markdown(f'<div class="eyebrow">In the database — {len(db.song_names)} songs</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, name in enumerate(sorted(db.song_names)):
        meta = db.song_meta.get(name, {})
        with cols[i % 4]:
            thumb = meta.get("constellation_thumb", [])
            fig, ax = plt.subplots(figsize=(2.6, 1.5))
            if thumb:
                tf = [p[0] for p in thumb]
                tt = [p[1] for p in thumb]
                ax.scatter(tt, tf, s=2, c="#2dd4bf", alpha=0.8)
            ax.set_facecolor("#11181a")
            fig.patch.set_facecolor("#11181a")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
            st.markdown(f"**{pretty_title(name)}**")
            st.caption(f"{meta.get('hash_count', 0):,} hashes")



# TAB 2 — IDENTIFY (single clip)

with tab_identify:
    st.markdown('<div class="eyebrow">Search</div>', unsafe_allow_html=True)
    st.subheader("Identify a clip")

    uploaded = st.file_uploader(
        "Upload a query clip", type=["wav", "flac", "mp3", "ogg"], key="single_upload"
    )

    st.markdown('<div class="eyebrow">Or try a sample</div>', unsafe_allow_html=True)
    sample_files = list_sample_files()
    if sample_files:
        for sf_name in sample_files:
            c1, c2, c3 = st.columns([1, 4, 1])
            c1.write(sf_name.replace(".wav", ""))
            c2.audio(os.path.join(SAMPLES_DIR, sf_name))
            if c3.button("Try", key=f"try_{sf_name}"):
                st.session_state["chosen_sample"] = sf_name
                st.session_state["single_upload_cleared"] = True
    else:
        st.caption("No bundled samples yet — run `python make_samples.py` to generate some.")

    query_bytes, query_label = None, None
    if uploaded is not None:
        query_bytes, query_label = uploaded.read(), uploaded.name
        st.session_state.pop("chosen_sample", None) 
    elif st.session_state.get("chosen_sample"):
        chosen_sample = st.session_state["chosen_sample"]
        with open(os.path.join(SAMPLES_DIR, chosen_sample), "rb") as fh:
            query_bytes, query_label = fh.read(), chosen_sample

    if query_label:
        st.caption(f"Selected query: **{query_label}**")

    run = st.button("Identify", type="primary", disabled=(query_bytes is None))

    if run and query_bytes is not None:
        with st.spinner("Running pipeline..."):
            sig = load_audio_bytes(query_bytes, db_sr)
            result = fingerprint_signal_timed(sig, db_sr)
            outcome = db.identify(result["hashes"])

        timing = {**result["timing"], **outcome["timing"]}
        total_ms = sum(timing.values())

        st.markdown("---")
        tcols = st.columns(6)
        tcols[0].metric("① Spectrogram", f"{timing['spectrogram_ms']:.0f} ms",
                         help=f"shape {result['shape']}")
        tcols[1].metric("② Constellation", f"{timing['constellation_ms']:.0f} ms",
                         help=f"{len(result['constellation'])} peaks")
        tcols[2].metric("③ Hashing", f"{timing['hashing_ms']:.0f} ms",
                         help=f"{len(result['hashes'])} hashes")
        tcols[3].metric("④ DB lookup", f"{timing['lookup_ms']:.0f} ms",
                         help=f"{len(db.song_names)} tracks")
        tcols[4].metric("⑤ Scoring", f"{timing['scoring_ms']:.0f} ms")
        tcols[5].metric("Total", f"{total_ms:.0f} ms")

        if outcome["confident"]:
            if outcome["margin"] == float("inf"):
                margin_html = '<b style="color:#e3a23a;">no rival candidates</b>'
            else:
                margin_html = f'<b style="color:#e3a23a;">{outcome["margin"]:.0f}×</b> the runner-up'
            st.markdown(
                f'<div class="match-found">'
                f'<div class="eyebrow">Match found</div>'
                f'<div style="font-size:2rem;font-weight:700;">{pretty_title(outcome["prediction"])}</div>'
                f'<div style="color:#9fb0ae;">cluster score <b style="color:#2dd4bf;">{outcome["top_score"]}</b>'
                f' &middot; {margin_html}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="match-none">'
                f'<div class="eyebrow">No confident match</div>'
                f'<div style="font-size:1.6rem;font-weight:700;">none</div>'
                f'<div style="color:#9fb0ae;">best candidate only scored '
                f'{outcome["top_score"]} votes — below the confidence threshold '
                f'(≥{db.MIN_VOTES} votes and ≥{db.MIN_MARGIN}× the runner-up).</div>'
                f'</div>', unsafe_allow_html=True)

        if outcome["ranked"]:
            st.markdown('<div class="eyebrow">Candidate scores</div>', unsafe_allow_html=True)
            cand_df = pd.DataFrame(
                [(pretty_title(s), sc) for s, sc in outcome["ranked"]],
                columns=["song", "votes"],
            )
            st.dataframe(cand_df, width='stretch', hide_index=True)

       
        st.markdown("---")
        st.markdown('<div class="eyebrow">Step 1 · Feature extraction</div>', unsafe_allow_html=True)
        st.markdown("**From spectrogram to constellation**")
        st.caption(
            f"The clip became a time-frequency map (left); brighter = louder at that "
            f"frequency and moment. Only the **{len(result['constellation'])} most "
            f"prominent peaks** were kept (right) — that sparsity is what makes the "
            f"fingerprint robust to EQ, volume changes, and mild noise."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.pcolormesh(result["t"], result["f"], result["Sxx_db"],
                          shading="gouraud", vmin=-80, vmax=0, cmap="magma")
            ax.set_ylim(0, 5000); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            if result["constellation"]:
                cf = [p[0] for p in result["constellation"]]
                ct = [p[1] for p in result["constellation"]]
                ax.scatter(ct, cf, s=10, c="#2dd4bf")
            ax.set_ylim(0, 5000); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
            st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="eyebrow">Step 2 · The proof</div>', unsafe_allow_html=True)
        st.markdown("**The alignment spike**")
        st.caption(
            "Every matched hash votes for a time offset (database time minus query time). "
            "Chance matches scatter their votes randomly, forming a flat noise floor. A "
            "genuine match makes them converge to a single spike — that convergence is "
            "the actual match decision, not a similarity score."
        )
        top_songs = sorted(outcome["offsets_per_song"].items(), key=lambda kv: -len(kv[1]))[:5]
        if top_songs:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            colors = plt.cm.tab10.colors
            for i, (song, offs) in enumerate(top_songs):
                ax.hist(offs, bins=60, alpha=0.7,
                        label=("★ " if song == outcome["prediction"] else "") + pretty_title(song),
                        color=colors[i % len(colors)])
            ax.set_xlabel("candidate offset (s)"); ax.set_ylabel("votes")
            ax.legend(fontsize=8)
            if outcome["confident"]:
                ax.annotate(f"{outcome['top_score']} hashes\nalign here",
                            xy=(outcome["best_offset"].get(outcome["prediction"], 0), outcome["top_score"]),
                            xytext=(15, -15), textcoords="offset points",
                            color="#e3a23a", fontsize=9,
                            arrowprops=dict(arrowstyle="->", color="#e3a23a"))
            st.pyplot(fig); plt.close(fig)
        else:
            st.caption("No hashes from this clip occurred anywhere in the database.")



# TAB 3 — BATCH

with tab_batch:
    st.markdown('<div class="eyebrow">Batch</div>', unsafe_allow_html=True)
    st.subheader("Identify many clips at once")
    st.caption(
        "Upload a set of query clips. Each is identified against the currently indexed "
        "library, and the results are written to a standardised `results.csv` with columns "
        "`filename, prediction`. `prediction` is the matched track's filename without its "
        f"extension, or **none** when no candidate clears the confidence threshold "
        f"(≥{db.MIN_VOTES} votes, ≥{db.MIN_MARGIN}× the runner-up)."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips",
        type=["wav", "flac", "mp3", "ogg", "zip"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if uploaded_files and st.button("Run batch", type="primary"):
        rows = []
        progress = st.progress(0.0)

        jobs = []
        for uf in uploaded_files:
            if uf.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                    for zi in zf.infolist():
                        if zi.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                            jobs.append((os.path.basename(zi.filename), zf.read(zi)))
            else:
                jobs.append((uf.name, uf.read()))

        n_none = 0
        for i, (fname, raw) in enumerate(jobs):
            try:
                sig = load_audio_bytes(raw, db_sr)
                result = fingerprint_signal_timed(sig, db_sr)
                outcome = db.identify(result["hashes"])
                prediction = outcome["prediction"] if outcome["confident"] else "none"
                if prediction == "none":
                    n_none += 1
            except Exception as e:
                prediction = "none"
                n_none += 1
                st.warning(f"Failed on {fname}: {e}")
            rows.append({"filename": fname, "prediction": prediction})
            progress.progress((i + 1) / len(jobs))

        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.markdown('<div class="eyebrow">Results</div>', unsafe_allow_html=True)
        st.dataframe(results_df, width='stretch', hide_index=True)
        st.caption(f"{len(jobs) - n_none} / {len(jobs)} clips matched to a track "
                   f"({n_none} returned `none`).")

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download results.csv", data=csv_bytes,
            file_name="results.csv", mime="text/csv",
        )
        results_df.to_csv("results.csv", index=False)
