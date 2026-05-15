#!/usr/bin/env python3
# Source-separation helper for SRT-AI.
#
# Splits an input audio file into:
#   - vocals.wav      (isolated speech/singing)
#   - no_vocals.wav   (everything else — music, ambient, body-cam, sfx)
#
# Used by the transcription pipeline to isolate narrator voice from
# overlapping body-cam / music / background so STT and Gemini get a clean
# signal and produce tighter timestamps.
#
# Usage:
#   python3 separate_vocals.py <input_audio> <output_dir>
#
# Output files written to <output_dir>:
#   - vocals.wav      (mono, 44.1kHz, PCM s16le)
#   - no_vocals.wav   (mono, 44.1kHz, PCM s16le)
#
# Demucs models are downloaded automatically on first run (~2GB) and cached
# under ~/.cache/torch/hub/checkpoints. After first download, runs offline.

import sys
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def fail(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, **kw):
    """Run cmd, raise on non-zero exit. Returns stdout."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout


def main():
    if len(sys.argv) != 3:
        fail("Usage: separate_vocals.py <input_audio> <output_dir>")

    input_audio = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()

    if not input_audio.exists():
        fail(f"input audio not found: {input_audio}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — only fails if demucs isn't installed
    try:
        import demucs.separate
    except ImportError:
        fail(
            "demucs not installed. Run:\n"
            "  python3 -m pip install --user demucs\n"
            "(downloads ~2GB of models on first separation run)"
        )

    # Demucs writes its output under <model_name>/<input_stem>/ — we use a
    # temp dir so we can flatten and rename to predictable filenames.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # `--two-stems vocals` is the lighter mode: vocals + everything else.
        # `-n htdemucs` is the default 2023 model — best quality/speed balance.
        # `--mp3-bitrate` ignored when --wav specified; we want WAV PCM for
        # downstream FFmpeg/STT compatibility.
        try:
            demucs.separate.main([
                "--two-stems", "vocals",
                "-n", "htdemucs",
                "--out", str(tmp_path),
                "--filename", "{track}__{stem}.{ext}",
                str(input_audio),
            ])
        except SystemExit as e:
            # demucs.separate.main calls sys.exit(0) on success
            if e.code not in (None, 0):
                raise

        stem = input_audio.stem
        # Default output layout: <out>/<model>/<stem>__<stem-name>.<ext>
        # With --filename above, files are: <out>/<model>/<stem>__vocals.wav
        # and <out>/<model>/<stem>__no_vocals.wav
        model_dir = tmp_path / "htdemucs"
        candidates = list(model_dir.glob(f"{stem}__*"))
        if len(candidates) < 2:
            fail(f"unexpected demucs output, found: {candidates}")

        for src in candidates:
            if "vocals" in src.name and "no_vocals" not in src.name:
                shutil.copy(src, out_dir / "vocals.wav")
            elif "no_vocals" in src.name:
                shutil.copy(src, out_dir / "no_vocals.wav")

    # Sanity check
    vocals = out_dir / "vocals.wav"
    other = out_dir / "no_vocals.wav"
    if not vocals.exists() or not other.exists():
        fail(f"expected output files missing in {out_dir}")

    print(f"OK\nvocals={vocals}\nno_vocals={other}")


if __name__ == "__main__":
    main()
