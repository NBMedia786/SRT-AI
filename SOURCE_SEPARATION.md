# Source Separation (Optional)

Splits the **vocals** (narrator/speech) from the **background** (music,
body-cam, sfx, ambient noise) **before** transcription runs. The downstream
STT and Gemini pipelines then see an isolated vocal track, which dramatically
tightens timestamps on overlay audio — documentary narration over body-cam,
voice-over-music, dialog over sound effects, etc.

## When to enable

| Audio type | Without separation | With separation |
|---|---|---|
| Single speaker, clean | ±300ms (already good) | ±100ms |
| Narrator over music | ±2s drift | ±200ms |
| **Narrator over body-cam / multi-speaker overlay** | **±5s, body-cam dialogue missing** | **±200ms** |
| Music with vocals | misses or mis-times | clean vocal SRT |

If your audio is clean single-speaker, separation gives a modest improvement
but isn't required. Enable it whenever the audio contains overlapping voices
or background music/effects.

## Install

One-time setup (~2GB model download on first transcription that uses it):

```bash
python3 -m pip install --user demucs
```

That's it — Demucs models are downloaded on demand and cached under
`~/.cache/torch/hub/checkpoints`. After first download the pipeline runs
fully offline for the separation step.

## Enable

In `.env`:

```env
ENABLE_SOURCE_SEPARATION=true
```

Restart the server. On startup you'll see:

```
🔊 Source separation: enabled (Demucs ready)
```

If the env var is `true` but Demucs isn't installed, the server logs a
warning and falls back to the original audio per request (graceful degrade,
no failed transcriptions).

## Performance

| Hardware | 22-min audio | 1-hr audio |
|---|---|---|
| CPU only (typical laptop) | ~30-60s added | ~2-3 min added |
| NVIDIA GPU (CUDA) | ~5-10s added | ~20-30s added |

This is preprocessing time added to each transcription. No GPU required —
CPU mode works, just slower. Demucs auto-detects GPU if available.

## Cost

**Free.** Demucs runs locally — zero API calls, zero per-minute charges.
After the one-time model download, no internet needed for separation.

In fact, the separated audio is *cleaner* than the original for the
downstream Gemini and STT calls, which sometimes reduces API costs by
producing fewer hallucinated cues.

## How it fits in the pipeline

```
Input audio (.wav)
       ↓
[Pre-process: transcode if needed]
       ↓
[Source separation (Demucs)]  ← only when ENABLE_SOURCE_SEPARATION=true
       ↓
vocals.wav (isolated narrator/speech)
       ↓
[GCS upload]
       ↓
[STT (chirp_2)] + [Gemini transcription]   ← both now see clean signal
       ↓
[LCS word-anchor alignment]
       ↓
Final SRT
```

The separation step replaces `audioPath` with `vocals.wav` in the
transcription job. `no_vocals.wav` is generated as a side-effect (could be
used later for body-cam transcription if desired) but currently discarded
along with the temp workdir at job end.

## Troubleshooting

**"demucs not installed" in logs**
Install with `python3 -m pip install --user demucs`. Restart server.

**"Python interpreter not found"**
Server expects `python3` on PATH. Override with `SRT_PYTHON_BIN=/path/to/python3` in `.env`.

**First transcription with separation enabled is unusually slow**
Demucs downloads ~2GB of model weights on first use. Subsequent runs are
fast (~30-60s on CPU).

**Out-of-memory error during separation**
Demucs uses ~3-4GB RAM for default `htdemucs` model. On a low-RAM machine,
either disable separation or use a smaller model (modify `scripts/separate_vocals.py`,
change `-n htdemucs` to `-n htdemucs_ft` or similar lighter variant).

**Want body-cam dialogue transcribed too**
The `no_vocals.wav` track contains body-cam / background dialog. To capture
it, run a second transcription pass on that track and merge cues. Not
implemented by default; the current pipeline drops `no_vocals.wav`.
