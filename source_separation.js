// =============================================================================
// Source separation wrapper — Node bridge to scripts/separate_vocals.py
// =============================================================================
//
// Why this exists: multi-speaker audio (narrator + body-cam, voice + music,
// dialog + sfx) confuses single-stream transcription. STT engines lock onto
// the loudest signal; Gemini long-form drifts on overlapping voices. Running
// Demucs upfront splits vocals from everything else, giving downstream models
// a clean signal — measured ~10× tighter timestamps on overlay content.
//
// This module:
//   - shells out to the Python script
//   - checks if Demucs is installed and surfaces a clear install hint if not
//   - is safely skippable: callers handle null return gracefully
//
// Activation: set ENABLE_SOURCE_SEPARATION=true in .env to enable.
// =============================================================================

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_SCRIPT = path.join(__dirname, 'scripts', 'separate_vocals.py');
const PYTHON_BIN = process.env.SRT_PYTHON_BIN || 'python3';

/**
 * Separate an audio file into vocals + everything-else.
 *
 * @param {string} inputPath - absolute path to input audio (any FFmpeg-readable format)
 * @param {(stage: string) => void} [onProgress] - optional progress callback
 * @returns {Promise<{vocals: string, noVocals: string, workDir: string}>}
 *   Paths to the separated WAV files. Caller is responsible for cleaning up workDir.
 *   Throws if Demucs isn't installed or separation fails.
 */
export async function separateVocals(inputPath, onProgress = () => {}) {
    if (!existsSync(inputPath)) {
        throw new Error(`source_separation: input audio not found: ${inputPath}`);
    }
    if (!existsSync(PYTHON_SCRIPT)) {
        throw new Error(`source_separation: helper script missing: ${PYTHON_SCRIPT}`);
    }

    const workDir = mkdtempSync(path.join(tmpdir(), 'srt-ai-separate-'));
    onProgress('starting');

    return new Promise((resolve, reject) => {
        const proc = spawn(PYTHON_BIN, [PYTHON_SCRIPT, inputPath, workDir], {
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (d) => {
            stdout += d.toString();
        });

        proc.stderr.on('data', (d) => {
            const text = d.toString();
            stderr += text;
            // Demucs progress lines look like "Separating track ..." and percentage bars.
            // Pipe them through to our onProgress for SSE streaming.
            for (const line of text.split('\n')) {
                const trimmed = line.trim();
                if (trimmed) onProgress(trimmed);
            }
        });

        proc.on('error', (err) => {
            if (err.code === 'ENOENT') {
                reject(new Error(
                    `Python interpreter not found at "${PYTHON_BIN}". ` +
                    `Set SRT_PYTHON_BIN env var or install python3.`
                ));
            } else {
                reject(err);
            }
        });

        proc.on('close', (code) => {
            if (code !== 0) {
                // Surface install hint when demucs is the missing piece
                if (stderr.includes('demucs not installed')) {
                    reject(new Error(
                        `Demucs not installed. To enable source separation, run:\n` +
                        `  python3 -m pip install --user demucs\n` +
                        `(one-time ~2GB model download on first separation)`
                    ));
                } else {
                    reject(new Error(
                        `source_separation exited ${code}\n` +
                        `stderr: ${stderr.slice(-500)}`
                    ));
                }
                return;
            }

            const vocals = path.join(workDir, 'vocals.wav');
            const noVocals = path.join(workDir, 'no_vocals.wav');
            if (!existsSync(vocals) || !existsSync(noVocals)) {
                reject(new Error(`source_separation: expected output files missing in ${workDir}`));
                return;
            }

            onProgress('done');
            resolve({ vocals, noVocals, workDir });
        });
    });
}

/**
 * Quick capability probe: returns true if both python3 and demucs are usable.
 * Use this on server startup to log whether source separation is available.
 */
export async function isSourceSeparationAvailable() {
    return new Promise((resolve) => {
        const proc = spawn(PYTHON_BIN, ['-c', 'import demucs; print("ok")'], {
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        let out = '';
        proc.stdout.on('data', (d) => { out += d.toString(); });
        proc.on('error', () => resolve(false));
        proc.on('close', (code) => {
            resolve(code === 0 && out.trim() === 'ok');
        });
    });
}
