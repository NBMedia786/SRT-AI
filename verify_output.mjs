// =============================================================================
// SRT-AI · Production verification report
// =============================================================================
// Usage: node verify_output.mjs <audio-file> <srt-file>
//
// Validates that the SRT output is production-ready for Premiere Pro:
//   1. SRT format strict-parses (CRLF, HH:MM:SS,mmm, monotonic, no overlap)
//   2. First cue lines up with first non-silent moment in audio (≤1.0s drift)
//   3. Last cue ends near actual end of speech (≤2.0s drift)
//   4. Every cue is within audio bounds (no negative, no overshoot)
//   5. Temporal coverage of speech (% of speech captured by cues)
//   6. Cue density distribution (sanity check — no dead zones)
//   7. Per-cue summary CSV for spot-checking the worst alignments
//
// Output: a human-readable report + a JSON file with cue-level diagnostics.
// =============================================================================

import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const [, , audioFile, srtFile] = process.argv;
if (!audioFile || !srtFile) {
    console.error('Usage: node verify_output.mjs <audio-file> <srt-file>');
    process.exit(2);
}
if (!existsSync(audioFile)) { console.error(`audio not found: ${audioFile}`); process.exit(2); }
if (!existsSync(srtFile))   { console.error(`srt not found:   ${srtFile}`); process.exit(2); }

const FFMPEG_BIN = 'ffmpeg';

// ─── helpers ──────────────────────────────────────────────────────────
function parseTimestamp(s) {
    const m = s.match(/^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
    if (!m) return NaN;
    return parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseInt(m[3]) + parseInt(m[4].padEnd(3, '0').slice(0, 3)) / 1000;
}

function fmtSec(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = (sec % 60).toFixed(3).padStart(6, '0');
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.replace('.', ',')}`;
}

function runFfmpeg(args) {
    return new Promise((resolve, reject) => {
        const proc = spawn(FFMPEG_BIN, args);
        let stderr = '';
        proc.stderr.on('data', d => { stderr += d.toString(); });
        proc.on('close', code => code === 0 ? resolve(stderr) : reject(new Error(`ffmpeg exit ${code}\n${stderr.slice(-500)}`)));
    });
}

function runFfprobeDuration(file) {
    return new Promise((resolve, reject) => {
        const proc = spawn('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file]);
        let out = '';
        proc.stdout.on('data', d => { out += d.toString(); });
        proc.on('close', code => code === 0 ? resolve(parseFloat(out.trim())) : reject(new Error(`ffprobe failed`)));
    });
}

// ─── SRT parser ───────────────────────────────────────────────────────
async function parseSRT(filePath) {
    const raw = await readFile(filePath, 'utf-8');
    const lineEnding = raw.includes('\r\n') ? 'CRLF' : (raw.includes('\n') ? 'LF' : 'NONE');

    const blocks = raw.replace(/\r\n/g, '\n').trim().split(/\n{2,}/);
    const cues = [];
    const issues = [];

    blocks.forEach((block, bi) => {
        const lines = block.split('\n').filter(l => l !== '');
        if (lines.length < 3) { issues.push(`Block ${bi + 1}: < 3 lines`); return; }
        const idx = parseInt(lines[0].trim());
        const tm  = lines[1].match(/^(\d{2}:\d{2}:\d{2},\d{3})\s-->\s(\d{2}:\d{2}:\d{2},\d{3})$/);
        if (!tm) { issues.push(`Block ${bi + 1}: bad timecode '${lines[1]}'`); return; }
        const start = parseTimestamp(tm[1]);
        const end   = parseTimestamp(tm[2]);
        const text  = lines.slice(2).join(' ').trim();
        cues.push({ idx, startSec: start, endSec: end, startStr: tm[1], endStr: tm[2], text });
        if (end <= start) issues.push(`Cue ${idx}: end <= start (${tm[1]} → ${tm[2]})`);
        if (!text)        issues.push(`Cue ${idx}: empty text`);
    });

    // monotonic index
    cues.forEach((c, i) => { if (c.idx !== i + 1) issues.push(`Cue index break at position ${i}: got ${c.idx}, expected ${i + 1}`); });
    // monotonic time
    for (let i = 1; i < cues.length; i++) {
        if (cues[i].startSec < cues[i - 1].endSec - 1e-6) {
            issues.push(`Cue ${cues[i].idx} overlaps cue ${cues[i - 1].idx} (${cues[i - 1].endStr} > ${cues[i].startStr})`);
        }
    }
    return { cues, issues, lineEnding };
}

// ─── ffmpeg silencedetect ────────────────────────────────────────────
async function detectSilences(audioPath, noiseDb = -35, minDur = 0.4) {
    const stderr = await runFfmpeg(['-i', audioPath, '-af', `silencedetect=noise=${noiseDb}dB:d=${minDur}`, '-f', 'null', '-']);
    const lines = stderr.split('\n');
    const silences = [];
    let pendingStart = null;
    for (const l of lines) {
        let m = l.match(/silence_start:\s*(-?[\d.]+)/);
        if (m) pendingStart = Math.max(0, parseFloat(m[1]));
        m = l.match(/silence_end:\s*([\d.]+)/);
        if (m && pendingStart !== null) {
            silences.push({ start: pendingStart, end: parseFloat(m[1]) });
            pendingStart = null;
        }
    }
    return silences;
}

// ─── analysis ─────────────────────────────────────────────────────────
function computeSpeechRegions(silences, duration) {
    // Speech = NOT silence. Stitch the complement.
    const speech = [];
    let cursor = 0;
    for (const s of silences) {
        if (s.start > cursor) speech.push({ start: cursor, end: s.start });
        cursor = s.end;
    }
    if (cursor < duration) speech.push({ start: cursor, end: duration });
    return speech;
}

function coverageStats(cues, speechRegions, duration) {
    const totalSpeechSec = speechRegions.reduce((acc, s) => acc + (s.end - s.start), 0);
    const totalCueSec    = cues.reduce((acc, c) => acc + (c.endSec - c.startSec), 0);

    // Overlap of cues with speech regions
    let speechCovered = 0;
    let cueInSilence  = 0;
    for (const c of cues) {
        for (const s of speechRegions) {
            const overlap = Math.max(0, Math.min(c.endSec, s.end) - Math.max(c.startSec, s.start));
            speechCovered += overlap;
        }
        const cueDur = c.endSec - c.startSec;
        // cueInSilence = how much of this cue's time is in a silent region
        let inSpeech = 0;
        for (const s of speechRegions) {
            inSpeech += Math.max(0, Math.min(c.endSec, s.end) - Math.max(c.startSec, s.start));
        }
        cueInSilence += Math.max(0, cueDur - inSpeech);
    }

    return { totalSpeechSec, totalCueSec, speechCovered, cueInSilence, audioDuration: duration };
}

// ─── main ─────────────────────────────────────────────────────────────
console.log('═'.repeat(72));
console.log('  SRT-AI · Production Verification Report');
console.log('═'.repeat(72));
console.log(`  Audio: ${audioFile}`);
console.log(`  SRT:   ${srtFile}`);
console.log();

const audioDuration = await runFfprobeDuration(audioFile);
console.log(`▸ Audio duration:   ${fmtSec(audioDuration)} (${audioDuration.toFixed(2)}s)`);

const { cues, issues, lineEnding } = await parseSRT(srtFile);
console.log(`▸ Cues in SRT:      ${cues.length}`);
console.log(`▸ Line endings:     ${lineEnding}  ${lineEnding === 'CRLF' ? '✓ Premiere-compatible' : '✗ Premiere may complain'}`);

console.log('\n── 1. Format / Structural validation ──');
if (issues.length === 0) {
    console.log('  ✓ No format issues (no overlap, no inversions, no missing indices, no empty text)');
} else {
    console.log(`  ✗ ${issues.length} issue(s):`);
    issues.slice(0, 10).forEach(i => console.log(`    - ${i}`));
    if (issues.length > 10) console.log(`    … and ${issues.length - 10} more`);
}

console.log('\n── 2. Bounds check ──');
const firstCue = cues[0];
const lastCue  = cues[cues.length - 1];
const negStart = cues.filter(c => c.startSec < 0).length;
const overEnd  = cues.filter(c => c.endSec > audioDuration + 0.5).length;
console.log(`  First cue:   ${firstCue?.startStr} (${firstCue?.startSec.toFixed(3)}s)`);
console.log(`  Last cue:    ${lastCue?.endStr}  (${lastCue?.endSec.toFixed(3)}s)`);
console.log(`  Audio end:   ${fmtSec(audioDuration)}`);
console.log(`  Cues with start < 0:    ${negStart} ${negStart ? '✗' : '✓'}`);
console.log(`  Cues with end > audio:  ${overEnd}  ${overEnd  ? '✗' : '✓'}`);

console.log('\n── 3. Speech alignment (silencedetect @ -35dB / 0.4s) ──');
console.log('  Running ffmpeg silencedetect (this takes ~30s on 26-min audio)...');
const silences = await detectSilences(audioFile);
const speechRegions = computeSpeechRegions(silences, audioDuration);
const firstSpeech = speechRegions[0]?.start ?? 0;
const lastSpeech  = speechRegions[speechRegions.length - 1]?.end ?? audioDuration;
console.log(`  Speech starts at:    ${fmtSec(firstSpeech)}`);
console.log(`  Speech ends at:      ${fmtSec(lastSpeech)}`);
console.log(`  Silence regions:     ${silences.length}`);
console.log(`  Total speech time:   ${(speechRegions.reduce((a, s) => a + (s.end - s.start), 0)).toFixed(1)}s (${(speechRegions.reduce((a, s) => a + (s.end - s.start), 0) / audioDuration * 100).toFixed(1)}% of audio)`);

const firstCueDrift = firstCue ? firstCue.startSec - firstSpeech : NaN;
const lastCueDrift  = lastCue  ? lastSpeech - lastCue.endSec     : NaN;
console.log(`\n  First cue vs first speech: ${firstCueDrift >= 0 ? '+' : ''}${firstCueDrift.toFixed(3)}s ${Math.abs(firstCueDrift) <= 1.0 ? '✓' : '⚠'}`);
console.log(`  Last cue vs last speech:   ${lastCueDrift  >= 0 ? '+' : ''}${lastCueDrift.toFixed(3)}s ${Math.abs(lastCueDrift)  <= 2.0 ? '✓' : '⚠'}`);
console.log('  (positive = SRT is later than speech, negative = earlier)');

console.log('\n── 4. Coverage analysis ──');
const cov = coverageStats(cues, speechRegions, audioDuration);
const speechCoveragePct = (cov.speechCovered / cov.totalSpeechSec) * 100;
const cueInSilencePct   = (cov.cueInSilence / cov.totalCueSec) * 100;
console.log(`  Total cue duration:        ${cov.totalCueSec.toFixed(1)}s`);
console.log(`  Speech covered by cues:    ${cov.speechCovered.toFixed(1)}s (${speechCoveragePct.toFixed(1)}% of all speech)`);
console.log(`  Cue time inside silence:   ${cov.cueInSilence.toFixed(1)}s (${cueInSilencePct.toFixed(1)}% of total cue time)`);
const coverageGrade = speechCoveragePct >= 80 ? '✓ excellent' :
                      speechCoveragePct >= 60 ? '○ acceptable' :
                                                '✗ poor — likely missing speech';
const silenceGrade  = cueInSilencePct <= 15 ? '✓ tight' :
                      cueInSilencePct <= 30 ? '○ loose' :
                                              '✗ cues drifting into silence';
console.log(`  ${coverageGrade} (coverage)`);
console.log(`  ${silenceGrade}  (precision)`);

console.log('\n── 5. Cue stats ──');
const durations = cues.map(c => c.endSec - c.startSec);
const gaps      = cues.slice(1).map((c, i) => c.startSec - cues[i].endSec);
const wpc = cues.map(c => c.text.split(/\s+/).filter(Boolean).length);
const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
console.log(`  Cue duration:    avg ${avg(durations).toFixed(2)}s · median ${med(durations).toFixed(2)}s · max ${Math.max(...durations).toFixed(2)}s · min ${Math.min(...durations).toFixed(2)}s`);
if (gaps.length) {
    console.log(`  Inter-cue gap:   avg ${avg(gaps).toFixed(2)}s · median ${med(gaps).toFixed(2)}s · max ${Math.max(...gaps).toFixed(2)}s`);
}
console.log(`  Words per cue:   avg ${avg(wpc).toFixed(1)} · max ${Math.max(...wpc)} · min ${Math.min(...wpc)}`);

console.log('\n── 6. Dead zones (≥10s of speech with no cue) ──');
const deadZones = [];
for (const sr of speechRegions) {
    // Find cue gaps inside this speech region
    const cuesInRegion = cues.filter(c => c.endSec > sr.start && c.startSec < sr.end)
        .map(c => ({ start: Math.max(c.startSec, sr.start), end: Math.min(c.endSec, sr.end) }))
        .sort((a, b) => a.start - b.start);
    let cursor = sr.start;
    for (const c of cuesInRegion) {
        if (c.start - cursor >= 10) deadZones.push({ start: cursor, end: c.start });
        cursor = Math.max(cursor, c.end);
    }
    if (sr.end - cursor >= 10) deadZones.push({ start: cursor, end: sr.end });
}
if (deadZones.length === 0) {
    console.log('  ✓ No dead zones — every 10+s span of speech has at least one cue');
} else {
    console.log(`  ⚠ ${deadZones.length} dead zone(s):`);
    deadZones.slice(0, 5).forEach(d => console.log(`    ${fmtSec(d.start)} → ${fmtSec(d.end)}  (${(d.end - d.start).toFixed(1)}s of speech with no cue)`));
    if (deadZones.length > 5) console.log(`    … and ${deadZones.length - 5} more`);
}

// ─── per-cue CSV for spot-checking ───────────────────────────────────
const csvLines = ['idx,start,end,duration_s,text,first_word_in_silence'];
for (const c of cues) {
    const inSilence = silences.some(s => c.startSec >= s.start - 0.05 && c.startSec <= s.end + 0.05);
    csvLines.push([
        c.idx,
        c.startStr,
        c.endStr,
        (c.endSec - c.startSec).toFixed(3),
        '"' + c.text.replace(/"/g, '""').substring(0, 60) + '"',
        inSilence ? 'YES' : '',
    ].join(','));
}
const csvPath = srtFile.replace(/\.srt$/i, '') + '_verification.csv';
await writeFile(csvPath, csvLines.join('\n'));
console.log(`\n▸ Per-cue CSV written: ${csvPath}`);

// ─── verdict ──────────────────────────────────────────────────────────
const verdict = {
    format_ok:    issues.length === 0,
    crlf:         lineEnding === 'CRLF',
    bounds_ok:    negStart === 0 && overEnd === 0,
    first_align:  Math.abs(firstCueDrift) <= 1.0,
    last_align:   Math.abs(lastCueDrift)  <= 2.0,
    coverage_ok:  speechCoveragePct >= 60,
    precision_ok: cueInSilencePct <= 30,
    no_deadzones: deadZones.length === 0,
};
const score = Object.values(verdict).filter(Boolean).length;
const total = Object.keys(verdict).length;

console.log('\n' + '═'.repeat(72));
console.log(`  VERDICT: ${score}/${total} checks passed`);
console.log('═'.repeat(72));
for (const [k, v] of Object.entries(verdict)) {
    console.log(`  ${v ? '✓' : '✗'}  ${k}`);
}
console.log();
if (score === total) {
    console.log('  ✓✓✓  PRODUCTION READY  ·  This SRT is Premiere Pro compatible and accurately aligned.');
} else if (score >= total - 1) {
    console.log('  ○  Mostly good — see flagged items above. Likely usable, may want to spot-check.');
} else {
    console.log('  ✗  Not production ready. Address the failed checks above.');
}
console.log();

// Also write a JSON summary
const summaryPath = srtFile.replace(/\.srt$/i, '') + '_verification.json';
await writeFile(summaryPath, JSON.stringify({
    audio: audioFile, srt: srtFile,
    audioDurationSec: audioDuration, cueCount: cues.length, lineEnding,
    issues, firstCueDrift, lastCueDrift,
    speechCoveragePct, cueInSilencePct, deadZones,
    verdict, score, total,
}, null, 2));
console.log(`▸ JSON summary written: ${summaryPath}`);
