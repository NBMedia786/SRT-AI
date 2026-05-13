// =============================================================================
// SRT-AI  ·  Whisper chunking split-point test
// =============================================================================
// Validates the split-point math in isolation. Real audio chunking requires
// ffmpeg and an actual file; this just verifies the split logic produces
// sensible boundaries.
//
// Run:  node test_chunking.mjs
// =============================================================================

import { strict as assert } from 'node:assert';

let passed = 0, failed = 0;
const test = (name, fn) => {
    try { fn(); passed++; console.log(`  PASS  ${name}`); }
    catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
};

// Inline the same constants and function from server.js
const WHISPER_CHUNK_TARGET_SEC = 660;
const WHISPER_CHUNK_TOLERANCE_SEC = 60;

function computeWhisperSplitPoints(totalDuration, silences, chunkTarget = WHISPER_CHUNK_TARGET_SEC) {
    const splits = [0];
    let target = chunkTarget;
    while (target < totalDuration - 30) {
        const minStart = splits[splits.length - 1] + 60;
        const candidates = silences
            .filter(s => s.start > minStart && Math.abs((s.start + s.end) / 2 - target) <= WHISPER_CHUNK_TOLERANCE_SEC)
            .sort((a, b) => Math.abs((a.start + a.end) / 2 - target) - Math.abs((b.start + b.end) / 2 - target));
        const splitAt = candidates.length ? (candidates[0].start + candidates[0].end) / 2 : target;
        splits.push(splitAt);
        target = splitAt + chunkTarget;
    }
    splits.push(totalDuration);
    return splits;
}

console.log('\n=== Chunking split-point math ===');

test('short audio: no intermediate splits', () => {
    const splits = computeWhisperSplitPoints(300, []);
    assert.deepEqual(splits, [0, 300]);
});

test('exactly at chunk target: no extra split', () => {
    const splits = computeWhisperSplitPoints(660, []);
    assert.deepEqual(splits, [0, 660]);
});

test('1561s file (your real file): 3 chunks at fixed boundaries when no silence', () => {
    const splits = computeWhisperSplitPoints(1561.5, []);
    assert.equal(splits.length, 4);   // [0, ?, ?, 1561.5]
    assert.equal(splits[0], 0);
    assert.equal(splits[splits.length - 1], 1561.5);
    // intermediate splits should be at 660 and 1320 with no silence to snap to
    assert.equal(splits[1], 660);
    assert.equal(splits[2], 1320);
});

test('1561s file with silence near 11min: snaps to silence midpoint', () => {
    const silences = [
        { start: 658.0, end: 659.4 },   // close to target 660s, midpoint 658.7
        { start: 1305.0, end: 1308.0 }, // close to target 1320, midpoint 1306.5
    ];
    const splits = computeWhisperSplitPoints(1561.5, silences);
    assert.equal(splits[1], 658.7);   // snapped to silence midpoint
    assert.equal(splits[2], (1305.0 + 1308.0) / 2);
});

test('silence outside tolerance window is ignored', () => {
    const silences = [{ start: 500, end: 502 }];   // 161s away from target 660 → too far
    const splits = computeWhisperSplitPoints(1561.5, silences);
    assert.equal(splits[1], 660);   // ignored silence, used fixed target
});

test('every chunk fits under 24MB at 16kHz mono PCM', () => {
    // 16kHz × 2 bytes × 1 ch = 32 KB/s. 24 MB = 24576 KB. Max sec = 768.
    // Our default target is 660s with ±60s tolerance → worst case 720s ≈ 22.5MB. Safe.
    const splits = computeWhisperSplitPoints(3600, []);   // 1 hour audio
    for (let i = 1; i < splits.length; i++) {
        const chunkSec = splits[i] - splits[i - 1];
        const estMB = (chunkSec * 32) / 1024;
        assert.ok(estMB < 24, `chunk ${i} estimated ${estMB.toFixed(1)}MB ≥ 24MB limit`);
    }
});

test('progress is always forward (no infinite loop on bad silences)', () => {
    // Silences only at the very start — would lock the loop if not handled
    const silences = [{ start: 5, end: 6 }, { start: 10, end: 12 }];
    const splits = computeWhisperSplitPoints(3000, silences);
    for (let i = 1; i < splits.length; i++) {
        assert.ok(splits[i] > splits[i - 1], `split ${i} did not advance`);
    }
});

test('final chunk is at least 30s (avoids tiny tail chunks)', () => {
    // 1330s file would give chunks of 660 + 670; the +670 is fine
    const splits = computeWhisperSplitPoints(680, []);
    assert.equal(splits.length, 2);   // no intermediate split — tail would be 20s, too short
    // 690s would give a 30s tail — borderline OK
    const splits2 = computeWhisperSplitPoints(690, []);
    assert.equal(splits2.length, 2);
    // 700s gives a 40s tail — should split
    const splits3 = computeWhisperSplitPoints(700, []);
    // actually the rule is "while target < totalDuration - 30" so target=660 < 670 → splits
    assert.equal(splits3.length, 3);
});

test('two hour file produces ~11 chunks', () => {
    const splits = computeWhisperSplitPoints(7200, []);
    const chunkCount = splits.length - 1;
    assert.ok(chunkCount >= 10 && chunkCount <= 12, `expected 10-12 chunks, got ${chunkCount}`);
});

console.log('\n' + '='.repeat(60));
console.log(`  RESULTS:  ${passed} passed · ${failed} failed`);
console.log('='.repeat(60));
if (failed > 0) process.exit(1);
console.log('\n  ✓ Chunking math is sane.\n');
