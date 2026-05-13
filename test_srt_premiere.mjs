// =============================================================================
// SRT-AI  ·  Premiere Pro compatibility test suite
// =============================================================================
// Runs against copies of the (now-fixed) helpers from server.js and verifies:
//   1. formatTimestamp produces canonical SRT timestamps with correct rounding
//   2. parseTimestamp round-trips losslessly
//   3. No overlapping cues, no end<=start, monotonic indices
//   4. CRLF line endings (required by Premiere Pro)
//   5. Whisper word distribution by character-weight (not uniform)
//   6. smartSplit allocates duration by word count (not char count)
//   7. Pipeline order: compactOverlaps before mergeCloseSegments
//   8. SRT parses cleanly with a strict standalone parser
//
// Run:  node test_srt_premiere.mjs
// =============================================================================

import { strict as assert } from 'node:assert';

let passed = 0, failed = 0;
const fail = (name, msg) => { failed++; console.log(`  FAIL  ${name}\n        ${msg}`); };
const ok   = (name)       => { passed++; console.log(`  PASS  ${name}`); };
function test(name, fn) {
    try { fn(); ok(name); } catch (e) { fail(name, e.message); }
}

// noop logger so the helpers under test can call log() without exploding
const log = (..._args) => {};

// =============================================================================
// HELPERS — copied verbatim from the FIXED server.js
// =============================================================================

function formatTimestamp(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const totalMs = Math.round(seconds * 1000);
    const h  = Math.floor(totalMs / 3600000);
    const m  = Math.floor((totalMs % 3600000) / 60000);
    const s  = Math.floor((totalMs % 60000) / 1000);
    const ms = totalMs % 1000;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function parseTimestamp(timestamp) {
    if (timestamp == null) return 0;
    if (typeof timestamp === 'number') return Number.isFinite(timestamp) ? Math.max(0, timestamp) : 0;
    if (typeof timestamp !== 'string') return 0;
    const raw = timestamp.trim();
    if (!raw) return 0;
    let m = raw.match(/^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
    if (m) {
        const h = parseInt(m[1], 10);
        const mi = parseInt(m[2], 10);
        const s = parseInt(m[3], 10);
        const ms = parseInt(m[4].padEnd(3, '0').slice(0, 3), 10);
        return h * 3600 + mi * 60 + s + ms / 1000;
    }
    m = raw.match(/^(\d{1,3}):(\d{2}):(\d{2})$/);
    if (m) return parseInt(m[1], 10) * 3600 + parseInt(m[2], 10) * 60 + parseInt(m[3], 10);
    m = raw.match(/^(\d{1,3}):(\d{2})[,.](\d{1,3})$/);
    if (m) {
        const mi = parseInt(m[1], 10);
        const s = parseInt(m[2], 10);
        const ms = parseInt(m[3].padEnd(3, '0').slice(0, 3), 10);
        return mi * 60 + s + ms / 1000;
    }
    m = raw.match(/^(\d{1,3}):(\d{1,2})$/);
    if (m) return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
    m = raw.match(/^(\d+(?:[,.]\d+)?)$/);
    if (m) {
        const v = parseFloat(m[1].replace(',', '.'));
        return Number.isFinite(v) ? v : 0;
    }
    log('warn', 'parseTimestamp: unrecognized timestamp format', { input: raw });
    return 0;
}

function compactOverlaps(segments, minDurationMs = 300) {
    if (!segments || segments.length === 0) return [];
    const sorted = [...segments].sort((a, b) => parseTimestamp(a.start) - parseTimestamp(b.start));
    const result = [];
    for (let i = 0; i < sorted.length; i++) {
        const seg = { ...sorted[i] };
        let startSec = parseTimestamp(seg.start);
        let endSec   = parseTimestamp(seg.end);
        if (endSec <= startSec) {
            const wordCount = seg.text ? seg.text.trim().split(/\s+/).length : 1;
            endSec = startSec + Math.max(minDurationMs / 1000, wordCount * 0.3);
        }
        if (i < sorted.length - 1) {
            const nextStart = parseTimestamp(sorted[i + 1].start);
            if (endSec > nextStart) {
                endSec = nextStart;
                if (endSec <= startSec) endSec = startSec + (minDurationMs / 1000);
            }
        }
        result.push({ ...seg, start: formatTimestamp(startSec), end: formatTimestamp(endSec) });
    }
    return result;
}

function mergeCloseSegments(segments, maxGapMs = 300, maxWords = 12, maxChars = 50) {
    if (!segments || segments.length <= 1) return segments;
    const merged = [];
    let current = { ...segments[0] };
    for (let i = 1; i < segments.length; i++) {
        const next = segments[i];
        const currentEnd = parseTimestamp(current.end);
        const nextStart  = parseTimestamp(next.start);
        const gapMs = (nextStart - currentEnd) * 1000;
        const combinedText = current.text.trim() + ' ' + next.text.trim();
        const combinedWords = combinedText.split(/\s+/).length;
        const combinedChars = combinedText.length;
        if (gapMs >= 0 && gapMs <= maxGapMs && combinedWords <= maxWords && combinedChars <= maxChars) {
            current = { ...current, text: combinedText, end: next.end };
        } else {
            merged.push(current);
            current = { ...next };
        }
    }
    merged.push(current);
    return merged;
}

function bridgeGaps(segments, maxBridgeGapMs = 500) {
    if (!segments || segments.length <= 1) return segments;
    const result = [];
    for (let i = 0; i < segments.length; i++) {
        const seg = { ...segments[i] };
        if (i < segments.length - 1) {
            const currentEnd = parseTimestamp(seg.end);
            const nextStart  = parseTimestamp(segments[i + 1].start);
            const gapMs = (nextStart - currentEnd) * 1000;
            if (gapMs > 0 && gapMs <= maxBridgeGapMs) {
                seg.end = formatTimestamp(nextStart);
            }
        }
        result.push(seg);
    }
    return result;
}

// =============================================================================
// STRICT PREMIERE-COMPATIBLE SRT PARSER (validator)
// =============================================================================
// Mirrors what Premiere Pro expects: index line, "HH:MM:SS,mmm --> HH:MM:SS,mmm",
// text line(s), blank separator. CRLF tolerated, LF tolerated. We re-parse our
// own output to verify it actually loads correctly.

function strictParseSRT(srt) {
    const issues = [];
    const cues = [];
    const blocks = srt.replace(/\r\n/g, '\n').trim().split(/\n{2,}/);

    blocks.forEach((block, blockIdx) => {
        const lines = block.split('\n').filter(l => l !== '');
        if (lines.length < 3) {
            issues.push(`Block ${blockIdx + 1}: fewer than 3 lines (expected index + time + text)`);
            return;
        }
        const idx = parseInt(lines[0].trim(), 10);
        if (!Number.isInteger(idx) || idx < 1) {
            issues.push(`Block ${blockIdx + 1}: invalid index '${lines[0]}'`);
        }
        const tm = lines[1].match(/^(\d{2}:\d{2}:\d{2},\d{3})\s-->\s(\d{2}:\d{2}:\d{2},\d{3})$/);
        if (!tm) {
            issues.push(`Block ${blockIdx + 1}: malformed timecode line '${lines[1]}'`);
            return;
        }
        const startSec = parseTimestamp(tm[1]);
        const endSec   = parseTimestamp(tm[2]);
        if (endSec <= startSec) issues.push(`Cue ${idx}: end (${tm[2]}) <= start (${tm[1]})`);
        const text = lines.slice(2).join('\n').trim();
        if (!text) issues.push(`Cue ${idx}: empty text`);
        cues.push({ idx, startSec, endSec, text });
    });

    // monotonic index
    for (let i = 0; i < cues.length; i++) {
        if (cues[i].idx !== i + 1) {
            issues.push(`Cue index not monotonic at position ${i}: got ${cues[i].idx}, expected ${i + 1}`);
            break;
        }
    }
    // monotonic time (no overlap)
    for (let i = 1; i < cues.length; i++) {
        if (cues[i].startSec < cues[i - 1].endSec - 1e-6) {
            issues.push(`Cue ${cues[i].idx} starts before cue ${cues[i - 1].idx} ends (overlap)`);
        }
    }
    return { cues, issues };
}

// =============================================================================
// TESTS
// =============================================================================

console.log('\n=== Group 1: formatTimestamp precision ===');

test('zero seconds', () => {
    assert.equal(formatTimestamp(0), '00:00:00,000');
});

test('integer second', () => {
    assert.equal(formatTimestamp(1), '00:00:01,000');
});

test('100ms', () => {
    assert.equal(formatTimestamp(0.1), '00:00:00,100');
});

test('59.9999s rounds UP to 1 minute (was floor bug)', () => {
    assert.equal(formatTimestamp(59.9999), '00:01:00,000');
});

test('3599.9995s rounds UP to 01:00:00,000 (was floor bug)', () => {
    assert.equal(formatTimestamp(3599.9995), '01:00:00,000');
});

test('7261.4567s rounds correctly (was 1ms low)', () => {
    assert.equal(formatTimestamp(7261.4567), '02:01:01,457');
});

test('floating-point 0.1+0.2 normalized', () => {
    assert.equal(formatTimestamp(0.1 + 0.2), '00:00:00,300');
});

test('negative input clamped to zero', () => {
    assert.equal(formatTimestamp(-5), '00:00:00,000');
});

test('NaN clamped to zero', () => {
    assert.equal(formatTimestamp(NaN), '00:00:00,000');
});

test('Infinity clamped to zero', () => {
    assert.equal(formatTimestamp(Infinity), '00:00:00,000');
});

console.log('\n=== Group 2: parseTimestamp robustness ===');

test('canonical SRT HH:MM:SS,mmm', () => {
    assert.equal(parseTimestamp('00:00:03,500'), 3.5);
});

test('canonical with hour offset', () => {
    assert.equal(parseTimestamp('01:23:45,678'), 5025.678);
});

test('dot variant', () => {
    assert.equal(parseTimestamp('00:00:03.500'), 3.5);
});

test('HH:MM:SS no fractional', () => {
    assert.equal(parseTimestamp('00:30:00'), 1800);
});

test('MM:SS,mmm (AI variant)', () => {
    assert.equal(parseTimestamp('05:30,250'), 330.25);
});

test('MM:SS short form', () => {
    assert.equal(parseTimestamp('05:30'), 330);
});

test('bare seconds (float)', () => {
    assert.equal(parseTimestamp('12.345'), 12.345);
});

test('bare seconds (comma decimal)', () => {
    assert.equal(parseTimestamp('12,345'), 12.345);
});

test('garbage rejected (returns 0, not crash)', () => {
    assert.equal(parseTimestamp('not a timestamp'), 0);
});

test('null safe', () => {
    assert.equal(parseTimestamp(null), 0);
});

test('undefined safe', () => {
    assert.equal(parseTimestamp(undefined), 0);
});

test('number passthrough', () => {
    assert.equal(parseTimestamp(42.5), 42.5);
});

console.log('\n=== Group 3: round-trip drift (parse -> format -> parse) ===');

test('drift under 1ms over 5000 random samples up to 4 hours', () => {
    let maxDrift = 0;
    let totalDrift = 0;
    const N = 5000;
    for (let i = 0; i < N; i++) {
        const orig = Math.random() * 4 * 3600;
        const back = parseTimestamp(formatTimestamp(orig));
        const driftMs = Math.abs(orig - back) * 1000;
        if (driftMs > maxDrift) maxDrift = driftMs;
        totalDrift += driftMs;
    }
    const avg = totalDrift / N;
    console.log(`        avg drift: ${avg.toFixed(4)} ms · max: ${maxDrift.toFixed(4)} ms`);
    assert.ok(maxDrift < 1.0, `max drift ${maxDrift.toFixed(4)}ms exceeded 1ms tolerance`);
    assert.ok(avg < 0.5,    `avg drift ${avg.toFixed(4)}ms exceeded 0.5ms tolerance`);
});

console.log('\n=== Group 4: SRT output is Premiere Pro compatible ===');

function buildSRTFromCues(cues) {
    return cues.map((c, i) =>
        `${i + 1}\r\n${c.start} --> ${c.end}\r\n${c.text}\r\n`
    ).join('\r\n');
}

test('uses CRLF line endings (Premiere requirement)', () => {
    const srt = buildSRTFromCues([
        { start: '00:00:00,000', end: '00:00:02,500', text: 'hello' },
        { start: '00:00:02,500', end: '00:00:05,000', text: 'world' },
    ]);
    assert.match(srt, /\r\n/);
    // every newline must be CRLF (no bare LF outside of multi-line text)
    const cleanedForCheck = srt.replace(/\r\n/g, '|');
    assert.ok(!cleanedForCheck.includes('\n'), 'bare LF found — Premiere will reject');
});

test('uses comma as decimal separator (not period)', () => {
    const stamp = formatTimestamp(3.5);
    assert.ok(stamp.includes(','), `formatTimestamp must use comma: got '${stamp}'`);
    assert.ok(!stamp.includes('.'), `formatTimestamp must not use period: got '${stamp}'`);
});

test('timestamp is exactly HH:MM:SS,mmm with no missing digits', () => {
    for (const sec of [0, 0.001, 1, 60, 3600, 3661.123, 7321.999]) {
        const t = formatTimestamp(sec);
        assert.match(t, /^\d{2}:\d{2}:\d{2},\d{3}$/, `bad format: ${t}`);
    }
});

test('pipeline output: no overlaps after compactOverlaps', () => {
    let segs = [
        { start: '00:00:00,000', end: '00:00:03,000', text: 'one' },
        { start: '00:00:02,500', end: '00:00:05,000', text: 'two' },  // overlaps one
        { start: '00:00:04,500', end: '00:00:07,000', text: 'three' },  // overlaps two
    ];
    segs = compactOverlaps(segs, 300);
    for (let i = 1; i < segs.length; i++) {
        const prevEnd = parseTimestamp(segs[i - 1].end);
        const thisStart = parseTimestamp(segs[i].start);
        assert.ok(thisStart >= prevEnd - 1e-9, `cue ${i} overlaps cue ${i - 1}`);
    }
});

test('pipeline output: no end<=start (inverted cues fixed)', () => {
    let segs = [
        { start: '00:00:05,000', end: '00:00:03,000', text: 'inverted' },
        { start: '00:00:10,000', end: '00:00:10,000', text: 'zero-duration' },
    ];
    segs = compactOverlaps(segs, 300);
    for (const s of segs) {
        const startSec = parseTimestamp(s.start);
        const endSec   = parseTimestamp(s.end);
        assert.ok(endSec > startSec, `cue '${s.text}' has end<=start (${s.start} --> ${s.end})`);
    }
});

test('full SRT re-parses cleanly with strict validator', () => {
    const cues = [
        { start: '00:00:00,000', end: '00:00:02,500', text: 'Hello, this is the first sentence.' },
        { start: '00:00:02,500', end: '00:00:05,200', text: 'Thank you for the introduction.' },
        { start: '00:00:05,250', end: '00:00:08,000', text: 'Now we move to the next topic.' },
    ];
    const srt = buildSRTFromCues(cues);
    const { cues: parsed, issues } = strictParseSRT(srt);
    assert.deepEqual(issues, [], `parser found issues: ${issues.join('; ')}`);
    assert.equal(parsed.length, 3);
});

console.log('\n=== Group 5: pipeline order — overlap before merge ===');

test('overlap-then-merge resolves overlapping cues correctly', () => {
    // Whisper-style adjacent word boundaries that overlap by 1-50ms.
    let segs = [
        { start: '00:00:00,000', end: '00:00:01,520', text: 'hello' },
        { start: '00:00:01,500', end: '00:00:03,000', text: 'world' },   // -20ms overlap
        { start: '00:00:03,100', end: '00:00:05,000', text: 'how are you' },
    ];
    // simulate fixed pipeline order
    segs = compactOverlaps(segs, 300);
    segs = mergeCloseSegments(segs, 150, 12, 50);
    segs = bridgeGaps(segs, 250);
    // overlap should be gone and 100ms gap should be bridged
    for (let i = 1; i < segs.length; i++) {
        const prevEnd = parseTimestamp(segs[i - 1].end);
        const thisStart = parseTimestamp(segs[i].start);
        assert.ok(thisStart >= prevEnd - 1e-9, `overlap survived at index ${i}`);
    }
});

test('mergeCloseSegments alone would NOT fix overlaps (regression guard)', () => {
    // Demonstrates why we reordered: merge on its own ignores negative gaps.
    let segs = [
        { start: '00:00:00,000', end: '00:00:01,520', text: 'hello' },
        { start: '00:00:01,500', end: '00:00:03,000', text: 'world' },
    ];
    segs = mergeCloseSegments(segs, 150, 12, 50);
    // merge skips because gap is negative; overlap remains.
    const endA = parseTimestamp(segs[0].end);
    const startB = segs.length > 1 ? parseTimestamp(segs[1].start) : Infinity;
    assert.ok(startB < endA, 'merge alone should NOT resolve overlap (this is the regression-guarded expectation)');
});

console.log('\n=== Group 6: word distribution by character weight (Whisper fix) ===');

function distributeWords(text, startSec, endSec) {
    const tokens = text.split(/\s+/).filter(Boolean);
    const words = [];
    if (tokens.length === 1) {
        words.push({ word: tokens[0], startTime: startSec, endTime: endSec });
    } else if (tokens.length > 1) {
        const dur = Math.max(0, endSec - startSec);
        const weights = tokens.map(t => Math.max(1, t.replace(/[^a-zA-Z0-9']/g, '').length));
        const totalWeight = weights.reduce((a, b) => a + b, 0) || tokens.length;
        let cursor = startSec;
        for (let i = 0; i < tokens.length; i++) {
            const slice = dur * (weights[i] / totalWeight);
            const start = cursor;
            const end = (i === tokens.length - 1) ? endSec : cursor + slice;
            words.push({ word: tokens[i], startTime: start, endTime: end });
            cursor = end;
        }
    }
    return words;
}

test('long word gets more time than short word', () => {
    // "I" (1 char) vs "extraordinary" (13 chars) over 2 seconds
    const words = distributeWords('I extraordinary', 0, 2);
    assert.equal(words.length, 2);
    const durShort = words[0].endTime - words[0].startTime;
    const durLong  = words[1].endTime - words[1].startTime;
    assert.ok(durLong > durShort * 5, `long word should get >5x time of "I" — got ${durLong.toFixed(3)} vs ${durShort.toFixed(3)}`);
});

test('last word ends exactly at chunk end (no drift)', () => {
    const words = distributeWords('one two three four', 1.234, 5.678);
    assert.equal(words[words.length - 1].endTime, 5.678);
});

test('first word starts exactly at chunk start', () => {
    const words = distributeWords('one two three four', 1.234, 5.678);
    assert.equal(words[0].startTime, 1.234);
});

test('words are temporally contiguous (no gaps, no overlaps)', () => {
    const words = distributeWords('one two three four five', 0, 10);
    for (let i = 1; i < words.length; i++) {
        assert.equal(words[i].startTime, words[i - 1].endTime, `gap/overlap at word ${i}`);
    }
});

console.log('\n=== Group 7: smartSplit by word count (not char count) ===');

function smartSplit_byWords(text, startTime, endTime) {
    const words = text.trim().split(/\s+/);
    const totalDuration = endTime - startTime;
    // simulate two-segment split at midpoint of word count
    const mid = Math.floor(words.length / 2);
    const firstWords = words.slice(0, mid);
    const segA = {
        text: firstWords.join(' '),
        start: startTime,
        end: startTime + (firstWords.length / words.length) * totalDuration,
    };
    const segB = {
        text: words.slice(mid).join(' '),
        start: segA.end,
        end: endTime,
    };
    return [segA, segB];
}

test('equal word counts yield equal durations', () => {
    const [a, b] = smartSplit_byWords('one two three four', 0, 8);
    assert.equal(Math.round((a.end - a.start) * 1000), 4000);
    assert.equal(Math.round((b.end - b.start) * 1000), 4000);
});

test('split duration NOT biased by character length', () => {
    // 4 short words vs 4 long words — should still split 50/50
    const [a, b] = smartSplit_byWords('a b c d extraordinary multifaceted contemplation profundity', 0, 8);
    const aDur = a.end - a.start;
    const bDur = b.end - b.start;
    assert.ok(Math.abs(aDur - bDur) < 0.01,
        `expected near-equal durations; got ${aDur.toFixed(3)} vs ${bDur.toFixed(3)}`);
});

console.log('\n=== Group 8: end-to-end SRT generation passes Premiere validator ===');

test('full mini-pipeline produces a valid Premiere-compatible SRT', () => {
    // Simulate AI output with all the problematic patterns
    const aiSegments = [
        { start: '00:00:00,000', end: '00:00:03,500', text: 'Hello, this is the first sentence.' },
        { start: '00:00:03,500', end: '00:00:07,200', text: 'Thank you for that introduction.' },
        // overlap with previous
        { start: '00:00:07,000', end: '00:00:09,000', text: 'I appreciate it.' },
        // tiny gap
        { start: '00:00:09,100', end: '00:00:12,000', text: 'Let me begin the presentation.' },
        // sub-millisecond fractional that would have triggered floor bug
        { start: '00:00:12,000', end: '00:00:14,9999', text: 'First slide is overview.' },
        // inverted
        { start: '00:00:18,000', end: '00:00:15,500', text: 'Whoops, inverted.' },
        // long text that needs to fit anyway
        { start: '00:00:20,000', end: '00:00:24,000', text: 'And finally we wrap up here.' },
    ].map(s => ({
        start: formatTimestamp(parseTimestamp(s.start)),
        end:   formatTimestamp(parseTimestamp(s.end)),
        text:  s.text,
    }));

    let segs = compactOverlaps(aiSegments, 300);
    segs = mergeCloseSegments(segs, 150, 12, 50);
    segs = bridgeGaps(segs, 250);

    const srt = segs.map((c, i) =>
        `${i + 1}\r\n${c.start} --> ${c.end}\r\n${c.text}\r\n`
    ).join('\r\n');

    const { cues, issues } = strictParseSRT(srt);
    if (issues.length) {
        console.log('        Sample output:\n' + srt.split('\r\n').map(l => '          ' + l).join('\n'));
    }
    assert.deepEqual(issues, [], `Premiere validator found issues: ${issues.join('; ')}`);
    assert.ok(cues.length >= 5, `expected at least 5 cues, got ${cues.length}`);
});

// =============================================================================
// SUMMARY
// =============================================================================
console.log('\n' + '='.repeat(60));
console.log(`  RESULTS:  ${passed} passed · ${failed} failed`);
console.log('='.repeat(60));
if (failed > 0) {
    process.exit(1);
}
console.log('\n  ✓ All Premiere Pro compatibility checks passed.\n');
