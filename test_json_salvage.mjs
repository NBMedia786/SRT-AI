// =============================================================================
// SRT-AI  ·  JSON salvage test
// =============================================================================
// Gemini's JSON mode occasionally emits a long cue array whose TAIL is corrupt
// (typically an unescaped quote inside a `text` value, or a truncated object).
// A plain JSON.parse() throws on the whole string, which previously caused the
// chunk loop to discard the ENTIRE chunk — silently dropping minutes of
// correctly-transcribed audio (real incident: chunk 2/7 → 0 cues from a ~148s
// segment that HAD transcribed fine).
//
// salvageJsonArray() must recover every complete leading object that parses,
// stopping at the first corruption.
//
// Run:  node test_json_salvage.mjs
// =============================================================================

import { strict as assert } from 'node:assert';
import { salvageJsonArray } from './json_salvage.js';

let passed = 0, failed = 0;
const test = (name, fn) => {
    try { fn(); passed++; console.log(`  PASS  ${name}`); }
    catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
};

console.log('\n=== JSON salvage ===');

test('valid array: returns all objects unchanged', () => {
    const json = JSON.stringify([
        { start: '00:00:00,000', end: '00:00:01,500', text: '11:30 Central.' },
        { start: '00:00:01,500', end: '00:00:03,000', text: 'Come on out.' },
    ]);
    const out = salvageJsonArray(json);
    assert.equal(out.length, 2);
    assert.equal(out[0].text, '11:30 Central.');
    assert.equal(out[1].text, 'Come on out.');
});

test('unescaped quote in a tail cue: recovers the valid prefix', () => {
    // Object 3 has unescaped inner quotes -> JSON.parse throws on the whole array.
    // We must still recover objects 1 and 2.
    const broken =
        '[\n' +
        '  {"start":"00:00:00,000","end":"00:00:01,500","text":"11:30 Central."},\n' +
        '  {"start":"00:00:01,500","end":"00:00:03,000","text":"Come on out."},\n' +
        '  {"start":"00:00:03,000","end":"00:00:05,000","text":"He said "get down" now."},\n' +
        '  {"start":"00:00:05,000","end":"00:00:07,000","text":"Last cue."}\n' +
        ']';
    assert.throws(() => JSON.parse(broken), 'sanity: payload must be invalid JSON');
    const out = salvageJsonArray(broken);
    assert.equal(out.length, 2);
    assert.equal(out[0].text, '11:30 Central.');
    assert.equal(out[1].text, 'Come on out.');
});

test('unquoted property name in a tail cue: recovers the valid prefix', () => {
    // Reproduces the exact production error: "Expected double-quoted property name".
    const broken =
        '[\n' +
        '  {"start":"00:00:00,000","end":"00:00:01,500","text":"one"},\n' +
        '  {"start":"00:00:01,500","end":"00:00:03,000","text":"two"},\n' +
        '  {"start":"00:00:03,000","end":"00:00:05,000",text:"three"}\n' +
        ']';
    assert.throws(() => JSON.parse(broken));
    const out = salvageJsonArray(broken);
    assert.equal(out.length, 2);
    assert.deepEqual(out.map(o => o.text), ['one', 'two']);
});

test('truncated mid-object (hit token limit): recovers complete objects', () => {
    const broken =
        '[\n' +
        '  {"start":"00:00:00,000","end":"00:00:01,500","text":"one"},\n' +
        '  {"start":"00:00:01,500","end":"00:00:03,000","text":"two"},\n' +
        '  {"start":"00:00:03,000","end":"00:00:0';   // cut off mid-value
    const out = salvageJsonArray(broken);
    assert.equal(out.length, 2);
    assert.deepEqual(out.map(o => o.text), ['one', 'two']);
});

test('trailing comma before close: still recovers all objects', () => {
    const broken =
        '[{"start":"0","end":"1","text":"a"},{"start":"1","end":"2","text":"b"},]';
    const out = salvageJsonArray(broken);
    assert.equal(out.length, 2);
    assert.deepEqual(out.map(o => o.text), ['a', 'b']);
});

test('escaped quotes and braces inside text are preserved, not miscounted', () => {
    const json = JSON.stringify([
        { start: '0', end: '1', text: 'she said "hi" to {him}' },
        { start: '1', end: '2', text: 'a:b [c] {d}' },
    ]);
    const out = salvageJsonArray(json);
    assert.equal(out.length, 2);
    assert.equal(out[0].text, 'she said "hi" to {him}');
    assert.equal(out[1].text, 'a:b [c] {d}');
});

test('empty array: returns []', () => {
    assert.deepEqual(salvageJsonArray('[]'), []);
});

test('no recoverable objects / garbage: returns []', () => {
    assert.deepEqual(salvageJsonArray('not json at all'), []);
    assert.deepEqual(salvageJsonArray(''), []);
    assert.deepEqual(salvageJsonArray(null), []);
});

test('first object already corrupt: returns [] (nothing salvageable)', () => {
    const broken = '[{"start":"0","end":"1","text":"bad "quote" here"},{"start":"1","end":"2","text":"b"}]';
    const out = salvageJsonArray(broken);
    assert.equal(out.length, 0);
});

console.log('\n' + '='.repeat(60));
console.log(`  RESULTS:  ${passed} passed · ${failed} failed`);
console.log('='.repeat(60));
if (failed > 0) process.exit(1);
console.log('\n  ✓ JSON salvage recovers valid cues from corrupt Gemini output.\n');
