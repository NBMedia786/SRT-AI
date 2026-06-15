// =============================================================================
// SRT-AI  ·  JSON salvage
// =============================================================================
// Gemini's JSON mode (responseMimeType: application/json) is *mostly* reliable
// but on long cue arrays it occasionally emits a corrupt tail — an unescaped
// quote inside a `text` value, an unquoted key, or a token-limit truncation.
// A plain JSON.parse() throws on the whole string. The chunk loop used to react
// to that by discarding the ENTIRE chunk, silently dropping minutes of audio
// that had actually transcribed fine (real incident: chunk 2/7 → 0 cues from a
// ~148s segment whose first ~38 cues were perfect).
//
// salvageJsonArray walks the array object-by-object and returns every complete
// LEADING object that parses on its own, stopping at the first corruption. The
// valid prefix is recovered; only the corrupt tail is lost.
// =============================================================================

/**
 * Recover the valid leading objects from a (possibly corrupt) JSON array string.
 *
 * Use only as a fallback after JSON.parse() throws — for well-formed input,
 * JSON.parse is faster and exact. This walker tracks string/escape state so
 * braces, brackets, colons and escaped quotes inside `text` values are not
 * miscounted; it stops at the first object that fails to parse, because once
 * the byte stream desyncs (e.g. an unescaped quote) later boundaries are no
 * longer trustworthy.
 *
 * @param {string} cleaned  output of cleanOutput() — expected to start with '['
 * @returns {Array<object>} recovered objects (possibly empty)
 */
export function salvageJsonArray(cleaned) {
    if (!cleaned || typeof cleaned !== 'string') return [];

    const objects = [];
    let depth = 0;          // object-brace nesting depth
    let inString = false;   // currently inside a JSON string?
    let escape = false;     // previous char was a backslash inside a string?
    let objStart = -1;      // index where the current top-level object began

    for (let i = 0; i < cleaned.length; i++) {
        const ch = cleaned[i];

        if (inString) {
            if (escape) { escape = false; continue; }
            if (ch === '\\') { escape = true; continue; }
            if (ch === '"') { inString = false; }
            continue;
        }

        if (ch === '"') { inString = true; continue; }

        if (ch === '{') {
            if (depth === 0) objStart = i;
            depth++;
        } else if (ch === '}') {
            depth--;
            if (depth === 0 && objStart !== -1) {
                const candidate = cleaned.slice(objStart, i + 1);
                let parsed;
                try {
                    parsed = JSON.parse(candidate);
                } catch {
                    // First unparseable object → corruption reached; the stream
                    // is no longer trustworthy past here, so stop and keep the
                    // valid prefix.
                    break;
                }
                if (parsed && typeof parsed === 'object') objects.push(parsed);
                objStart = -1;
            } else if (depth < 0) {
                // Structurally broken (a stray '}') — stop.
                break;
            }
        }
    }

    return objects;
}
