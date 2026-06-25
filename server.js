import express from 'express';
import multer from 'multer';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/server';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import cors from 'cors';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import { HfInference } from '@huggingface/inference';
import { fileURLToPath } from 'url';

// Fix __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Configure FFmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

// Middleware
app.use(cors());
app.use(express.static('.')); // Serve static files from current directory
app.use(express.json()); // Parse JSON bodies

// Configure Multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname)) // Append extension
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 2 * 1024 * 1024 * 1024 } // 2GB limit
});

// Ensure uploads directory exists
if (!fs.existsSync('uploads')) {
    fs.mkdirSync('uploads');
}

// Initialize Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// Initialize File Manager
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);

// Choose Model with Deterministic Settings
const MODEL_NAME = process.env.GEMINI_MODEL || "gemini-2.5-pro";
console.log('--- SERVER STARTING (Gemini 2.x Architecture) ---');
console.log(`[DEBUG] GEMINI_MODEL from env: ${process.env.GEMINI_MODEL}`);
console.log(`[DEBUG] GEMINI_MODEL final value: ${MODEL_NAME}`);
const model = genAI.getGenerativeModel({
    model: MODEL_NAME,
    generationConfig: {
        temperature: 0, // Deterministic output for consistent transcriptions
    }
});

// Initialize Hugging Face for Whisper fallback
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Store active processing jobs for progress tracking
const jobs = {};

// SSE Endpoint for progress updates
app.get('/api/progress/:jobId', (req, res) => {
    const { jobId } = req.params;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Send initial connection message
    res.write(`data: ${JSON.stringify({ stage: 'connected', percent: 0, message: 'Connected to progress stream' })}\n\n`);

    // Store the response object to send updates
    jobs[jobId] = res;

    // Clean up when client disconnects
    req.on('close', () => {
        delete jobs[jobId];
    });
});

// Helper to send progress
function sendProgress(jobId, stage, percent, message) {
    if (jobs[jobId]) {
        jobs[jobId].write(`data: ${JSON.stringify({ stage, percent, message })}\n\n`);
    }
}

// Helper: Convert Audio to WAV (Mono, 16kHz) for better AI compatibility
function convertToWav(inputPath) {
    return new Promise((resolve, reject) => {
        const outputPath = inputPath + '.wav';
        ffmpeg(inputPath)
            .toFormat('wav')
            .audioChannels(1) // Mono
            .audioFrequency(16000) // 16kHz
            .on('end', () => resolve(outputPath))
            .on('error', (err) => reject(err))
            .save(outputPath);
    });
}

// Helper: Get Audio Duration in Seconds
function getAudioDuration(filePath) {
    return new Promise((resolve, reject) => {
        ffmpeg.ffprobe(filePath, (err, metadata) => {
            if (err) return reject(err);
            resolve(metadata.format.duration);
        });
    });
}

// Helper: Convert Seconds to SRT Timestamp (HH:MM:SS,mmm)
function formatTimestamp(seconds) {
    const date = new Date(0);
    date.setMilliseconds(seconds * 1000);
    const hh = String(date.getUTCHours()).padStart(2, '0');
    const mm = String(date.getUTCMinutes()).padStart(2, '0');
    const ss = String(date.getUTCSeconds()).padStart(2, '0');
    const mmm = String(date.getUTCMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss},${mmm}`;
}

// Helper: Parse SRT Timestamp to Seconds (Robust)
// Handles "MM:SS", "HH:MM:SS", "HH:MM:SS,mmm" correctly
function parseTimestamp(timestamp) {
    if (!timestamp) return 0;

    // Cleanup
    timestamp = timestamp.trim().replace(',', '.');
    const parts = timestamp.split(':');

    // Handle different formats
    if (parts.length === 3) {
        // HH:MM:SS
        const h = parseInt(parts[0]) || 0;
        const m = parseInt(parts[1]) || 0;
        const s = parseFloat(parts[2]) || 0;
        return (h * 3600) + (m * 60) + s;
    } else if (parts.length === 2) {
        // MM:SS (AI sometimes returns this)
        const m = parseInt(parts[0]) || 0;
        const s = parseFloat(parts[1]) || 0;
        return (m * 60) + s;
    } else if (parts.length === 1) {
        // SS (Rare)
        return parseFloat(parts[0]) || 0;
    }
    return 0; // Fallback
}


// Helper: Split Audio into Chunks
function splitAudio(inputPath, chunkDuration = 600) { // 10 minutes default
    return new Promise((resolve, reject) => {
        const outputPattern = inputPath + '_chunk_%03d.wav';
        ffmpeg(inputPath)
            .outputOptions([
                '-f segment',
                `-segment_time ${chunkDuration}`,
                '-c copy'
            ])
            .on('end', () => {
                // Find all generated chunks
                const dir = path.dirname(inputPath);
                const baseName = path.basename(inputPath);
                fs.readdir(dir, (err, files) => {
                    if (err) return reject(err);
                    const chunks = files
                        .filter(f => f.startsWith(baseName + '_chunk_'))
                        .map(f => path.join(dir, f))
                        .sort();
                    resolve(chunks);
                });
            })
            .on('error', reject)
            .save(outputPattern);
    });
}

// Helper: Wait for file to likely be active on Google servers
async function waitForFileActive(file) {
    let state = file.state;
    // Poll up to 60 times (2 mins roughly)
    for (let i = 0; i < 60; i++) {
        if (state === "ACTIVE") return true;
        if (state === "FAILED") return false;
        await new Promise(r => setTimeout(r, 2000));
        const currentFile = await fileManager.getFile(file.name);
        state = currentFile.state;
    }
    return false;
}

// Helper: Retry API calls with exponential backoff
async function retryWithBackoff(fn, maxRetries = 3, initialDelay = 1000) {
    let lastError;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            const isNetworkError = error.message?.includes('fetch failed') ||
                error.message?.includes('ECONNRESET') ||
                error.message?.includes('ETIMEDOUT') ||
                error.message?.includes('network');

            if (!isNetworkError || attempt === maxRetries - 1) {
                throw error; // Don't retry non-network errors or final attempt
            }

            const delay = initialDelay * Math.pow(2, attempt);
            console.log(`[Retry] Attempt ${attempt + 1}/${maxRetries} failed. Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw lastError;
}

// Helper: Smart split a segment at natural break points
function smartSplitSegment(text, startTime, endTime, maxWords = 10, maxChars = 50) {
    const words = text.trim().split(/\s+/);

    // If already within limits, return as-is
    if (words.length <= maxWords && text.length <= maxChars) {
        return [{ text: text.trim(), start: startTime, end: endTime }];
    }

    const segments = [];
    const totalDuration = endTime - startTime;

    // Weak words that should not end a segment
    const weakWords = new Set(['the', 'a', 'an', 'and', 'but', 'or', 'of', 'in', 'at', 'to', 'is', 'was', 'are', 'were', 'she', 'he', 'it', 'they', 'we', 'you', 'your', 'his', 'her', 'my', 'our', 'their']);

    // Try to find sentence boundaries first (highest priority)
    let currentSegment = [];
    let currentChars = 0;
    let segmentStartTime = startTime;

    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const wordWithSpace = (currentSegment.length > 0 ? ' ' : '') + word;

        // Check remaining words after this one
        const remainingWords = words.length - (i + 1);

        // Check if adding this word would exceed limits
        const wouldExceedWords = currentSegment.length + 1 > maxWords;
        const wouldExceedChars = currentChars + wordWithSpace.length > maxChars;

        // Check if this is a sentence boundary (period, exclamation, question mark)
        const hasPeriod = word.includes('.');
        const hasQuestion = word.includes('?');
        const hasExclamation = word.includes('!');
        const isSentenceEnd = hasPeriod || hasQuestion || hasExclamation;

        // Check if this is a natural break point within a sentence
        const hasComma = word.includes(',');
        const hasSemicolon = word.includes(';');
        const nextIsConjunction = i + 1 < words.length && ['and', 'but', 'or', 'so'].includes(words[i + 1].toLowerCase());
        const isInnerBreak = hasComma || hasSemicolon || nextIsConjunction;

        // Add word to current segment
        currentSegment.push(word);
        currentChars += wordWithSpace.length;

        // Decide whether to break here
        let shouldBreak = false;

        // PRIORITY 1: Sentence boundaries (if we have enough content and won't orphan words)
        if (isSentenceEnd && currentSegment.length >= 3 && remainingWords >= 3) {
            shouldBreak = true;
        }
        // PRIORITY 2: Exceeded limits and at a natural break
        else if ((wouldExceedWords || wouldExceedChars) && isInnerBreak && remainingWords >= 3) {
            const lastWord = word.replace(/[.,!?;]+$/, '').toLowerCase();
            if (!weakWords.has(lastWord)) {
                shouldBreak = true;
            }
        }
        // PRIORITY 3: Hard limit exceeded (must break to avoid overflow)
        else if (currentSegment.length >= maxWords || currentChars >= maxChars) {
            // Only break if we won't orphan 1-2 words
            if (remainingWords >= 3) {
                const lastWord = currentSegment[currentSegment.length - 1].replace(/[.,!?;]+$/, '').toLowerCase();
                // Try to avoid ending with weak words, but break anyway if we must
                if (!weakWords.has(lastWord) || currentChars > maxChars * 1.2) {
                    shouldBreak = true;
                }
            }
        }

        // Execute the break
        if (shouldBreak) {
            const segmentText = currentSegment.join(' ');
            const segmentDuration = (segmentText.length / text.length) * totalDuration;
            const segmentEndTime = segmentStartTime + segmentDuration;

            segments.push({
                text: segmentText,
                start: segmentStartTime,
                end: segmentEndTime
            });

            currentSegment = [];
            currentChars = 0;
            segmentStartTime = segmentEndTime;
        }
    }

    // Add remaining segment (merge with last if too short)
    if (currentSegment.length > 0) {
        // If this segment is only 1-2 words, merge with previous segment
        if (currentSegment.length <= 2 && segments.length > 0) {
            const lastSegment = segments[segments.length - 1];
            const mergedText = lastSegment.text + ' ' + currentSegment.join(' ');

            // Only merge if the combined segment isn't too long
            if (mergedText.split(/\s+/).length <= maxWords * 1.5 && mergedText.length <= maxChars * 1.5) {
                segments[segments.length - 1] = {
                    text: mergedText,
                    start: lastSegment.start,
                    end: endTime
                };
            } else {
                // Can't merge, add as separate segment
                segments.push({
                    text: currentSegment.join(' '),
                    start: segmentStartTime,
                    end: endTime
                });
            }
        } else {
            segments.push({
                text: currentSegment.join(' '),
                start: segmentStartTime,
                end: endTime
            });
        }
    }

    return segments;
}

// Helper: Make a cue list bulletproof for Adobe Premiere Pro import.
// Premiere rejects (or silently drops) SRTs with: empty cue text, zero/negative
// duration, overlapping or out-of-order cues, or line breaks/blank lines inside
// a cue. This pass guarantees: clean single-line text, chronological order,
// strictly positive duration, and no overlaps (monotonic, gap-or-touch only).
function sanitizeCuesForPremiere(rawCues) {
    const MIN_DUR = 0.05; // 50ms floor so end is ALWAYS strictly greater than start

    const cleaned = rawCues
        .map(c => {
            // Collapse any internal line breaks / control chars so a cue is always one block.
            const text = String(c.text == null ? '' : c.text)
                .replace(/\r\n|\r|\n/g, ' ')
                .replace(new RegExp('[\\u0000-\\u001F\\u007F]', 'g'), ' ')
                .replace(/\s+/g, ' ')
                .trim();
            let start = Number(c.start);
            let end = Number(c.end);
            if (!isFinite(start) || start < 0) start = 0;
            if (!isFinite(end)) end = start;
            return { start, end, text };
        })
        .filter(c => c.text.length > 0); // drop empty-text cues (Premiere errors on these)

    // Chronological order (then by end) so the overlap sweep is correct.
    cleaned.sort((a, b) => a.start - b.start || a.end - b.end);

    // Forward sweep: every cue starts no earlier than the previous one ended, and
    // every cue has a strictly positive duration. This is the structure Premiere wants.
    const out = [];
    let lastEnd = 0;
    for (const c of cleaned) {
        let s = c.start < lastEnd ? lastEnd : c.start;
        let e = c.end;
        if (!(e > s + 0.001)) e = s + MIN_DUR;
        out.push({ start: s, end: e, text: c.text });
        lastEnd = e;
    }
    return out;
}

// Helper: Convert JSON response from AI to SRT format
// Smart-splits long segments, then runs a Premiere-safe sanitization pass.
function jsonToSrt(jsonString, wordLimit) {
    try {
        // Clean the string (remove markdown code blocks if present)
        let cleanJson = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();

        // Find the array start and end
        const start = cleanJson.indexOf('[');
        const end = cleanJson.lastIndexOf(']');

        if (start === -1 || end === -1) {
            console.error("Invalid JSON structure received:", cleanJson);
            return ""; // Or throw error
        }

        cleanJson = cleanJson.substring(start, end + 1);
        const segments = JSON.parse(cleanJson);

        // Build a NUMERIC cue list (seconds) so the sanitizer can do exact math,
        // formatting to HH:MM:SS,mmm only once at the very end.
        const rawCues = [];
        const maxWords = 12; // Stricter limits: 12 words OR 50 characters
        const maxChars = 50;

        segments.forEach((seg) => {
            if (!seg || typeof seg.text !== 'string') return;
            const text = seg.text.trim();
            if (!text) return;

            // Normalize timestamps (handles AI returning MM:SS etc) -> seconds.
            const tStart = parseTimestamp(seg.start);
            const tEnd = parseTimestamp(seg.end);

            const words = text.split(/\s+/);
            if (words.length <= maxWords && text.length <= maxChars) {
                // Within limits - trust the AI segmentation.
                rawCues.push({ start: tStart, end: tEnd, text });
            } else {
                // Exceeds limits - smart-split into readable sub-cues.
                smartSplitSegment(text, tStart, tEnd, maxWords, maxChars)
                    .forEach(s => rawCues.push({ start: s.start, end: s.end, text: s.text }));
            }
        });

        // Premiere-safe pass: clean, order, de-overlap, guarantee positive duration.
        const cues = sanitizeCuesForPremiere(rawCues);
        if (cues.length === 0) return "";

        // Emit SRT: sequential 1-based numbering, CRLF line endings, blank line between cues.
        let srtOutput = "";
        cues.forEach((seg, i) => {
            let cleanText = seg.text;
            // Clean single-word segments: remove trailing punctuation.
            if (cleanText.split(/\s+/).length === 1) {
                cleanText = cleanText.replace(/[.,!?]+$/, '');
            }
            srtOutput += `${i + 1}\r\n${formatTimestamp(seg.start)} --> ${formatTimestamp(seg.end)}\r\n${cleanText}\r\n\r\n`;
        });

        // Ensure a single clean trailing newline.
        return srtOutput.trimEnd() + '\r\n';
    } catch (e) {
        console.error("Error parsing JSON to SRT:", e);
        return "";
    }
}

// Helper: Whisper Fallback Logic
async function transcribeWithWhisper(audioPath, jobId, wordLimit) {
    try {
        console.log(`[Job ${jobId}] Starting Whisper Fallback...`);
        sendProgress(jobId, 'transcribing', 0, 'Gemini blocked content. Switching to Whisper (Fallback)...');

        // Read file buffer
        const audioBuffer = fs.readFileSync(audioPath);

        // Call Hugging Face API
        const response = await hf.automaticSpeechRecognition({
            model: 'openai/whisper-large-v3',
            data: audioBuffer,
        });

        console.log(`[Job ${jobId}] Whisper Raw Output:`, response.text.substring(0, 50) + "...");

        // Convert Whisper text (giant string) to SRT
        const duration = await getAudioDuration(audioPath);
        return convertWhisperTextToSRT(response.text, wordLimit, duration);

    } catch (error) {
        console.error(`[Job ${jobId}] Whisper Error:`, error);
        throw new Error(`Whisper Fallback Failed: ${error.message}`);
    }
}

// Convert Whisper plain text to SRT (Approximate timing)
function convertWhisperTextToSRT(text, wordLimit = 8, audioDuration = null) {
    const words = text.trim().split(/\s+/);
    const segments = [];
    const totalWords = words.length;

    // Estimate time per word
    const timePerWord = audioDuration ? (audioDuration / totalWords) : 0.4; // 0.4s default backup

    let currentTime = 0;

    for (let i = 0; i < words.length; i += wordLimit) {
        const segmentWords = words.slice(i, i + wordLimit);
        const segmentText = segmentWords.join(' ');

        const segmentDuration = segmentWords.length * timePerWord;
        const endTime = currentTime + segmentDuration;

        segments.push({
            start: formatTimestamp(currentTime),
            end: formatTimestamp(endTime),
            text: segmentText
        });

        currentTime = endTime;
    }

    // Build SRT with CRLF line endings for Premiere Pro compatibility
    const srtOutput = segments.map((seg, idx) => `${idx + 1}\r\n${seg.start} --> ${seg.end}\r\n${seg.text}\r\n\r\n`).join('');
    return srtOutput.trimEnd() + '\r\n';
}


// --- HISTORY STORAGE ---
const HISTORY_FILE = path.join(__dirname, 'history.json');
const MAX_HISTORY_ITEMS = 50;

// Helper: Read history from file
function readHistory() {
    try {
        if (fs.existsSync(HISTORY_FILE)) {
            const data = fs.readFileSync(HISTORY_FILE, 'utf8');
            return JSON.parse(data);
        }
        return [];
    } catch (error) {
        console.error('Error reading history:', error);
        return [];
    }
}

// Helper: Write history to file
function writeHistory(history) {
    try {
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2), 'utf8');
        return true;
    } catch (error) {
        console.error('Error writing history:', error);
        return false;
    }
}

// GET /api/history - Retrieve all history
app.get('/api/history', (req, res) => {
    try {
        const history = readHistory();
        res.json(history);
    } catch (error) {
        console.error('Error fetching history:', error);
        res.status(500).json({ error: 'Failed to fetch history' });
    }
});

// POST /api/history - Save new history entry
app.post('/api/history', (req, res) => {
    try {
        const { name, content, words } = req.body;

        if (!name || !content) {
            return res.status(400).json({ error: 'Missing required fields: name, content' });
        }

        const history = readHistory();

        // Create new entry
        const newEntry = {
            id: Date.now(),
            name,
            content,
            words: words || 8,
            date: new Date().toISOString()
        };

        // Add to beginning and limit to MAX_HISTORY_ITEMS
        history.unshift(newEntry);
        const limitedHistory = history.slice(0, MAX_HISTORY_ITEMS);

        if (writeHistory(limitedHistory)) {
            res.json({ success: true, entry: newEntry });
        } else {
            res.status(500).json({ error: 'Failed to save history' });
        }
    } catch (error) {
        console.error('Error saving history:', error);
        res.status(500).json({ error: 'Failed to save history' });
    }
});

// DELETE /api/history/:id - Delete specific history entry
app.delete('/api/history/:id', (req, res) => {
    try {
        const id = parseInt(req.params.id);
        const history = readHistory();
        const filteredHistory = history.filter(item => item.id !== id);

        if (writeHistory(filteredHistory)) {
            res.json({ success: true });
        } else {
            res.status(500).json({ error: 'Failed to delete history entry' });
        }
    } catch (error) {
        console.error('Error deleting history entry:', error);
        res.status(500).json({ error: 'Failed to delete history entry' });
    }
});

// DELETE /api/history - Clear all history
app.delete('/api/history', (req, res) => {
    try {
        if (writeHistory([])) {
            res.json({ success: true });
        } else {
            res.status(500).json({ error: 'Failed to clear history' });
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        res.status(500).json({ error: 'Failed to clear history' });
    }
});


// --- PROMPT CONSTRUCTION ---
function buildEnhancedPrompt(wordLimit, vocabulary = []) {
    const vocabString = vocabulary.length > 0
        ? `\n\nVOCABULARY LIST (Prioritize these spellings): ${vocabulary.join(', ')}`
        : '';

    return `You are a forensic transcription engine. Your task is to transcribe ONLY HUMAN SPEECH from audio to a JSON array.
${vocabString}

CRITICAL INSTRUCTIONS:
1. ONLY transcribe human speech (spoken words, dialogue, narration)
2. IGNORE all music, instrumental sections, sound effects, and background noise
3. SKIP any portions of audio that contain no human speech
4. If the audio contains music with speech, transcribe ONLY the speech parts
5. If there is NO human speech at all in the audio, return an empty array: []

MULTIPLE / OVERLAPPING SPEAKERS (read carefully — this audio can contain two or more people, sometimes talking at the SAME time):
6. NEVER merge two different speakers into one cue. Give each speaker's utterance its OWN separate cue.
7. Anchor every cue's start/end to the EXACT moment those specific words are spoken. Do NOT let one speaker's words push, pull, or shift another speaker's timing.
8. When two voices overlap, transcribe the clearest / foreground voice for that span, then continue in time order — never blend words from both speakers into a single line.
9. If overlapping speech is genuinely unintelligible, transcribe only what is clearly audible and skip the rest. NEVER invent or guess words to fill an overlap.
10. Output cues in strict CHRONOLOGICAL order. A cue's start time must never be earlier than the previous cue's start time, and no timestamp may exceed the audio's total length.

You MUST return ONLY a valid JSON array. Nothing else. Start with [ and end with ].

EXAMPLE FORMAT (copy this structure exactly):

[{"start":"00:00:00,000","end":"00:00:03,500","text":"Hello, this is the first sentence."},{"start":"00:00:04,000","end":"00:00:07,200","text":"Thank you for that introduction."}]

SEGMENTATION RULES (STRICT - CRITICAL FOR READABILITY):
1. **VISUAL TARGET**: Each segment MUST fit on ONE LINE when displayed on screen.
   - **Maximum 8-10 words** per segment (strict limit)
   - **Maximum 50 characters** per segment (including spaces)
   - Target 42 characters for optimal readability

2. **NATURAL SPEECH PAUSES**: Split at natural pause points in the speaker's speech:
   - Commas (,) - natural breath points
   - Conjunctions (and, but, or, so) - logical connectors
   - After complete thoughts or clauses
   - Where the speaker naturally pauses or takes a breath

3. **PHRASE-BASED SPLITTING**: Split long sentences into natural logical phrases:
   - *Bad*: "Jacob Ken wanted the little girl for himself and his" (21 words, split mid-thought)
   - *Good*: "Jacob Ken wanted the little girl for himself" (9 words) THEN "and his avenue to do that was through Melissa Norby" (10 words)
   - *Bad*: "We search your house, your person, your clothes, we find no connection, guess what? We move on to the next person." (21 words)
   - *Good*: "We search your house, your person, your clothes" (8 words) THEN "we find no connection, guess what?" (6 words) THEN "We move on to the next person." (7 words)

4. **FORBIDDEN ENDINGS**: NEVER end a segment with a "weak" word unless it is the absolute end of the sentence.
   - **Weak Words**: "the", "a", "an", "and", "but", "or", "of", "in", "at", "to", "is", "was", "are", "were", "she", "he", "it", "they", "we", "you", "your", "his", "her"
   - *Wrong*: "... make it look like she" (Break) "was assaulted..."
   - *Right*: "... make it look like she was assaulted"
   - *Wrong*: "... wanted the little girl for himself and his" (ends with "his")
   - *Right*: "... wanted the little girl for himself"

5. **COMPLETE THOUGHTS**: Each segment should express a complete thought or phrase when possible.
   - Prefer splitting at punctuation (commas, semicolons)
   - Split before conjunctions if the sentence is too long
   - Keep subject-verb-object together when under the word limit

6. Do NOT include speaker labels.

REQUIREMENTS:
1. Start with [
2. End with ]
3. Keys: "start", "end", "text"
4. Timestamps: HH:MM:SS,mmm format (Always use 3 digits for milliseconds)
5. VERY IMPORTANT: Do NOT use MM:SS format. Always include Hours (00:).
6. CRITICAL: Respect the 8-10 word and 50 character limits strictly - this ensures single-line display.`;
}


// Handle Upload & Transcribe
app.post('/api/transcribe', upload.single('file'), async (req, res) => {
    const jobId = req.body.jobId;
    const wordLimit = parseInt(req.body.wordLimit) || 12; // Default to 12 if undefined

    // Default response flags
    let usedFallback = false;
    let fallbackReason = null;

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        const inputPath = req.file.path;

        sendProgress(jobId, 'converting', 10, 'Preparing audio...');
        const wavPath = await convertToWav(inputPath);
        const duration = await getAudioDuration(wavPath);

        let finalText = "";
        let finalSrt = "";

        // --- LONG AUDIO HANDLING ---
        // If > 20 minutes (1200s), split into chunks
        // Increasing from small chunks to larger chunks to reduce API calls
        // 10 minute chunks is reasonable for Gemini 2.0 Pro context
        if (duration > 1200) {
            finalSrt = await processLongAudio(wavPath, jobId, async (chunkPath, idx, total) => {
                sendProgress(jobId, 'processing', 20 + Math.floor((idx / total) * 50), `Processing part ${idx + 1}/${total}...`);

                try {
                    // Process Chunk Logic with retry
                    const uploadResponse = await retryWithBackoff(async () => {
                        return await fileManager.uploadFile(chunkPath, {
                            mimeType: "audio/wav",
                            displayName: `Chunk_${idx}`,
                        });
                    });

                    await waitForFileActive(uploadResponse.file);

                    // Generate content with retry logic
                    const result = await retryWithBackoff(async () => {
                        return await model.generateContent([
                            buildEnhancedPrompt(wordLimit),
                            {
                                fileData: {
                                    mimeType: uploadResponse.file.mimeType,
                                    fileUri: uploadResponse.file.uri
                                }
                            }
                        ]);
                    });

                    // Cleanup uploaded file
                    try {
                        await fileManager.deleteFile(uploadResponse.file.name);
                    } catch (e) {
                        console.warn(`[Warning] Failed to delete chunk file: ${e.message}`);
                    }

                    // IMPORTANT: Normalize SRT before returning to ensure timestamps are HH:MM:SS,mmm
                    // This guarantees adjustSrtTimestamps regex works.
                    return jsonToSrt(result.response.text(), wordLimit);
                } catch (error) {
                    console.error(`[Error] Failed to process chunk ${idx + 1}/${total}:`, error.message);
                    sendProgress(jobId, 'error', 0, `Failed to process part ${idx + 1}/${total}: ${error.message}`);
                    throw new Error(`Chunk ${idx + 1} processing failed: ${error.message}`);
                }
            });
        }
        // --- SHORT AUDIO HANDLING ---
        else {
            sendProgress(jobId, 'uploading', 30, 'Uploading to AI...');
            const uploadResponse = await fileManager.uploadFile(wavPath, {
                mimeType: "audio/wav",
                displayName: "Audio File",
            });

            await waitForFileActive(uploadResponse.file);

            sendProgress(jobId, 'transcribing', 60, 'AI is listening...');

            try {
                // Use retry logic for API call
                const result = await retryWithBackoff(async () => {
                    return await model.generateContent([
                        buildEnhancedPrompt(wordLimit),
                        {
                            fileData: {
                                mimeType: uploadResponse.file.mimeType,
                                fileUri: uploadResponse.file.uri
                            }
                        }
                    ]);
                });

                const text = result.response.text();
                // Check for refusal/safety blocks (empty text usually)
                if (!text || text.length < 10) {
                    throw new Error("Gemini returned empty response (Possible safety block).");
                }

                finalSrt = jsonToSrt(text, wordLimit);

            } catch (geminiError) {
                // --- FALLBACK TRIGGER ---
                console.error("Gemini Error:", geminiError);

                // Check if it's a network error
                const isNetworkError = geminiError.message?.includes('fetch failed') ||
                    geminiError.message?.includes('ECONNRESET') ||
                    geminiError.message?.includes('ETIMEDOUT');

                if (process.env.ENABLE_WHISPER_FALLBACK === 'true') {
                    usedFallback = true;
                    fallbackReason = isNetworkError ? "Network Error" : "Safety/Content Block";
                    sendProgress(jobId, 'transcribing', 65, `Switching to Whisper fallback (${fallbackReason})...`);
                    finalSrt = await transcribeWithWhisper(wavPath, jobId, wordLimit);
                } else {
                    throw geminiError; // Rethrow if fallback disabled
                }
            }

            // Cleanup
            try { await fileManager.deleteFile(uploadResponse.file.name); } catch (e) { }
        }


        // Cleanup Temp Files
        try {
            fs.unlinkSync(req.file.path);
            if (fs.existsSync(wavPath)) fs.unlinkSync(wavPath);
        } catch (e) { console.error("Cleanup error:", e); }

        sendProgress(jobId, 'complete', 100, 'Done!');

        // Return SRT + Meta
        res.json({
            srt: finalSrt,
            usedFallback: usedFallback,
            fallbackReason: fallbackReason
        });

    } catch (error) {
        console.error('Processing error:', error);
        sendProgress(jobId, 'error', 0, error.message);
        res.status(500).json({ error: error.message });
    }
});

// Helper: Process Long Audio (Mockup for strict logic application same as short)
async function processLongAudio(wavPath, jobId, processChunkFn) {
    // Implementation would split files and map over them
    // Reusing the same 'processChunkFn' which calls the AI
    const chunks = await splitAudio(wavPath);
    let fullSrt = "";
    let timeOffset = 0;

    for (let i = 0; i < chunks.length; i++) {
        const chunkSrt = await processChunkFn(chunks[i], i, chunks.length);

        // Adjust timestamps for chunk offset
        const chunkDuration = await getAudioDuration(chunks[i]);

        const shiftedSrt = adjustSrtTimestamps(chunkSrt, timeOffset);
        fullSrt += shiftedSrt + "\r\n";

        timeOffset += chunkDuration;

        // Cleanup chunk
        try { fs.unlinkSync(chunks[i]); } catch (e) { }
    }
    return fullSrt;
}

// Helper: Adjust SRT Timestamps
function adjustSrtTimestamps(srtText, offsetSeconds) {
    if (!srtText) return "";
    return srtText.replace(/(\d{2}):(\d{2}):(\d{2}),(\d{3})/g, (match, h, m, s, ms) => {
        let totalSeconds = (parseInt(h) * 3600) + (parseInt(m) * 60) + parseInt(s) + (parseInt(ms) / 1000);
        totalSeconds += offsetSeconds;
        return formatTimestamp(totalSeconds);
    });
}


app.listen(port, () => {
    console.log(`\n══════════════════════════════════════════`);
    console.log(`🎬 SRT-AI Server v2.1.0 (Chunking Enabled)`);
    console.log(`📍 URL:   http://localhost:${port}`);
    console.log(`🤖 Model: ${MODEL_NAME}`);
    console.log(`══════════════════════════════════════════\n`);
    // Setup generic favicon to stop 404 noise
    app.get('/favicon.ico', (req, res) => res.status(204).end());
});
