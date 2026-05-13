import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import os from 'os';
import path from 'path';
import fs from 'fs';
import { promises as fsPromises } from 'fs';
import crypto from 'crypto';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

// --- SDK IMPORTS ---
import { GoogleGenAI } from '@google/genai';
import { HfInference } from '@huggingface/inference';
import { Storage } from '@google-cloud/storage';
import speech from '@google-cloud/speech';

// Load environment variables
dotenv.config();

console.log('--- SERVER STARTING (Gemini 2.x Architecture) ---');

// ═══════════════════════════════════════════════════════════
//   CONFIGURATION
// ═══════════════════════════════════════════════════════════

const PORT = parseInt(process.env.PORT) || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';
const isProduction = NODE_ENV === 'production';

// Default to Gemini 2.5 Pro
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-pro';
console.log('[DEBUG] GEMINI_MODEL:', GEMINI_MODEL);

// GCS + Vertex AI configuration
const GCS_BUCKET_NAME = process.env.GCS_BUCKET_NAME;
const GCP_PROJECT_ID = process.env.GCP_PROJECT_ID;
const VERTEX_REGION = process.env.VERTEX_REGION || 'us-central1';

if (!GCP_PROJECT_ID) {
    console.error('\n[FATAL ERROR] GCP_PROJECT_ID missing in .env\n');
    process.exit(1);
}
if (!GCS_BUCKET_NAME) {
    console.error('\n[FATAL ERROR] GCS_BUCKET_NAME missing in .env\n');
    process.exit(1);
}

// Production Configuration
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS) || 30 * 60 * 1000; // 30 minutes
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE_MB) * 1024 * 1024 || 2 * 1024 * 1024 * 1024; // 2GB
const RATE_LIMIT_REQUESTS = parseInt(process.env.RATE_LIMIT_REQUESTS) || 10; // per hour
const LOG_LEVEL = process.env.LOG_LEVEL || (isProduction ? 'info' : 'debug');

// Supported file formats
const SUPPORTED_AUDIO_FORMATS = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/m4a', 'audio/x-m4a', 'audio/flac', 'audio/ogg'];
const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
const SUPPORTED_FORMATS = [...SUPPORTED_AUDIO_FORMATS, ...SUPPORTED_VIDEO_FORMATS];

// ═══════════════════════════════════════════════════════════
//   SETUP & INITIALIZATION
// ═══════════════════════════════════════════════════════════

// 1. Initialize Google GenAI Client (Vertex AI)
// httpOptions.timeout raised to 25 min — long audio files need more than the default 5 min
const aiClient = new GoogleGenAI({
    vertexai: true,
    project: GCP_PROJECT_ID,
    location: VERTEX_REGION,
    httpOptions: { timeout: 25 * 60 * 1000 },
});

// 2. Initialize Google Cloud Storage Client
const gcsClient = new Storage();
const gcsBucket = gcsClient.bucket(GCS_BUCKET_NAME);

// 3. Initialize Speech-to-Text Clients
//    - V1 client: longRunningRecognize with 'latest_long' (reliable fallback)
//    - V2 client: batchRecognize with 'chirp_2' (Google's newest model — catches
//      ~10-30% more words than latest_long on challenging audio). V2 requires a
//      regional endpoint, so we bind it to VERTEX_REGION.
const sttClient = new speech.SpeechClient();
const sttV2Client = new speech.v2.SpeechClient({
    apiEndpoint: `${VERTEX_REGION}-speech.googleapis.com`,
});

// 2. Initialize Hugging Face Client (Whisper Fallback)
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const ENABLE_WHISPER_FALLBACK = process.env.ENABLE_WHISPER_FALLBACK !== 'false';
const hfClient = HUGGINGFACE_API_KEY ? new HfInference(HUGGINGFACE_API_KEY) : null;

if (ENABLE_WHISPER_FALLBACK && !hfClient) {
    console.warn('[WARNING] Whisper fallback is enabled but HUGGINGFACE_API_KEY is missing.');
    console.warn('Blocked content will not have a fallback option.');
}

// 3. Setup FFmpeg (try static first, fallback to system)
let ffmpegPath = ffmpegStatic;
if (!ffmpegPath) {
    ffmpegPath = 'ffmpeg'; // Use system FFmpeg
    console.log('[Setup] Using system FFmpeg');
} else {
    console.log(`[Setup] Using bundled FFmpeg: ${ffmpegPath}`);
}
ffmpeg.setFfmpegPath(ffmpegPath);

// Test FFmpeg availability
try {
    execSync(`"${ffmpegPath}" -version`, { stdio: 'ignore' });
    console.log('[Setup] FFmpeg is working ✓');
} catch (e) {
    console.error('[WARNING] FFmpeg not found! Install it with: apt install ffmpeg');
}

// 3. Express App & Middleware
const app = express();

if (isProduction) {
    app.set('trust proxy', 1);
}

app.use(cors({
    origin: true,
    credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Keep-alive for long uploads
app.use((req, res, next) => {
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Keep-Alive', 'timeout=1800');
    next();
});

// Request logging middleware
app.use((req, res, next) => {
    const start = Date.now();
    const requestId = crypto.randomUUID().substring(0, 8);
    req.requestId = requestId;

    log('info', `[${requestId}] ${req.method} ${req.path}`, {
        ip: req.ip,
        userAgent: req.get('user-agent')?.substring(0, 100)
    });

    res.on('finish', () => {
        const duration = Date.now() - start;
        log('info', `[${requestId}] ${res.statusCode} ${duration}ms`);
    });

    next();
});

// Request timeout middleware
app.use((req, res, next) => {
    req.setTimeout(REQUEST_TIMEOUT_MS, () => {
        log('error', `Request timeout after ${REQUEST_TIMEOUT_MS}ms`, { path: req.path });
        if (!res.headersSent) {
            res.status(408).json({
                error: 'Request timeout',
                message: 'Processing took too long. Please try with a shorter file or reduce quality.',
                timeout: REQUEST_TIMEOUT_MS
            });
        }
    });
    next();
});

// 4. Static Files (Frontend)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const staticRoot = process.env.STATIC_ROOT || __dirname;
app.use(express.static(staticRoot));

// 5. Upload Configuration (Supports 1+ hour files)
const upload = multer({
    storage: multer.diskStorage({
        destination: (req, file, cb) => {
            const tmpDir = os.tmpdir();
            // Ensure temp directory exists and is writable
            try {
                if (!fs.existsSync(tmpDir)) {
                    fs.mkdirSync(tmpDir, { recursive: true });
                }
                cb(null, tmpDir);
            } catch (err) {
                log('error', 'Failed to access temp directory', { error: err.message });
                cb(new Error('Server storage error. Please try again.'));
            }
        },
        filename: (req, file, cb) => {
            const ext = path.extname(file.originalname) || '.bin';
            const safeName = `srt-ai-${crypto.randomUUID()}${ext}`;
            cb(null, safeName);
        }
    }),
    limits: {
        fileSize: MAX_FILE_SIZE,
        files: 1,
        fields: 10,
        parts: 20
    },
    fileFilter: (req, file, cb) => {
        // Basic MIME type check (will do deeper validation later)
        if (SUPPORTED_FORMATS.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported file format: ${file.mimetype}. Supported formats: MP4, MP3, WAV, MOV, AVI, M4A, FLAC, OGG`));
        }
    }
});

// Rate limiting store
const rateLimitStore = new Map();

function checkRateLimit(ip) {
    const now = Date.now();
    const hourAgo = now - 60 * 60 * 1000;

    if (!rateLimitStore.has(ip)) {
        rateLimitStore.set(ip, []);
    }

    const requests = rateLimitStore.get(ip).filter(time => time > hourAgo);

    if (requests.length >= RATE_LIMIT_REQUESTS) {
        return false;
    }

    requests.push(now);
    rateLimitStore.set(ip, requests);
    return true;
}

// Clean up rate limit store every hour
setInterval(() => {
    const hourAgo = Date.now() - 60 * 60 * 1000;
    for (const [ip, requests] of rateLimitStore.entries()) {
        const filtered = requests.filter(time => time > hourAgo);
        if (filtered.length === 0) {
            rateLimitStore.delete(ip);
        } else {
            rateLimitStore.set(ip, filtered);
        }
    }
}, 60 * 60 * 1000);

// ═══════════════════════════════════════════════════════════
//   LOGGING SYSTEM
// ═══════════════════════════════════════════════════════════

const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };
const currentLogLevel = LOG_LEVELS[LOG_LEVEL] || LOG_LEVELS.info;

function log(level, message, metadata = {}) {
    if (LOG_LEVELS[level] < currentLogLevel) return;

    const timestamp = new Date().toISOString();
    const metaStr = Object.keys(metadata).length > 0 ? ` ${JSON.stringify(metadata)}` : '';
    console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}${metaStr}`);
}

// ═══════════════════════════════════════════════════════════
//   PROGRESS TRACKING SYSTEM
// ═══════════════════════════════════════════════════════════

// Store active jobs and their progress
const activeJobs = new Map();

// SSE clients for progress updates
const progressClients = new Map();

function updateProgress(jobId, stage, percent, message) {
    const progress = { stage, percent, message, timestamp: Date.now() };
    activeJobs.set(jobId, progress);

    // Send to connected SSE clients
    const client = progressClients.get(jobId);
    if (client) {
        client.write(`data: ${JSON.stringify(progress)}\n\n`);
    }

    log('debug', `[Progress] ${jobId.substring(0, 8)}... | ${stage} | ${percent}% | ${message}`);
}

// ═══════════════════════════════════════════════════════════
//   API ROUTES
// ═══════════════════════════════════════════════════════════

/**
 * Health Check
 */
app.get('/api/health', async (req, res) => {
    const health = {
        status: 'online',
        system: 'Gemini 2.x Architecture',
        model: GEMINI_MODEL,
        time: new Date().toISOString(),
        version: '2.0.0',
        environment: NODE_ENV,
        checks: {}
    };

    // Check FFmpeg
    try {
        execSync(`"${ffmpegPath}" -version`, { stdio: 'ignore' });
        health.checks.ffmpeg = 'ok';
    } catch (e) {
        health.checks.ffmpeg = 'error';
        health.status = 'degraded';
    }

    // Check Gemini API
    try {
        await aiClient.models.list();
        health.checks.geminiApi = 'ok';
    } catch (e) {
        health.checks.geminiApi = 'error';
        health.status = 'degraded';
        log('error', 'Gemini API health check failed', { error: e.message });
    }

    // System resources
    health.system = {
        platform: os.platform(),
        nodeVersion: process.version,
        memory: {
            total: Math.round(os.totalmem() / 1024 / 1024) + 'MB',
            free: Math.round(os.freemem() / 1024 / 1024) + 'MB',
            used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + 'MB'
        },
        uptime: Math.round(process.uptime()) + 's'
    };

    res.json(health);
});

/**
 * Progress Stream (SSE)
 */
app.get('/api/progress/:jobId', (req, res) => {
    const { jobId } = req.params;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    // Send initial connection message
    res.write(`data: ${JSON.stringify({ stage: 'connected', percent: 0, message: 'Connected to progress stream' })}\n\n`);

    // Store this client
    progressClients.set(jobId, res);

    // Send current progress if job already started
    if (activeJobs.has(jobId)) {
        res.write(`data: ${JSON.stringify(activeJobs.get(jobId))}\n\n`);
    }

    // Cleanup on disconnect
    req.on('close', () => {
        progressClients.delete(jobId);
    });
});

/**
 * Main Transcription Endpoint
 */
// Upload error handling middleware
app.use('/api/transcribe', (err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        log('warn', 'Multer upload error', { error: err.message, code: err.code });

        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                error: 'file_too_large',
                message: `File size exceeds the maximum limit of ${Math.round(MAX_FILE_SIZE / 1024 / 1024)}MB. Please compress or split your file.`,
                maxSize: MAX_FILE_SIZE
            });
        }

        if (err.code === 'LIMIT_UNEXPECTED_FILE') {
            return res.status(400).json({
                error: 'invalid_upload',
                message: 'Invalid file upload. Please upload only one file at a time.'
            });
        }

        return res.status(400).json({
            error: 'upload_error',
            message: `Upload failed: ${err.message}. Please try again.`,
            code: err.code
        });
    }

    if (err) {
        log('error', 'Upload error', { error: err.message });
        return res.status(500).json({
            error: 'upload_failed',
            message: err.message || 'File upload failed. Please try again.'
        });
    }

    next();
});

app.post('/api/transcribe', upload.single('file'), async (req, res) => {
    const jobId = req.body.jobId || crypto.randomUUID();
    const startTime = Date.now();
    const cleanupPaths = [];
    let geminiFileName = null;

    try {
        // 1. Rate Limiting
        const clientIp = req.ip;
        if (!checkRateLimit(clientIp)) {
            log('warn', 'Rate limit exceeded', { ip: clientIp, jobId });
            return res.status(429).json({
                error: 'Rate limit exceeded',
                message: `Too many requests. Maximum ${RATE_LIMIT_REQUESTS} requests per hour allowed. Please try again later.`,
                retryAfter: 3600
            });
        }

        // 2. File Upload Validation
        if (!req.file) {
            return res.status(400).json({
                error: 'No file uploaded',
                message: 'Please select a file to upload.'
            });
        }

        cleanupPaths.push(req.file.path);

        log('info', `New transcription request`, {
            jobId: jobId.substring(0, 8),
            filename: req.file.originalname,
            size: Math.round(req.file.size / 1024 / 1024) + 'MB',
            mimetype: req.file.mimetype,
            ip: clientIp
        });

        // 3. Enhanced File Validation
        const validationError = await validateFile(req.file);
        if (validationError) {
            log('warn', 'File validation failed', { jobId, error: validationError });
            return res.status(400).json({
                error: 'Invalid file',
                message: validationError,
                supportedFormats: 'MP4, MP3, WAV, MOV, AVI, M4A, FLAC, OGG',
                maxSize: '2GB'
            });
        }

        updateProgress(jobId, 'upload', 5, 'File validated, starting processing...');

        // -------------------------------------------------------
        // STEP 1: Audio Pre-processing (Forensic Cleaning)
        // -------------------------------------------------------
        let audioPath = req.file.path;
        const isWav = req.file.mimetype === 'audio/wav' || req.file.originalname.toLowerCase().endsWith('.wav');

        if (!isWav) {
            updateProgress(jobId, 'converting', 10, 'Converting audio to high-quality WAV...');
            audioPath = await transcodeToWav(req.file.path, (percent) => {
                // Map 0-100 FFmpeg progress to 10-40 overall progress
                const overallPercent = 10 + Math.round(percent * 0.3);
                updateProgress(jobId, 'converting', overallPercent, `Converting audio: ${percent.toFixed(0)}%`);
            });
            cleanupPaths.push(audioPath);
        } else {
            updateProgress(jobId, 'converting', 40, 'File is already WAV, skipping conversion');
        }

        // Get audio duration for prompt accuracy and timestamp validation
        const audioDuration = await getAudioDuration(audioPath);
        log('info', `Audio duration: ${audioDuration.toFixed(1)}s`, { jobId: jobId.substring(0, 8) });

        // NOTE: FFmpeg speech-boundary alignment was tested but removed —
        // it worsened timestamps on audio with background music/ambient sound.

        // -------------------------------------------------------
        // STEP 2: Upload audio to GCS bucket
        // Vertex AI uses gs:// URIs instead of File API
        // This gives the AI server-side processed audio = accurate timestamps
        // (equivalent to old GoogleAIFileManager.uploadFile() approach)
        // -------------------------------------------------------
        updateProgress(jobId, 'uploading', 45, 'Uploading audio to Cloud Storage...');

        const gcsFileName = `srt-ai/${jobId}-${Date.now()}.wav`;
        geminiFileName = gcsFileName; // Store for cleanup in finally block

        try {
            await gcsBucket.upload(audioPath, {
                destination: gcsFileName,
                metadata: { contentType: 'audio/wav' },
            });
        } catch (uploadErr) {
            log('error', 'GCS upload failed', { error: uploadErr.message });
            throw new Error(`Failed to upload audio to Cloud Storage: ${uploadErr.message}`);
        }

        const gcsUri = `gs://${GCS_BUCKET_NAME}/${gcsFileName}`;
        log('info', `Audio uploaded to GCS`, { gcsUri, jobId: jobId.substring(0, 8) });

        const wordLimit = parseInt(req.body.wordLimit) || 8;
        const language = req.body.language || 'auto';
        const vocabulary = req.body.vocabulary ? req.body.vocabulary.split(',').map(v => v.trim()).filter(v => v) : [];

        // -------------------------------------------------------
        // STEP 3 (PRIMARY): Google Speech-to-Text — waveform-accurate
        // word-level timings AND segmentation at real audio pauses.
        // Trying to merge Gemini's text into STT timings produced worse
        // alignment than pure STT, so we keep this path simple.
        //
        // Skipped entirely when TRANSCRIPTION_MODE=vertex — the user explicitly
        // wants Gemini-on-Vertex (e.g. when they have credits there and have
        // not provisioned STT IAM / HF inference permissions).
        // -------------------------------------------------------
        const transcriptionMode = (process.env.TRANSCRIPTION_MODE || 'auto').toLowerCase();
        let jsonResponse = null;
        let usedSTT = false;

        let sttWords = null;
        let sttModel = null;

        if (transcriptionMode === 'vertex') {
            log('info', 'TRANSCRIPTION_MODE=vertex — skipping STT cascade, using Gemini directly', { jobId: jobId.substring(0, 8) });
        } else {
            // Tier 1: HuggingFace Whisper-large-v3 (same model TurboScribe uses —
            // highest word coverage). Constrained by HF's 25MB upload limit, so
            // only works on shorter clips (~13 min of 16kHz mono WAV).
            if (hfClient) {
                try {
                    updateProgress(jobId, 'transcribing', 55, 'Transcribing with Whisper-large-v3...');
                    sttWords = await getWordTimestampsWhisperHF(audioPath);
                    if (sttWords && sttWords.length > 0) {
                        sttModel = 'whisper-large-v3 (HF)';
                    } else {
                        sttWords = null;
                    }
                } catch (hfErr) {
                    log('warn', 'HF Whisper unavailable — trying chirp_2', {
                        jobId: jobId.substring(0, 8),
                        error: hfErr.message,
                    });
                }
            }

            // Tier 2: chirp_2 (Google's newest model — V2 API)
            if (!sttWords) {
                try {
                    updateProgress(jobId, 'transcribing', 60, 'Analyzing speech with chirp_2 model...');
                    sttWords = await getWordTimestampsChirp2(gcsUri, language);
                    if (sttWords && sttWords.length > 0) {
                        sttModel = 'chirp_2';
                    } else {
                        log('warn', 'chirp_2 returned no words — trying latest_long', { jobId: jobId.substring(0, 8) });
                        sttWords = null;
                    }
                } catch (chirpErr) {
                    log('warn', 'chirp_2 failed — trying latest_long', {
                        jobId: jobId.substring(0, 8),
                        error: chirpErr.message,
                    });
                }
            }

            // Tier 3: latest_long (V1 API — reliable baseline)
            if (!sttWords) {
                try {
                    updateProgress(jobId, 'transcribing', 65, 'Analyzing speech with latest_long model...');
                    sttWords = await getWordTimestamps(gcsUri, language);
                    if (sttWords && sttWords.length > 0) {
                        sttModel = 'latest_long';
                    }
                } catch (sttErr) {
                    log('warn', 'latest_long also failed — falling back to Gemini', {
                        jobId: jobId.substring(0, 8),
                        error: sttErr.message,
                    });
                }
            }
        }

        if (sttWords && sttWords.length > 0) {
            jsonResponse = buildSegmentsFromSTT(sttWords, wordLimit);
            usedSTT = true;
            log('info', `STT transcription successful via ${sttModel}`, {
                jobId: jobId.substring(0, 8),
                wordCount: sttWords.length,
                model: sttModel,
            });
        }

        // -------------------------------------------------------
        // STEP 3 (FALLBACK): Gemini via Vertex AI — used only when STT
        // fails. Timestamps will be estimated, less accurate than STT.
        // -------------------------------------------------------
        if (!jsonResponse) {
        updateProgress(jobId, 'transcribing', 75, `Transcribing with ${GEMINI_MODEL}...`);

        // -- Long-audio path: chunk into 8-min pieces and call Vertex per chunk.
        // Vertex Gemini-2.5-Pro silently truncates audio >~15 min when sent in one
        // request. Chunking forces the model to fully process every region.
        let response = null;
        const useVertexChunking = audioDuration > VERTEX_CHUNK_THRESHOLD_SEC;

        if (useVertexChunking) {
            log('info', `Audio ${audioDuration.toFixed(1)}s > ${VERTEX_CHUNK_THRESHOLD_SEC}s — using Vertex chunking path`, { jobId: jobId.substring(0, 8) });
            jsonResponse = await transcribeVertexInChunks(
                audioPath, audioDuration, wordLimit, vocabulary, language, jobId
            );
            // Skip the rest of the single-shot block — jsonResponse is set.
        } else {

        let transcriptionAttempts = 0;

        while (transcriptionAttempts < 2) {
            try {
                log('info', `Transcription attempt ${transcriptionAttempts + 1}`, { jobId: jobId.substring(0, 8), model: GEMINI_MODEL });
                response = await aiClient.models.generateContent({
                    model: GEMINI_MODEL,
                    config: {
                        temperature: 0,  // Fully deterministic — critical for consistent timestamps
                        topP: 0.95,
                        maxOutputTokens: 65536,  // Gemini 2.5 Pro cap — avoid mid-transcription truncation on long audio
                        responseMimeType: 'application/json',  // Force clean JSON (no ```json fences)
                        // Match AI Studio permissiveness so transcription isn't silently
                        // rewritten/dropped by Vertex's stricter default safety filters.
                        safetySettings: VERTEX_SAFETY_SETTINGS,
                    },
                    contents: [
                        {
                            role: 'user',
                            parts: [
                                { text: buildEnhancedPrompt(wordLimit, vocabulary, audioDuration, language) },
                                { fileData: { mimeType: 'audio/wav', fileUri: gcsUri } }
                            ]
                        }
                    ]
                });
                break; // Success
            } catch (genError) {
                transcriptionAttempts++;
                log('warn', `Transcription attempt ${transcriptionAttempts} failed`, { error: genError.message });
                if (transcriptionAttempts >= 2) {
                    throw new Error(`AI transcription failed: ${genError.message}`);
                }
                // Wait 15s before retry (not 2s — network errors need a real pause)
                await new Promise(resolve => setTimeout(resolve, 15000));
                updateProgress(jobId, 'transcribing', 76, `Retrying transcription...`);
            }
        }
        }  // end of single-shot else branch

        updateProgress(jobId, 'formatting', 90, 'Formatting SRT output...');

        // -------------------------------------------------------
        // STEP 5: Convert JSON to SRT (Perfect Formatting)
        // -------------------------------------------------------

        // Skip single-shot response parsing if chunked path already produced jsonResponse
        if (!useVertexChunking) {

        // Debug: Check for safety blocks or empty candidates
        if (response.promptFeedback?.blockReason) {
            log('warn', 'Content blocked by Gemini safety filters', {
                blockReason: response.promptFeedback.blockReason,
                safetyRatings: response.promptFeedback.safetyRatings
            });

            // Try Whisper fallback if available
            if (ENABLE_WHISPER_FALLBACK && hfClient) {
                log('info', 'Attempting Whisper fallback for blocked content', { jobId: jobId.substring(0, 8) });

                try {
                    const whisperSRT = await transcribeWithWhisper(wavPath, wordLimit, jobId);

                    // Success! Return the Whisper result
                    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                    updateProgress(jobId, 'complete', 100, `Completed via Whisper in ${duration}s`);

                    const subtitleCount = (whisperSRT.match(/\n\n/g) || []).length + 1;
                    log('info', 'Transcription completed successfully via Whisper fallback', {
                        jobId: jobId.substring(0, 8),
                        duration: `${duration}s`,
                        subtitleCount
                    });

                    return res.json({
                        success: true,
                        srt: whisperSRT,
                        duration: `${duration}s`,
                        subtitleCount,
                        usedFallback: true,
                        fallbackReason: 'Content blocked by Gemini safety filters'
                    });
                } catch (whisperError) {
                    log('error', 'Whisper fallback also failed', { error: whisperError.message });
                    throw new Error(`Content blocked by AI safety filters and Whisper fallback failed: ${whisperError.message}`);
                }
            }

            // No fallback available
            throw new Error(`Content blocked by AI safety filters: ${response.promptFeedback.blockReason}. Please try a different audio file.`);
        }

        if (!response.candidates || response.candidates.length === 0) {
            log('error', 'No candidates in response', {
                hasPromptFeedback: !!response.promptFeedback,
                responseKeys: Object.keys(response)
            });
            throw new Error('AI did not generate any response. This may be due to audio quality issues or content restrictions. Please try a different audio file.');
        }

        // Check if candidate was blocked
        const candidate = response.candidates[0];
        if (candidate.finishReason && candidate.finishReason !== 'STOP') {
            log('warn', 'Generation stopped early', {
                finishReason: candidate.finishReason,
                safetyRatings: candidate.safetyRatings
            });

            if (candidate.finishReason === 'SAFETY') {
                throw new Error('Content generation stopped due to safety concerns. Please try a different audio file.');
            }

            if (candidate.finishReason === 'MAX_TOKENS') {
                log('warn', 'Transcription incomplete due to MAX_TOKENS limit', {
                    jobId: jobId.substring(0, 8),
                    message: 'Audio file is too long for single transcription. Output will be incomplete.'
                });
                // Continue processing but the output will be incomplete
                // The user will see a warning in the logs
            }
        }

        // Debug: Log the response structure
        log('debug', 'Response object keys', { keys: Object.keys(response || {}) });
        log('debug', 'Response type', { type: typeof response, hasText: !!response?.text });

        const rawResponse = response?.text || response?.candidates?.[0]?.content?.parts?.[0]?.text || '';
        log('debug', 'AI response received', { length: rawResponse.length, preview: rawResponse.substring(0, 200) });

        jsonResponse = cleanOutput(rawResponse);

        if (!jsonResponse || jsonResponse.trim().length === 0) {
            log('error', 'Empty response after cleaning', {
                rawLength: rawResponse.length,
                cleanedLength: jsonResponse?.length || 0,
                responseStructure: JSON.stringify(response).substring(0, 500)
            });
            throw new Error('Empty response from AI. Please try again or use a different audio file.');
        }
        }  // end of !useVertexChunking branch
        } // end of Gemini block

        if (!jsonResponse || jsonResponse.trim() === '[]') {
            log('warn', 'No speech detected in audio', { jobId });
            throw new Error('No human speech detected in the audio file. The file may contain only music, instrumental audio, or be too quiet. Please ensure your audio contains clear spoken words.');
        }

        // -------------------------------------------------------
        // STEP 4: Convert transcription JSON → SRT
        // jsonResponse is from STT (primary) or Gemini fallback.
        // -------------------------------------------------------
        log('info', `Transcription source: ${usedSTT ? 'Speech-to-Text' : 'Gemini'}`, { jobId: jobId.substring(0, 8) });

        const { srt: srtOutput, timingReport } = jsonToSrt(jsonResponse, wordLimit, audioDuration);

        if (!srtOutput || srtOutput.includes('[Error]')) {
            log('error', 'SRT conversion failed', { jsonResponse: jsonResponse.substring(0, 200) });
            throw new Error('Failed to generate subtitles. The audio may be unclear or too short.');
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        updateProgress(jobId, 'complete', 100, `Completed in ${duration}s`);

        log('info', `Job completed successfully`, {
            jobId: jobId.substring(0, 8),
            duration: duration + 's',
            subtitleCount: srtOutput.split('\n\n').length
        });

        res.json({ srt: srtOutput, timingReport, jobId, duration, language });

    } catch (error) {
        log('error', 'Transcription failed', {
            jobId: jobId.substring(0, 8),
            error: error.message,
            stack: error.stack?.substring(0, 500)
        });

        updateProgress(jobId, 'error', 0, error.message);

        // Categorize errors and provide user-friendly messages
        let statusCode = 500;
        let userMessage = 'An unexpected error occurred. Please try again.';
        let errorType = 'internal_error';

        if (error.message.includes('quota') || error.message.includes('429')) {
            statusCode = 429;
            userMessage = 'API quota exceeded. Please try again in a few minutes.';
            errorType = 'quota_exceeded';
        } else if (error.message.includes('timeout') || error.message.includes('ETIMEDOUT')) {
            statusCode = 408;
            userMessage = 'Processing timeout. Please try with a shorter file or reduce quality.';
            errorType = 'timeout';
        } else if (error.message.includes('ENOENT') || error.message.includes('file')) {
            statusCode = 400;
            userMessage = 'File processing error. The file may be corrupted or in an unsupported format.';
            errorType = 'file_error';
        } else if (error.message.includes('FFmpeg') || error.message.includes('codec')) {
            statusCode = 400;
            userMessage = 'Audio conversion failed. Please ensure the file is a valid audio/video file.';
            errorType = 'conversion_error';
        } else if (error.message.includes('FAILED')) {
            statusCode = 500;
            userMessage = 'Gemini AI processing failed. Please try again or use a different file.';
            errorType = 'ai_processing_error';
        }

        if (!res.headersSent) {
            res.status(statusCode).json({
                error: errorType,
                message: userMessage,
                details: isProduction ? undefined : error.message,
                jobId,
                timestamp: new Date().toISOString()
            });
        }

    } finally {
        // Cleanup
        for (const p of cleanupPaths) {
            await deleteLocalFile(p);
        }

        // Cleanup GCS file
        if (geminiFileName) {
            try {
                await gcsBucket.file(geminiFileName).delete();
                log('debug', 'Deleted GCS file', { file: geminiFileName });
            } catch (e) {
                log('warn', 'Failed to delete GCS file (will auto-expire)', { file: geminiFileName, error: e.message });
            }
        }

        // Close SSE connection
        const client = progressClients.get(jobId);
        if (client) {
            client.write(`data: ${JSON.stringify({ stage: 'done', percent: 100, message: 'Stream closed' })}\n\n`);
            progressClients.delete(jobId);
        }

        // Remove job from tracking after 5 minutes
        setTimeout(() => activeJobs.delete(jobId), 5 * 60 * 1000);
    }
});

// Fallback for SPA
app.get('*', (req, res) => {
    res.sendFile(path.join(staticRoot, 'index.html'));
});

// ═══════════════════════════════════════════════════════════
//   SPEECH-TO-TEXT (accurate word-level timestamps)
// ═══════════════════════════════════════════════════════════

// Map user-facing language names → BCP-47 codes for Speech-to-Text
const LANG_CODE_MAP = {
    'english': 'en-US', 'hindi': 'hi-IN', 'spanish': 'es-ES',
    'french': 'fr-FR', 'german': 'de-DE', 'japanese': 'ja-JP',
    'chinese': 'zh-CN', 'korean': 'ko-KR', 'arabic': 'ar-SA',
    'portuguese': 'pt-BR', 'russian': 'ru-RU', 'italian': 'it-IT',
    'dutch': 'nl-NL', 'turkish': 'tr-TR', 'polish': 'pl-PL',
    'thai': 'th-TH', 'vietnamese': 'vi-VN', 'indonesian': 'id-ID',
    'malay': 'ms-MY', 'tamil': 'ta-IN', 'telugu': 'te-IN',
    'bengali': 'bn-IN', 'urdu': 'ur-PK', 'marathi': 'mr-IN',
    'gujarati': 'gu-IN', 'kannada': 'kn-IN', 'malayalam': 'ml-IN',
    'punjabi': 'pa-IN', 'swedish': 'sv-SE', 'norwegian': 'no-NO',
    'danish': 'da-DK', 'finnish': 'fi-FI', 'czech': 'cs-CZ',
    'romanian': 'ro-RO', 'hungarian': 'hu-HU', 'greek': 'el-GR',
    'hebrew': 'he-IL', 'ukrainian': 'uk-UA', 'filipino': 'fil-PH',
};

function resolveLanguageCode(lang) {
    if (!lang || lang === 'auto') return 'en-US';
    // Already a BCP-47 code like "en-US"
    if (/^[a-z]{2}(-[A-Z]{2})?$/.test(lang)) return lang;
    return LANG_CODE_MAP[lang.toLowerCase()] || 'en-US';
}

/**
 * Call Google Cloud Speech-to-Text to get word-level timestamps.
 * Uses longRunningRecognize (works for any audio length).
 * @returns {Array<{word: string, startTime: number, endTime: number}>}
 */
async function getWordTimestamps(gcsUri, language = 'auto') {
    const langCode = resolveLanguageCode(language);

    const request = {
        audio: { uri: gcsUri },
        config: {
            encoding: 'LINEAR16',
            sampleRateHertz: 16000,
            languageCode: langCode,
            enableWordTimeOffsets: true,
            enableAutomaticPunctuation: true,
            // 'latest_long' is Google's current best general-purpose
            // long-form model. For clean English narration it catches
            // more words than the media-tuned 'video' model.
            model: 'latest_long',
            enableWordConfidence: true,
        },
    };

    const [operation] = await sttClient.longRunningRecognize(request);
    const [response] = await operation.promise();

    const words = [];
    for (const result of (response.results || [])) {
        const alt = result.alternatives && result.alternatives[0];
        if (!alt || !alt.words) continue;
        for (const w of alt.words) {
            words.push({
                word: w.word,
                startTime: parseDuration(w.startTime),
                endTime: parseDuration(w.endTime),
            });
        }
    }
    return words;
}

/** Parse protobuf Duration ({seconds, nanos}) → float seconds */
function parseDuration(d) {
    if (!d) return 0;
    const sec = parseInt(d.seconds || '0', 10);
    const nano = parseInt(d.nanos || '0', 10);
    return sec + nano / 1e9;
}

// ─── HF Whisper chunking constants ─────────────────────────────────────
// 16kHz mono PCM s16le = ~32 KB/s. HF Inference API caps uploads at ~25MB.
// 660s (11 min) = ~21MB which leaves margin for the WAV header + HTTP overhead.
const WHISPER_CHUNK_TARGET_SEC = 660;
const WHISPER_MAX_FILE_MB = 22;   // safety margin under HF's ~25MB hard limit
const WHISPER_CHUNK_TOLERANCE_SEC = 60;   // search ±60s of target for a silence point
// ───────────────────────────────────────────────────────────────────────

/**
 * Use ffmpeg silencedetect to find silence regions. Returns array of
 * {start, end} times in seconds. Used to pick natural split points for
 * Whisper chunking — cutting mid-word loses ~1 word at every boundary.
 */
async function detectSilences(audioPath, noiseDb = -30, minDurationSec = 0.4) {
    return new Promise((resolve) => {
        const silences = [];
        let pendingStart = null;
        const proc = ffmpeg(audioPath)
            .audioFilters(`silencedetect=noise=${noiseDb}dB:d=${minDurationSec}`)
            .format('null')
            .output('-')
            .on('stderr', (line) => {
                let m = line.match(/silence_start:\s*(-?[\d.]+)/);
                if (m) pendingStart = Math.max(0, parseFloat(m[1]));
                m = line.match(/silence_end:\s*([\d.]+)/);
                if (m && pendingStart !== null) {
                    silences.push({ start: pendingStart, end: parseFloat(m[1]) });
                    pendingStart = null;
                }
            })
            .on('end', () => resolve(silences))
            .on('error', () => resolve([]));   // detection failure → fall back to fixed splits
        proc.run();
    });
}

/**
 * Compute split points for an audio file: zero, intermediate boundaries
 * (snapped to silence when possible), and totalDuration. Returns absolute
 * timestamps in seconds.
 */
function computeWhisperSplitPoints(totalDuration, silences, chunkTarget = WHISPER_CHUNK_TARGET_SEC) {
    const splits = [0];
    let target = chunkTarget;
    while (target < totalDuration - 30) {   // keep last chunk ≥30s
        // Find silences near the target, after the previous split
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

/**
 * Extract a chunk of WAV from inputPath into a new temp file. Re-encodes
 * to 16kHz mono PCM to match what HF Whisper expects and keep chunks small.
 * Returns the new path.
 */
async function extractWavChunk(inputPath, startSec, durationSec) {
    const outPath = path.join(os.tmpdir(), `srt-ai-whisperchunk-${crypto.randomUUID()}.wav`);
    await new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .setStartTime(startSec)      // ffmpeg places -ss AFTER -i with this API → sample-accurate
            .setDuration(durationSec)
            .audioCodec('pcm_s16le')
            .audioFrequency(16000)
            .audioChannels(1)
            .format('wav')
            .output(outPath)
            .on('end', resolve)
            .on('error', reject)
            .run();
    });
    return outPath;
}

/**
 * Call HF Whisper once for a single chunk file. Returns words with
 * timestamps shifted by `offsetSec` so the caller can concatenate chunks
 * and get global-time word boundaries.
 */
async function _whisperTranscribeOnce(audioPath, offsetSec = 0) {
    const audioBuffer = await fsPromises.readFile(audioPath);
    // MIME type MUST be set explicitly — HF routes whisper-large-v3 through
    // the fal-ai provider which rejects Blobs without a content-type.
    const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });

    // Force HF's first-party serverless endpoint. HF Inference now auto-routes
    // popular models through third-party providers (fal-ai, etc.) which require
    // an extra token permission ("Make calls to Inference Providers"). Forcing
    // `hf-inference` keeps the original behaviour and avoids that permission.
    const result = await hfClient.automaticSpeechRecognition({
        data: audioBlob,
        model: 'openai/whisper-large-v3',
        provider: 'hf-inference',
        parameters: { return_timestamps: 'word' },
    });

    if (!result || !Array.isArray(result.chunks) || result.chunks.length === 0) {
        throw new Error('HF Whisper returned no chunks (word-level timestamps may be unsupported on this endpoint)');
    }

    const words = [];
    for (const chunk of result.chunks) {
        const ts = chunk.timestamp || chunk.timestamps;
        const text = (chunk.text || chunk.word || '').trim();
        if (!text || !Array.isArray(ts) || ts.length < 2) continue;
        if (!Number.isFinite(ts[0]) || !Number.isFinite(ts[1])) continue;

        const tokens = text.split(/\s+/).filter(Boolean);
        if (tokens.length === 1) {
            words.push({ word: tokens[0], startTime: ts[0] + offsetSec, endTime: ts[1] + offsetSec });
        } else if (tokens.length > 1) {
            // Multi-word chunk → distribute by character-length weight (more
            // accurate than uniform; long words occupy proportionally more time).
            const dur = Math.max(0, ts[1] - ts[0]);
            const weights = tokens.map(t => Math.max(1, t.replace(/[^a-zA-Z0-9']/g, '').length));
            const totalWeight = weights.reduce((a, b) => a + b, 0) || tokens.length;
            let cursor = ts[0];
            for (let i = 0; i < tokens.length; i++) {
                const slice = dur * (weights[i] / totalWeight);
                const start = cursor;
                const end = (i === tokens.length - 1) ? ts[1] : cursor + slice;
                words.push({ word: tokens[i], startTime: start + offsetSec, endTime: end + offsetSec });
                cursor = end;
            }
        }
    }
    return words;
}

/**
 * Transcribe with HuggingFace Whisper-large-v3 and request word-level
 * timestamps. For files larger than HF's ~25MB upload cap, the audio is
 * split into ~11-minute chunks at silence boundaries, each chunk is sent
 * to HF, and the word lists are concatenated with absolute timestamps.
 *
 * @param {string} audioPath - local WAV path
 * @returns {Array<{word, startTime, endTime}>}
 */
async function getWordTimestampsWhisperHF(audioPath) {
    if (!hfClient) {
        throw new Error('HUGGINGFACE_API_KEY not configured');
    }

    const stats = await fsPromises.stat(audioPath);
    const sizeMB = stats.size / 1024 / 1024;

    // Single-chunk fast path
    if (sizeMB <= WHISPER_MAX_FILE_MB) {
        return await _whisperTranscribeOnce(audioPath, 0);
    }

    // Multi-chunk path: split, transcribe each, merge.
    log('info', `Audio ${sizeMB.toFixed(1)}MB exceeds HF limit — chunking for Whisper`);

    let totalDuration;
    try {
        totalDuration = await getAudioDuration(audioPath);
    } catch (e) {
        throw new Error(`Cannot chunk audio: duration probe failed (${e.message})`);
    }

    const silences = await detectSilences(audioPath);
    log('debug', `Silence detection: ${silences.length} regions found`);

    const splitPoints = computeWhisperSplitPoints(totalDuration, silences);
    log('info', `Whisper chunking: ${splitPoints.length - 1} chunks, boundaries at [${splitPoints.map(s => s.toFixed(1)).join(', ')}]s`);

    const chunkPaths = [];
    try {
        const allWords = [];
        for (let i = 0; i < splitPoints.length - 1; i++) {
            const start = splitPoints[i];
            const dur   = splitPoints[i + 1] - start;
            const chunkPath = await extractWavChunk(audioPath, start, dur);
            chunkPaths.push(chunkPath);

            const chunkSizeMB = (await fsPromises.stat(chunkPath)).size / 1024 / 1024;
            log('info', `Whisper chunk ${i + 1}/${splitPoints.length - 1}: offset=${start.toFixed(1)}s dur=${dur.toFixed(1)}s size=${chunkSizeMB.toFixed(1)}MB`);

            if (chunkSizeMB > 24) {
                throw new Error(`Chunk ${i + 1} is ${chunkSizeMB.toFixed(1)}MB — still too large for HF (silence-split fallback failed)`);
            }

            // One retry on transient HF errors before failing the whole Whisper path.
            let words = null;
            for (let attempt = 1; attempt <= 2; attempt++) {
                try {
                    words = await _whisperTranscribeOnce(chunkPath, start);
                    break;
                } catch (err) {
                    if (attempt === 2) throw err;
                    log('warn', `Whisper chunk ${i + 1} attempt ${attempt} failed, retrying`, { error: err.message });
                    await new Promise(r => setTimeout(r, 5000));
                }
            }
            allWords.push(...words);
        }
        return allWords;
    } finally {
        for (const p of chunkPaths) {
            try { await fsPromises.unlink(p); } catch {}
        }
    }
}

// ─── Vertex Gemini chunking ────────────────────────────────────────────
// Gemini-2.5-Pro on Vertex silently truncates audio >~15 min — confirmed
// empirically on a 26-min file where Vertex transcribed only the first 18 min
// (43% missed). Splitting the audio into ~8-min chunks and calling Vertex
// per-chunk forces the model to process every region. Each chunk's local
// timestamps are offset by the chunk's start time before merging.
const VERTEX_CHUNK_TARGET_SEC    = parseInt(process.env.VERTEX_CHUNK_TARGET_SEC)    || 480;  // 8 min
const VERTEX_CHUNK_THRESHOLD_SEC = parseInt(process.env.VERTEX_CHUNK_THRESHOLD_SEC) || 600;  // chunk if audio > 10 min
const VERTEX_SAFETY_SETTINGS = [
    { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'OFF' },
    { category: 'HARM_CATEGORY_HATE_SPEECH',       threshold: 'OFF' },
    { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'OFF' },
    { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'OFF' },
    { category: 'HARM_CATEGORY_CIVIC_INTEGRITY',   threshold: 'OFF' },
];

async function _vertexCallOnChunkUri(chunkGcsUri, chunkDurSec, wordLimit, vocabulary, language) {
    return aiClient.models.generateContent({
        model: GEMINI_MODEL,
        config: {
            temperature: 0,
            topP: 0.95,
            maxOutputTokens: 65536,
            responseMimeType: 'application/json',
            safetySettings: VERTEX_SAFETY_SETTINGS,
        },
        contents: [{
            role: 'user',
            parts: [
                { text: buildEnhancedPrompt(wordLimit, vocabulary, chunkDurSec, language) },
                { fileData: { mimeType: 'audio/wav', fileUri: chunkGcsUri } },
            ],
        }],
    });
}

/**
 * Transcribe long audio by splitting into ~8-min chunks at silence boundaries,
 * sending each chunk to Vertex Gemini separately, and merging the per-chunk
 * JSON arrays with timestamps offset to global time.
 *
 * @returns {string} JSON string of merged [{start,end,text}, ...]
 */
async function transcribeVertexInChunks(audioPath, totalDuration, wordLimit, vocabulary, language, jobId) {
    log('info', `Vertex chunking: ${totalDuration.toFixed(1)}s audio → target ${VERTEX_CHUNK_TARGET_SEC}s/chunk`, { jobId: jobId.substring(0, 8) });

    const silences = await detectSilences(audioPath);
    const splitPoints = computeWhisperSplitPoints(totalDuration, silences, VERTEX_CHUNK_TARGET_SEC);
    const chunkCount = splitPoints.length - 1;
    log('info', `Vertex chunking: ${chunkCount} chunks at [${splitPoints.map(s => s.toFixed(1)).join(', ')}]s`, { jobId: jobId.substring(0, 8) });

    const chunkLocalPaths = [];
    const chunkGcsNames  = [];
    const allSegments    = [];

    try {
        for (let i = 0; i < chunkCount; i++) {
            const start    = splitPoints[i];
            const end      = splitPoints[i + 1];
            const chunkDur = end - start;

            updateProgress(jobId, 'transcribing', 75 + Math.floor((i / chunkCount) * 15),
                `Transcribing chunk ${i + 1}/${chunkCount} (offset ${start.toFixed(0)}s)...`);
            log('info', `Vertex chunk ${i + 1}/${chunkCount}: offset=${start.toFixed(1)}s dur=${chunkDur.toFixed(1)}s`,
                { jobId: jobId.substring(0, 8) });

            // 1. Extract local WAV chunk
            const localPath = await extractWavChunk(audioPath, start, chunkDur);
            chunkLocalPaths.push(localPath);

            // 2. Upload chunk to GCS
            const gcsName = `srt-ai/${jobId}-chunk${i + 1}-${Date.now()}.wav`;
            await gcsBucket.upload(localPath, {
                destination: gcsName,
                metadata: { contentType: 'audio/wav' },
            });
            chunkGcsNames.push(gcsName);
            const chunkGcsUri = `gs://${GCS_BUCKET_NAME}/${gcsName}`;

            // 3. Call Vertex Gemini with retry
            let response = null;
            for (let attempt = 1; attempt <= 2; attempt++) {
                try {
                    response = await _vertexCallOnChunkUri(chunkGcsUri, chunkDur, wordLimit, vocabulary, language);
                    break;
                } catch (err) {
                    if (attempt === 2) {
                        log('error', `Vertex chunk ${i + 1} failed after 2 attempts — skipping`, { error: err.message });
                        response = null;
                        break;
                    }
                    log('warn', `Vertex chunk ${i + 1} attempt ${attempt} failed, retrying in 15s`, { error: err.message });
                    await new Promise(r => setTimeout(r, 15000));
                }
            }
            if (!response) continue;

            // 4. Check for safety block on this chunk
            if (response.promptFeedback?.blockReason) {
                log('warn', `Vertex chunk ${i + 1} blocked by safety filter — skipping`, {
                    blockReason: response.promptFeedback.blockReason,
                });
                continue;
            }
            const candidate = response.candidates?.[0];
            if (candidate?.finishReason === 'SAFETY') {
                log('warn', `Vertex chunk ${i + 1} stopped on SAFETY — skipping`);
                continue;
            }

            // 5. Extract + parse JSON
            const rawText = response?.text || candidate?.content?.parts?.[0]?.text || '';
            const cleaned = cleanOutput(rawText);
            if (!cleaned || cleaned.trim() === '[]') {
                log('warn', `Vertex chunk ${i + 1} returned empty — skipping`);
                continue;
            }
            let chunkSegments;
            try {
                chunkSegments = JSON.parse(cleaned);
                if (!Array.isArray(chunkSegments)) throw new Error('response is not a JSON array');
            } catch (parseErr) {
                log('warn', `Vertex chunk ${i + 1}: failed to parse JSON`, {
                    error: parseErr.message,
                    preview: cleaned.substring(0, 200),
                });
                continue;
            }

            // 6. Offset every timestamp by chunk start, format back to SRT timestamp string
            for (const seg of chunkSegments) {
                if (!seg || typeof seg !== 'object') continue;
                if (seg.start != null) seg.start = formatTimestamp(parseTimestamp(seg.start) + start);
                if (seg.end   != null) seg.end   = formatTimestamp(parseTimestamp(seg.end)   + start);
                allSegments.push(seg);
            }
            log('info', `Vertex chunk ${i + 1}/${chunkCount} → ${chunkSegments.length} cues`, { jobId: jobId.substring(0, 8) });
        }

        if (allSegments.length === 0) {
            throw new Error('All Vertex chunks failed or returned no cues');
        }
        log('info', `Vertex chunking complete: ${allSegments.length} total cues from ${chunkCount} chunks`, { jobId: jobId.substring(0, 8) });
        return JSON.stringify(allSegments);
    } finally {
        // Local temp WAVs
        for (const p of chunkLocalPaths) {
            try { await fsPromises.unlink(p); } catch {}
        }
        // GCS chunk objects (best effort — auto-expire as fallback)
        for (const name of chunkGcsNames) {
            try { await gcsBucket.file(name).delete(); } catch {}
        }
    }
}

/**
 * Call Google Cloud Speech-to-Text V2 with the 'chirp_2' model.
 * chirp_2 is Google's latest universal speech model (released late 2024) and
 * catches noticeably more words than V1 'latest_long' on challenging audio
 * (fast speech, soft consonants, quiet endings, mixed acoustic conditions).
 *
 * Uses batchRecognize with inline response — works for audio up to several hours.
 * V2 uses startOffset/endOffset instead of V1's startTime/endTime.
 *
 * @returns {Array<{word: string, startTime: number, endTime: number}>}
 */
async function getWordTimestampsChirp2(gcsUri, language = 'auto') {
    const langCode = resolveLanguageCode(language);
    const recognizer = `projects/${GCP_PROJECT_ID}/locations/${VERTEX_REGION}/recognizers/_`;

    const request = {
        recognizer,
        config: {
            autoDecodingConfig: {},
            languageCodes: [langCode],
            model: 'chirp_2',
            features: {
                enableWordTimeOffsets: true,
                enableAutomaticPunctuation: true,
            },
        },
        files: [{ uri: gcsUri }],
        recognitionOutputConfig: {
            inlineResponseConfig: {},
        },
    };

    const [operation] = await sttV2Client.batchRecognize(request);
    const [response] = await operation.promise();

    const words = [];
    const fileResult = response.results?.[gcsUri];
    const transcript = fileResult?.transcript;
    if (!transcript?.results) return words;

    for (const result of transcript.results) {
        const alt = result.alternatives && result.alternatives[0];
        if (!alt || !alt.words) continue;
        for (const w of alt.words) {
            words.push({
                word: w.word,
                startTime: parseDuration(w.startOffset),
                endTime: parseDuration(w.endOffset),
            });
        }
    }
    return words;
}

/**
 * Build subtitle segments directly from STT word timestamps.
 * Uses STT for BOTH text and timing — avoids Gemini word hallucinations entirely.
 * Segments are split at natural speech pauses and the word limit.
 *
 * @param {Array<{word, startTime, endTime}>} sttWords
 * @param {number} wordLimit - max words per subtitle line
 * @returns {string} JSON string in the same format as AI output
 */
function buildSegmentsFromSTT(sttWords, wordLimit = 8) {
    if (!sttWords || sttWords.length === 0) return '[]';

    const PAUSE_THRESHOLD = 0.30;           // gap larger than this starts a new segment
    const CONJUNCTION_GAP = 0.15;           // smaller gap is enough if next word is a clause-starter
    const SENTENCE_END = /[.!?]$/;
    const CLAUSE_END = /[,;:]$/;
    const MIN_WORDS_FOR_SOFT_BREAK = 4;

    // Words that almost always begin a new clause/sentence in English.
    // When STT shows even a short gap before one of these, that's a real break.
    const CLAUSE_STARTERS = new Set([
        'but', 'and', 'or', 'so', 'because', 'however', 'although',
        'while', 'then', 'though', 'yet', 'still', 'also', 'plus',
        'meanwhile', 'instead', 'otherwise', 'therefore', 'thus',
    ]);

    // Drop a likely STT artifact at the very start: a tiny word ending with
    // a period followed by a long silence before real speech begins (e.g.
    // "Her." appearing before the actual narration starts).
    let words = sttWords;
    if (words.length >= 2) {
        const first = words[0];
        const second = words[1];
        const cleanedFirst = first.word.replace(/[^a-zA-Z']/g, '');
        const looksLikeArtifact =
            cleanedFirst.length <= 3 &&
            SENTENCE_END.test(first.word) &&
            (second.startTime - first.endTime) >= 0.5;
        if (looksLikeArtifact) {
            words = words.slice(1);
        }
    }

    const segments = [];
    let group = [];

    const flush = (endTime) => {
        if (group.length === 0) return;
        segments.push({
            start: formatTimestamp(group[0].startTime),
            end: formatTimestamp(endTime ?? group[group.length - 1].endTime),
            text: group.map(w => w.word).join(' '),
        });
        group = [];
    };

    for (let i = 0; i < words.length; i++) {
        const w = words[i];
        const next = words[i + 1];

        // PRE-PUSH: if this word is a clause-starter ("but", "and", ...) and
        // there's even a small pause before it, split BEFORE adding it. This
        // catches real clause breaks that STT marks with a sub-threshold gap.
        if (group.length >= MIN_WORDS_FOR_SOFT_BREAK) {
            const cleanWord = w.word.toLowerCase().replace(/[^a-z]/g, '');
            const prevEnd = group[group.length - 1].endTime;
            const gapBefore = w.startTime - prevEnd;
            if (CLAUSE_STARTERS.has(cleanWord) && gapBefore >= CONJUNCTION_GAP) {
                flush();
            }
        }

        group.push(w);

        const naturalPause = next && (next.startTime - w.endTime) >= PAUSE_THRESHOLD;
        const hitLimit = group.length >= wordLimit;
        const isLast = !next;
        const sentenceEnd = SENTENCE_END.test(w.word);
        const clauseEnd = CLAUSE_END.test(w.word) && group.length >= MIN_WORDS_FOR_SOFT_BREAK;

        // Always flush after sentence-ending punctuation — never carry a
        // sentence across a subtitle break.
        if (sentenceEnd && !isLast) {
            flush();
            continue;
        }

        if (isLast || naturalPause || hitLimit || clauseEnd) {
            // If we hit the hard word limit without any natural break point,
            // back up to the last punctuation in the final 3 words to avoid
            // chopping mid-phrase.
            if (hitLimit && !naturalPause && !isLast && !clauseEnd) {
                let breakAt = -1;
                for (let k = group.length - 1; k >= Math.max(0, group.length - 3); k--) {
                    if (/[,;.!?]$/.test(group[k].word)) {
                        breakAt = k;
                        break;
                    }
                }
                if (breakAt !== -1 && breakAt < group.length - 1) {
                    const keep = group.splice(breakAt + 1);
                    flush(group[group.length - 1].endTime);
                    group = keep;
                    continue;
                }
            }
            flush();
        }
    }

    return JSON.stringify(segments);
}

/**
 * Replace approximate AI timestamps with accurate Speech-to-Text word timestamps.
 *
 * Key improvements over naive approach:
 * - Uses AI timestamp as a position hint to bias the search window toward the correct audio region
 * - Verifies first-word matches with a 2-word anchor to avoid false positives on common words
 * - Falls back to 2nd-word anchor when the first word is missing from STT output
 * - Advances ptr using AI timestamp when no match is found, preventing stale-pointer drift
 *   (the main cause of later subtitles getting timestamps from early in the audio)
 */
function alignJsonTimestamps(jsonString, sttWords) {
    try {
        let cleaned = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();
        const s = cleaned.indexOf('[');
        const e = cleaned.lastIndexOf(']');
        if (s === -1 || e === -1) return jsonString;
        cleaned = cleaned.substring(s, e + 1).replace(/,\s*([}\]])/g, '$1');

        const segments = JSON.parse(cleaned);
        if (!Array.isArray(segments) || segments.length === 0) return jsonString;

        const norm = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, '');

        let ptr = 0; // sequential pointer into sttWords

        for (const seg of segments) {
            const segWords = seg.text.replace(/[^a-zA-Z0-9'\s]/g, '').split(/\s+/).filter(w => w);
            if (segWords.length === 0 || ptr >= sttWords.length) continue;

            // Use AI timestamp as a hint: bias the search window to start near where
            // the AI says this segment begins, not just at ptr.
            const aiStartSec = parseTimestamp(seg.start);
            let biasPtr = ptr;
            for (let i = ptr; i < sttWords.length; i++) {
                if (sttWords[i].startTime >= aiStartSec - 3) {
                    biasPtr = Math.max(ptr, i - 8); // look 8 words before expected position
                    break;
                }
            }

            const windowEnd = Math.min(biasPtr + 80, sttWords.length);

            // Strategy 1: Find first word with 2-word anchor verification (avoids false positives)
            let foundStart = -1;
            for (let i = biasPtr; i < windowEnd; i++) {
                if (norm(sttWords[i].word) === norm(segWords[0])) {
                    if (segWords.length > 1 && i + 1 < sttWords.length) {
                        if (norm(sttWords[i + 1].word) === norm(segWords[1])) {
                            foundStart = i;
                            break;
                        }
                        // 2-word anchor failed — keep searching; don't lock on common single word yet
                    } else {
                        foundStart = i; // single-word segment, no choice
                        break;
                    }
                }
            }

            // Strategy 2: First word without anchor verification (if anchored search failed)
            if (foundStart === -1) {
                for (let i = biasPtr; i < windowEnd; i++) {
                    if (norm(sttWords[i].word) === norm(segWords[0])) {
                        foundStart = i;
                        break;
                    }
                }
            }

            // Strategy 3: Use 2nd word as anchor when first word is missing from STT
            if (foundStart === -1 && segWords.length > 1) {
                for (let i = biasPtr; i < windowEnd; i++) {
                    if (norm(sttWords[i].word) === norm(segWords[1])) {
                        foundStart = Math.max(ptr, i - 1);
                        break;
                    }
                }
            }

            if (foundStart === -1) {
                // Advance ptr to AI timestamp position so future segments aren't affected
                // by a stale ptr pointing to an already-passed audio region.
                while (ptr < sttWords.length && sttWords[ptr].startTime < aiStartSec) ptr++;
                continue; // keep AI timestamp for this segment
            }

            // Walk forward through segment words to find the last matching word
            let foundEnd = foundStart;
            let sttIdx = foundStart;
            for (let si = 0; si < segWords.length; si++) {
                if (sttIdx >= sttWords.length) break;
                if (norm(sttWords[sttIdx].word) === norm(segWords[si])) {
                    foundEnd = sttIdx;
                    sttIdx++;
                } else {
                    // STT inserted an extra word — skip it and retry current AI word
                    if (sttIdx + 1 < sttWords.length && norm(sttWords[sttIdx + 1].word) === norm(segWords[si])) {
                        sttIdx++;
                        foundEnd = sttIdx;
                        sttIdx++;
                    }
                    // If no match either way, AI word was deleted from STT — just advance si (for loop)
                }
            }

            seg.start = formatTimestamp(sttWords[foundStart].startTime);
            seg.end = formatTimestamp(sttWords[foundEnd].endTime);
            ptr = foundEnd + 1;
        }

        return JSON.stringify(segments);
    } catch (err) {
        console.warn('[alignJsonTimestamps] failed, keeping AI timestamps:', err.message);
        return jsonString;
    }
}

// ═══════════════════════════════════════════════════════════
//   HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

function buildEnhancedPrompt(wordLimit, vocabulary = [], audioDurationSec = null, language = 'auto') {
    const vocabString = vocabulary.length > 0
        ? `\n\nVOCABULARY LIST (Prioritize these spellings): ${vocabulary.join(', ')}`
        : '';

    // NOTE: Do NOT phrase this as "timestamps MUST NOT exceed X". On Vertex,
    // Gemini-2.5-Pro reads that as a hard constraint and silently compresses or
    // skips early speech to fit. The reference is informational only.
    const durationHint = audioDurationSec
        ? `\n\nAUDIO DURATION REFERENCE: This audio is approximately ${Math.ceil(audioDurationSec)} seconds long. Transcribe ALL speech from start to finish — do not stop early, and place each timestamp at the actual moment the speech occurs in the audio.`
        : '';

    let languageInstruction = '';
    if (language && language !== 'auto') {
        languageInstruction = `\n\nLANGUAGE INSTRUCTION: Transcribe the audio and output ALL subtitle text in **${language}**. If the spoken language in the audio is different from ${language}, translate the speech into ${language} while maintaining accurate timestamps that match when the speech occurs in the audio. The timestamps must still align with the original spoken audio timing, but the text must be in ${language}.`;
    }

    return `You are a forensic transcription engine. Your task is to transcribe ONLY HUMAN SPEECH from audio to a JSON array.
${vocabString}${durationHint}${languageInstruction}

CRITICAL INSTRUCTIONS:
1. ONLY transcribe human speech (spoken words, dialogue, narration)
2. IGNORE all music, instrumental sections, sound effects, and background noise
3. SKIP any portions of audio that contain no human speech
4. If the audio contains music with speech, transcribe ONLY the speech parts
5. If there is NO human speech at all in the audio, return an empty array: []

You MUST return ONLY a valid JSON array. Nothing else. Start with [ and end with ].

EXAMPLE FORMAT (copy this structure exactly):

[{"start":"00:00:00,000","end":"00:00:03,500","text":"Hello, this is the first sentence."},{"start":"00:00:03,500","end":"00:00:07,200","text":"Thank you for that introduction."}]

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

4. **FORBIDDEN ENDINGS**: NEVER end a segment with a "weak" word unless it is the absolute end of the sentence.
   - **Weak Words**: "the", "a", "an", "and", "but", "or", "of", "in", "at", "to", "is", "was", "are", "were", "she", "he", "it", "they", "we", "you", "your", "his", "her"

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
6. CRITICAL: Respect the 8-10 word and 50 character limits strictly - this ensures single-line display.

DO NOT:
- Add markdown code blocks
- Add explanations before or after
- Add any text outside the JSON array
- Include speaker labels like "Speaker 1" or "[Speaker]"

OUTPUT NOW (JSON array only):`;
}

/**
 * Transcribe audio using Hugging Face Whisper API (Fallback for blocked content)
 */
async function transcribeWithWhisper(audioPath, wordLimit, jobId) {
    if (!hfClient) {
        throw new Error('Hugging Face API key not configured. Cannot use Whisper fallback.');
    }

    try {
        log('info', 'Using Whisper fallback for transcription', { jobId: jobId.substring(0, 8) });
        updateProgress(jobId, 'transcribing', 75, 'Transcribing with Whisper (fallback)...');

        // Read the audio file
        const audioBuffer = await fsPromises.readFile(audioPath);
        // MIME type required: HF routes whisper-large-v3 through fal-ai which
        // rejects Blobs without an explicit content-type.
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });

        // Get audio duration using ffprobe
        const audioDuration = await getAudioDuration(audioPath);
        log('debug', 'Audio duration detected', { duration: audioDuration, jobId: jobId.substring(0, 8) });

        // Call Hugging Face Whisper API
        // Note: HF's free API may truncate long audio, but we'll work with what we get
        // Force hf-inference (HF first-party) to avoid third-party provider
        // permission issues — see _whisperTranscribeOnce for context.
        const result = await hfClient.automaticSpeechRecognition({
            data: audioBlob,
            model: 'openai/whisper-large-v3',
            provider: 'hf-inference',
        });

        if (!result || !result.text) {
            throw new Error('Whisper returned empty transcription');
        }

        log('info', 'Whisper transcription successful', {
            textLength: result.text.length,
            wordCount: result.text.split(/\s+/).length,
            jobId: jobId.substring(0, 8)
        });

        // Convert Whisper text to SRT format with actual audio duration
        return convertWhisperTextToSRT(result.text, wordLimit, audioDuration);

    } catch (error) {
        log('error', 'Whisper transcription failed', { error: error.message });
        throw new Error(`Whisper fallback failed: ${error.message}`);
    }
}

/**
 * Get audio duration in seconds using ffprobe
 */
async function getAudioDuration(audioPath) {
    return new Promise((resolve, reject) => {
        ffmpeg.ffprobe(audioPath, (err, metadata) => {
            if (err) {
                log('error', 'ffprobe failed — cannot determine audio duration', {
                    error: err.message, path: audioPath,
                });
                // Fail loud. Silently defaulting to 60s previously compressed long
                // files into 60s of cues when combined with the prompt duration hint.
                reject(new Error(`Audio duration detection failed: ${err.message}`));
                return;
            }
            const duration = metadata?.format?.duration;
            if (!duration || !Number.isFinite(duration) || duration <= 0) {
                log('error', 'ffprobe returned invalid duration', { duration, path: audioPath });
                reject(new Error('Audio file has invalid duration (zero or unreadable). File may be corrupted.'));
                return;
            }
            resolve(duration);
        });
    });
}

/**
 * Convert Whisper plain text to SRT format with estimated timestamps
 */
function convertWhisperTextToSRT(text, wordLimit = 6, audioDuration = null) {
    const words = text.trim().split(/\s+/);
    const segments = [];

    // Calculate timing based on actual audio duration if available
    // Otherwise use average speaking rate (2.5 words per second)
    const totalWords = words.length;
    const estimatedDuration = audioDuration || (totalWords / 2.5);
    const secondsPerWord = estimatedDuration / totalWords;

    let currentTime = 0;

    for (let i = 0; i < words.length; i += wordLimit) {
        const segmentWords = words.slice(i, i + wordLimit);
        const segmentText = segmentWords.join(' ');

        // Calculate duration for this segment
        const duration = segmentWords.length * secondsPerWord;
        const startTime = currentTime;
        const endTime = Math.min(currentTime + duration, estimatedDuration);

        segments.push({
            index: segments.length + 1,
            start: formatSRTTime(startTime),
            end: formatSRTTime(endTime),
            text: segmentText
        });

        currentTime = endTime;

        // Stop if we've reached the end of the audio
        if (currentTime >= estimatedDuration) {
            break;
        }
    }

    // Convert to SRT format with proper line breaks
    return segments.map(seg =>
        `${seg.index}\n${seg.start} --> ${seg.end}\n${seg.text}\n`
    ).join('\n');
}

/**
 * Format seconds to SRT timestamp format (HH:MM:SS,mmm)
 */
function formatSRTTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);

    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function transcodeToWav(inputPath, onProgress = () => { }) {
    const outputPath = path.join(os.tmpdir(), `srt-ai-hq-${crypto.randomUUID()}.wav`);

    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error('Transcoding timed out - file may be too large'));
        }, 60 * 60 * 1000);

        ffmpeg(inputPath)
            .noVideo()
            .audioChannels(1)       // Mono — standard for speech recognition
            .audioFrequency(16000)  // 16kHz — optimal for ASR, smaller file = better AI timestamps
            .audioCodec('pcm_s16le')
            // Pre-ASR audio cleanup. Main goal: boost quiet speech so
            // STT's voice-activity detector doesn't skip over it.
            //   highpass=f=80          — strip sub-80Hz rumble (HVAC,
            //                            mic handling) that STT can
            //                            misclassify as noise.
            //   dynaudnorm=f=150:g=15:p=0.9
            //                          — frame-local loudness equalizer:
            //                            lifts quiet passages without
            //                            squashing loud ones. Single
            //                            biggest fix for "STT missed
            //                            words" on real-world footage
            //                            with inconsistent mic levels.
            .audioFilters([
                'highpass=f=80',
                'dynaudnorm=f=150:g=15:p=0.9',
            ])
            .format('wav')
            .on('progress', (progress) => {
                if (progress.percent) {
                    onProgress(progress.percent);
                }
            })
            .on('end', () => {
                clearTimeout(timeout);
                resolve(outputPath);
            })
            .on('error', (err) => {
                clearTimeout(timeout);
                reject(err);
            })
            .save(outputPath);
    });
}

async function deleteLocalFile(filePath) {
    try {
        await fsPromises.unlink(filePath);
        log('debug', 'Deleted temp file', { path: filePath });
    } catch (e) {
        log('warn', 'Failed to delete temp file', { path: filePath, error: e.message });
    }
}

/**
 * Validate uploaded file
 * @param {Object} file - Multer file object
 * @returns {string|null} Error message or null if valid
 */
async function validateFile(file) {
    // Check file size
    if (file.size === 0) {
        return 'File is empty. Please upload a valid file.';
    }

    if (file.size > MAX_FILE_SIZE) {
        const sizeMB = Math.round(file.size / 1024 / 1024);
        const maxMB = Math.round(MAX_FILE_SIZE / 1024 / 1024);
        return `File too large (${sizeMB}MB). Maximum size is ${maxMB}MB. Please compress or split your file.`;
    }

    // Check MIME type
    if (!SUPPORTED_FORMATS.includes(file.mimetype)) {
        return `Unsupported file format: ${file.mimetype}. Please use MP4, MP3, WAV, MOV, AVI, M4A, FLAC, or OGG.`;
    }

    // Check file extension matches MIME type
    const ext = path.extname(file.originalname).toLowerCase();
    const validExtensions = ['.mp4', '.mp3', '.wav', '.mov', '.avi', '.m4a', '.flac', '.ogg', '.mkv'];
    if (!validExtensions.includes(ext)) {
        return `Invalid file extension: ${ext}. Please use a valid audio/video file.`;
    }

    // Try to read file header to verify it's not corrupted
    try {
        const buffer = await fsPromises.readFile(file.path, { encoding: null, flag: 'r' });
        if (buffer.length < 100) {
            return 'File appears to be corrupted or incomplete.';
        }

        // Basic magic number validation for common formats
        const header = buffer.toString('hex', 0, 12);
        const validHeaders = [
            'fff', // MP3
            '494433', // MP3 with ID3
            '52494646', // WAV/AVI (RIFF)
            '000000', // MP4/M4A (ftyp)
            '66747970', // MP4 (ftyp)
            '1a45dfa3', // MKV
            '664c6143', // FLAC
            '4f676753'  // OGG
        ];

        const isValidHeader = validHeaders.some(h => header.startsWith(h));
        if (!isValidHeader && !header.startsWith('00000')) {
            log('warn', 'Suspicious file header', { header: header.substring(0, 20), filename: file.originalname });
            // Don't reject, as some valid files might have unusual headers
        }
    } catch (e) {
        log('error', 'File validation read error', { error: e.message });
        return 'Unable to read file. The file may be corrupted.';
    }

    return null; // File is valid
}

function cleanOutput(text) {
    if (!text) return '';

    let cleaned = text
        .replace(/```json/gi, '')
        .replace(/```/g, '')
        .trim();

    const jsonMatch = cleaned.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
        cleaned = jsonMatch[0];
    }

    const firstBracket = cleaned.indexOf('[');
    const lastBracket = cleaned.lastIndexOf(']');
    if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
        cleaned = cleaned.substring(firstBracket, lastBracket + 1);
    }

    return cleaned;
}

// ═══════════════════════════════════════════════════════════
//   TIMESTAMP HELPERS (Robust parsing for AI output)
// ═══════════════════════════════════════════════════════════

// Parse SRT Timestamp to Seconds.
// Accepts the canonical SRT form (HH:MM:SS,mmm or HH:MM:SS.mmm) and a few common
// AI-emitted variants. Rejects clearly malformed strings rather than silently
// returning 0 (which previously caused "all cues at 00:00:00" output).
function parseTimestamp(timestamp) {
    if (timestamp == null) return 0;
    if (typeof timestamp === 'number') return Number.isFinite(timestamp) ? Math.max(0, timestamp) : 0;
    if (typeof timestamp !== 'string') return 0;

    const raw = timestamp.trim();
    if (!raw) return 0;

    // Canonical SRT: HH:MM:SS,mmm  OR  HH:MM:SS.mmm
    let m = raw.match(/^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
    if (m) {
        const h = parseInt(m[1], 10);
        const mi = parseInt(m[2], 10);
        const s = parseInt(m[3], 10);
        const ms = parseInt(m[4].padEnd(3, '0').slice(0, 3), 10);
        return h * 3600 + mi * 60 + s + ms / 1000;
    }

    // HH:MM:SS (no fractional)
    m = raw.match(/^(\d{1,3}):(\d{2}):(\d{2})$/);
    if (m) return parseInt(m[1], 10) * 3600 + parseInt(m[2], 10) * 60 + parseInt(m[3], 10);

    // MM:SS,mmm or MM:SS.mmm (AI variant — assume MM:SS, NOT HH:MM)
    m = raw.match(/^(\d{1,3}):(\d{2})[,.](\d{1,3})$/);
    if (m) {
        const mi = parseInt(m[1], 10);
        const s = parseInt(m[2], 10);
        const ms = parseInt(m[3].padEnd(3, '0').slice(0, 3), 10);
        return mi * 60 + s + ms / 1000;
    }

    // MM:SS
    m = raw.match(/^(\d{1,3}):(\d{1,2})$/);
    if (m) return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);

    // Bare seconds (float)
    m = raw.match(/^(\d+(?:[,.]\d+)?)$/);
    if (m) {
        const v = parseFloat(m[1].replace(',', '.'));
        return Number.isFinite(v) ? v : 0;
    }

    log('warn', 'parseTimestamp: unrecognized timestamp format', { input: raw });
    return 0;
}

// Convert Seconds to SRT Timestamp (HH:MM:SS,mmm)
// Uses direct integer math with proper rounding (NOT Date.setMilliseconds which floors).
// Floor-rounding caused cumulative drift of up to 1ms per cue that Premiere Pro picks up
// as audio/text misalignment over long files. Confirmed via round-trip test.
function formatTimestamp(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const totalMs = Math.round(seconds * 1000);
    const h  = Math.floor(totalMs / 3600000);
    const m  = Math.floor((totalMs % 3600000) / 60000);
    const s  = Math.floor((totalMs % 60000) / 1000);
    const ms = totalMs % 1000;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

// ═══════════════════════════════════════════════════════════
//   POST-PROCESSING PIPELINE (Fix AI timestamp mistakes)
// ═══════════════════════════════════════════════════════════

// Fix overlapping timestamps — trims end times (preserves AI start times)
function compactOverlaps(segments, minDurationMs = 300) {
    if (!segments || segments.length === 0) return [];

    const sorted = [...segments].sort((a, b) => {
        const aStart = parseTimestamp(a.start);
        const bStart = parseTimestamp(b.start);
        return aStart - bStart;
    });

    const result = [];

    for (let i = 0; i < sorted.length; i++) {
        const seg = { ...sorted[i] };
        let startSec = parseTimestamp(seg.start);
        let endSec = parseTimestamp(seg.end);

        // Fix: Ensure end > start (minimum duration)
        if (endSec <= startSec) {
            const wordCount = seg.text ? seg.text.trim().split(/\s+/).length : 1;
            endSec = startSec + Math.max(minDurationMs / 1000, wordCount * 0.3);
        }

        // Fix: If this segment overlaps with next segment's start, trim this segment's end
        if (i < sorted.length - 1) {
            const nextStart = parseTimestamp(sorted[i + 1].start);
            if (endSec > nextStart) {
                endSec = nextStart;
                if (endSec <= startSec) {
                    endSec = startSec + (minDurationMs / 1000);
                }
            }
        }

        result.push({
            ...seg,
            start: formatTimestamp(startSec),
            end: formatTimestamp(endSec)
        });
    }

    return result;
}

// Merge consecutive segments with small gaps (prevents visual breaks)
function mergeCloseSegments(segments, maxGapMs = 300, maxWords = 12, maxChars = 50) {
    if (!segments || segments.length <= 1) return segments;

    const merged = [];
    let current = { ...segments[0] };

    for (let i = 1; i < segments.length; i++) {
        const next = segments[i];
        const currentEnd = parseTimestamp(current.end);
        const nextStart = parseTimestamp(next.start);
        const gapMs = (nextStart - currentEnd) * 1000;

        const combinedText = current.text.trim() + ' ' + next.text.trim();
        const combinedWords = combinedText.split(/\s+/).length;
        const combinedChars = combinedText.length;

        if (gapMs >= 0 && gapMs <= maxGapMs && combinedWords <= maxWords && combinedChars <= maxChars) {
            current = {
                ...current,
                text: combinedText,
                end: next.end
            };
        } else {
            merged.push(current);
            current = { ...next };
        }
    }
    merged.push(current);

    return merged;
}

// Bridge small gaps by extending end times (visual continuity in video editors)
function bridgeGaps(segments, maxBridgeGapMs = 500) {
    if (!segments || segments.length <= 1) return segments;

    const result = [];
    for (let i = 0; i < segments.length; i++) {
        const seg = { ...segments[i] };

        if (i < segments.length - 1) {
            const currentEnd = parseTimestamp(seg.end);
            const nextStart = parseTimestamp(segments[i + 1].start);
            const gapMs = (nextStart - currentEnd) * 1000;

            if (gapMs > 0 && gapMs <= maxBridgeGapMs) {
                seg.end = segments[i + 1].start;
            }
        }

        result.push(seg);
    }

    return result;
}

// Smart split a segment at natural break points (commas, conjunctions, sentence ends)
function smartSplitSegment(text, startTime, endTime, maxWords = 10, maxChars = 50) {
    const words = text.trim().split(/\s+/);

    if (endTime <= startTime) {
        endTime = startTime + Math.max(2.0, words.length * 0.3);
    }

    if (words.length <= maxWords && text.length <= maxChars) {
        return [{ text: text.trim(), start: startTime, end: endTime }];
    }

    const segments = [];
    const totalDuration = Math.max(0.5, endTime - startTime);
    const weakWords = new Set(['the', 'a', 'an', 'and', 'but', 'or', 'of', 'in', 'at', 'to', 'is', 'was', 'are', 'were', 'she', 'he', 'it', 'they', 'we', 'you', 'your', 'his', 'her', 'my', 'our', 'their']);

    let currentSegment = [];
    let currentChars = 0;
    let segmentStartTime = startTime;

    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const wordWithSpace = (currentSegment.length > 0 ? ' ' : '') + word;
        const remainingWords = words.length - (i + 1);

        const wouldExceedWords = currentSegment.length + 1 > maxWords;
        const wouldExceedChars = currentChars + wordWithSpace.length > maxChars;

        const isSentenceEnd = /[.?!]/.test(word);
        const hasComma = word.includes(',');
        const hasSemicolon = word.includes(';');
        const nextIsConjunction = i + 1 < words.length && ['and', 'but', 'or', 'so'].includes(words[i + 1].toLowerCase());
        const isInnerBreak = hasComma || hasSemicolon || nextIsConjunction;

        currentSegment.push(word);
        currentChars += wordWithSpace.length;

        let shouldBreak = false;

        if (isSentenceEnd && currentSegment.length >= 3 && remainingWords >= 3) {
            shouldBreak = true;
        } else if ((wouldExceedWords || wouldExceedChars) && isInnerBreak && remainingWords >= 3) {
            const lastWord = word.replace(/[.,!?;]+$/, '').toLowerCase();
            if (!weakWords.has(lastWord)) {
                shouldBreak = true;
            }
        } else if (currentSegment.length >= maxWords || currentChars >= maxChars) {
            if (remainingWords >= 3) {
                const lastWord = currentSegment[currentSegment.length - 1].replace(/[.,!?;]+$/, '').toLowerCase();
                if (!weakWords.has(lastWord) || currentChars > maxChars * 1.2) {
                    shouldBreak = true;
                }
            }
        }

        if (shouldBreak) {
            const segmentText = currentSegment.join(' ');
            // Allocate duration proportional to WORD COUNT (rough syllable proxy)
            // rather than character count. Character-weighted splitting biases
            // toward long words and produces visibly wrong cue boundaries on
            // dialogue mixing short and long words.
            const segmentDuration = (currentSegment.length / words.length) * totalDuration;
            const segmentEndTime = segmentStartTime + segmentDuration;

            segments.push({ text: segmentText, start: segmentStartTime, end: segmentEndTime });
            currentSegment = [];
            currentChars = 0;
            segmentStartTime = segmentEndTime;
        }
    }

    // Add remaining segment (merge with last if too short)
    if (currentSegment.length > 0) {
        if (currentSegment.length <= 2 && segments.length > 0) {
            const lastSegment = segments[segments.length - 1];
            const mergedText = lastSegment.text + ' ' + currentSegment.join(' ');

            if (mergedText.split(/\s+/).length <= maxWords * 1.5 && mergedText.length <= maxChars * 1.5) {
                segments[segments.length - 1] = { text: mergedText, start: lastSegment.start, end: endTime };
            } else {
                segments.push({ text: currentSegment.join(' '), start: segmentStartTime, end: endTime });
            }
        } else {
            segments.push({ text: currentSegment.join(' '), start: segmentStartTime, end: endTime });
        }
    }

    return segments;
}

// ═══════════════════════════════════════════════════════════
//   JSON TO SRT CONVERTER (with full post-processing)
// ═══════════════════════════════════════════════════════════

function jsonToSrt(jsonString, wordLimit, audioDurationSec = null) {
    if (!jsonString || !jsonString.trim()) {
        return { srt: `1\r\n00:00:00,000 --> 00:00:05,000\r\n[Error] Empty transcription response.\r\n`, timingReport: '' };
    }

    try {
        let cleanJson = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();

        const start = cleanJson.indexOf('[');
        const end = cleanJson.lastIndexOf(']');

        if (start === -1 || end === -1) {
            throw new Error('No JSON array found');
        }

        cleanJson = cleanJson.substring(start, end + 1);
        cleanJson = cleanJson.replace(/,\s*([}\]])/g, '$1');

        const segments = JSON.parse(cleanJson);

        if (!Array.isArray(segments) || segments.length === 0) {
            throw new Error('Invalid or empty array');
        }

        let srtOutput = "";
        let finalSegments = [];
        let segmentIndex = 1;

        const maxWords = 12;
        const maxChars = 50;

        // STEP 1: Parse, validate timestamps, and smart-split long segments
        segments.forEach((seg) => {
            let tStart = parseTimestamp(seg.start);
            let tEnd = parseTimestamp(seg.end);

            // VALIDATION: Ensure end > start
            if (tEnd <= tStart) {
                const words = seg.text.trim().split(/\s+/);
                tEnd = tStart + Math.max(0.5, words.length * 0.25);
            }

            // VALIDATION: Only clamp clearly-absurd cue durations. The previous
            // 30s ceiling was too aggressive — Gemini can legitimately emit a
            // 35-45s cue around long pauses, and snapping the end inward made
            // every subsequent cue drift relative to audio. We now only reject
            // genuinely impossible durations (>2 minutes per cue OR more than
            // 5s/word, whichever is higher).
            const duration = tEnd - tStart;
            const wordCount = seg.text.trim().split(/\s+/).filter(Boolean).length || 1;
            const sanityCap = Math.max(15.0, Math.min(120.0, wordCount * 5.0));
            if (duration > sanityCap) {
                log('warn', 'Cue duration exceeds sanity cap — clamping', {
                    duration: duration.toFixed(2), cap: sanityCap.toFixed(2),
                    wordCount, text: seg.text?.substring(0, 60),
                });
                tEnd = tStart + sanityCap;
            }

            const startStr = formatTimestamp(tStart);
            const endStr = formatTimestamp(tEnd);

            const words = seg.text.trim().split(/\s+/);
            const charCount = seg.text.trim().length;

            if (words.length <= maxWords && charCount <= maxChars) {
                finalSegments.push({
                    index: segmentIndex++,
                    start: startStr,
                    end: endStr,
                    text: seg.text.trim()
                });
            } else {
                const splitSegments = smartSplitSegment(seg.text.trim(), tStart, tEnd, maxWords, maxChars);
                splitSegments.forEach(splitSeg => {
                    finalSegments.push({
                        index: segmentIndex++,
                        start: formatTimestamp(splitSeg.start),
                        end: formatTimestamp(splitSeg.end),
                        text: splitSeg.text
                    });
                });
            }
        });

        // STEP 2: Fix overlapping timestamps FIRST so adjacent cues are non-overlapping
        // before merging. (Whisper word boundaries often overlap by 1-50ms; if we
        // merge first, mergeCloseSegments skips them because gap<0, and overlap
        // remains in the final SRT.)
        finalSegments = compactOverlaps(finalSegments, 300);

        // STEP 3: Merge very close segments (< 150ms gap) to prevent visual breaks
        finalSegments = mergeCloseSegments(finalSegments, 150, maxWords, maxChars);

        // STEP 4: Bridge tiny gaps (< 250ms) for visual continuity
        const bridgedSegments = bridgeGaps(finalSegments, 250);

        // Re-index after all processing
        bridgedSegments.forEach((seg, idx) => {
            seg.index = idx + 1;
        });

        // Build SRT String with CRLF line endings (Premiere Pro compatible)
        bridgedSegments.forEach(seg => {
            let cleanText = seg.text;
            const wc = seg.text.trim().split(/\s+/).length;
            if (wc === 1) {
                cleanText = seg.text.replace(/[.,!?]+$/, '');
            }
            srtOutput += `${seg.index}\r\n${seg.start} --> ${seg.end}\r\n${cleanText}\r\n\r\n`;
        });

        // Build timing report: RAW AI output vs FINAL output side by side
        const durationLine = audioDurationSec
            ? `Audio Duration : ${Math.floor(audioDurationSec / 60)}m ${(audioDurationSec % 60).toFixed(1)}s (${audioDurationSec.toFixed(1)}s total)`
            : 'Audio Duration : unknown';

        const col = (s, w) => String(s).padEnd(w);
        const divider = '-'.repeat(80);

        let report = [
            '='.repeat(80),
            '  SRT TIMING REPORT  —  Use this to spot misalignment',
            '='.repeat(80),
            `Generated      : ${new Date().toISOString()}`,
            durationLine,
            '',
            '[ RAW AI OUTPUT — timestamps exactly as the AI returned them ]',
            divider,
            `${col('#', 5)} ${col('START', 16)} ${col('END', 16)} TEXT`,
            divider,
        ];

        segments.forEach((seg, i) => {
            report.push(`${col(i + 1, 5)} ${col(seg.start, 16)} ${col(seg.end, 16)} ${seg.text}`);
        });

        report.push('');
        report.push('[ FINAL OUTPUT — after post-processing (split / merge / overlap fix) ]');
        report.push(divider);
        report.push(`${col('#', 5)} ${col('START', 16)} ${col('END', 16)} TEXT`);
        report.push(divider);

        bridgedSegments.forEach((seg) => {
            report.push(`${col(seg.index, 5)} ${col(seg.start, 16)} ${col(seg.end, 16)} ${seg.text}`);
        });

        report.push('');
        report.push('='.repeat(80));
        report.push('HOW TO READ: Compare RAW vs FINAL timestamps against your audio.');
        report.push('If RAW timestamps are wrong → AI alignment issue (prompt/model).');
        report.push('If RAW is correct but FINAL is wrong → post-processing bug.');
        report.push('='.repeat(80));

        const timingReport = report.join('\n');

        return { srt: srtOutput.trimEnd() + '\r\n', timingReport };

    } catch (e) {
        console.error('[Error] JSON parsing failed:', e.message);

        // Try regex recovery
        try {
            const objects = [];
            const regex = /\{"start":"([^"]+)","end":"([^"]+)","text":"([^"]+)"\}/g;
            let match;
            while ((match = regex.exec(jsonString)) !== null) {
                objects.push({ start: match[1], end: match[2], text: match[3] });
            }

            if (objects.length > 0) {
                const srt = objects.map((entry, index) => {
                    const seq = index + 1;
                    return `${seq}\r\n${entry.start} --> ${entry.end}\r\n${entry.text}`;
                }).join('\r\n\r\n') + '\r\n';
                return { srt, timingReport: '' };
            }
        } catch (recoveryErr) { }

        return { srt: `1\r\n00:00:00,000 --> 00:00:05,000\r\n[Error] Failed to parse transcription.\r\n`, timingReport: '' };
    }
}

// Helper functions for time conversion
function timeToMs(timeStr) {
    const parts = timeStr.split(/[:,]/);
    const hours = parseInt(parts[0]) || 0;
    const minutes = parseInt(parts[1]) || 0;
    const seconds = parseInt(parts[2]) || 0;
    const ms = parseInt(parts[3]) || 0;
    return (hours * 3600000) + (minutes * 60000) + (seconds * 1000) + ms;
}

function msToTime(ms) {
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    const milliseconds = Math.floor(ms % 1000);

    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
}

// ═══════════════════════════════════════════════════════════
//   START SERVER
// ═══════════════════════════════════════════════════════════

// Graceful shutdown
process.on('SIGTERM', () => {
    log('info', 'SIGTERM received, shutting down gracefully');
    process.exit(0);
});

process.on('SIGINT', () => {
    log('info', 'SIGINT received, shutting down gracefully');
    process.exit(0);
});

process.on('uncaughtException', (error) => {
    log('error', 'Uncaught exception', { error: error.message, stack: error.stack });
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    log('error', 'Unhandled rejection', { reason, promise });
});

app.listen(PORT, () => {
    log('info', '══════════════════════════════════════════');
    log('info', `🎬 SRT-AI Server v2.0.0 (Production Ready)`);
    log('info', `📍 URL:   http://localhost:${PORT}`);
    log('info', `🤖 Model: ${GEMINI_MODEL}`);
    log('info', `⚙️  Environment: ${NODE_ENV}`);
    log('info', `⏱️  Timeout: ${REQUEST_TIMEOUT_MS / 1000}s`);
    log('info', `📊 Rate Limit: ${RATE_LIMIT_REQUESTS} req/hour`);
    log('info', `📝 Log Level: ${LOG_LEVEL}`);
    log('info', '══════════════════════════════════════════');
});


