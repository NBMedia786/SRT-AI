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
import { Agent, setGlobalDispatcher } from 'undici';

// --- SDK IMPORTS ---
import { GoogleGenAI } from '@google/genai';
import { HfInference } from '@huggingface/inference';
import { separateVocals, isSourceSeparationAvailable } from './source_separation.js';
import { Storage } from '@google-cloud/storage';
import speech from '@google-cloud/speech';

// Node 22's default undici fetch caps connection/body/header timeouts at 5 min,
// which kills long-running Vertex AI generateContent calls on bigger audio
// files (observed "fetch failed" exactly at 5:06 across multiple runs).
// Bumping to 30 min lets Vertex finish chunked long-audio inference.
setGlobalDispatcher(new Agent({
    headersTimeout: 30 * 60 * 1000,
    bodyTimeout:    30 * 60 * 1000,
    connectTimeout: 60 * 1000,
}));

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
// Gemini-on-Vertex location. Gemini 3.x (e.g. gemini-3.1-pro-preview) is served
// ONLY on the 'global' endpoint, so keep this separate from VERTEX_REGION (which
// also drives the GCS bucket region and the regional Speech-to-Text endpoints).
const VERTEX_GENAI_LOCATION = process.env.VERTEX_GENAI_LOCATION || 'global';

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
    location: VERTEX_GENAI_LOCATION,
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

// Completed/failed results kept short-term so a reloaded client can recover its job.
// Stored in-memory keyed by jobId. Auto-evicted after RESULT_TTL_MS.
const jobResults = new Map();
const RESULT_TTL_MS = 30 * 60 * 1000; // 30 minutes

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
    let separationWorkDir = null;

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
        // STEP 1.5: OPTIONAL source separation (Demucs)
        // Multi-speaker / narrator-over-bodycam / music-with-vocals audio
        // confuses single-stream transcription. Isolating the vocals track
        // upfront gives STT and Gemini a clean signal and dramatically tightens
        // timestamps on overlay content.
        //
        // Enable via ENABLE_SOURCE_SEPARATION=true in .env. Requires demucs:
        //   python3 -m pip install --user demucs
        // -------------------------------------------------------
        if (process.env.ENABLE_SOURCE_SEPARATION === 'true') {
            try {
                updateProgress(jobId, 'separating', 42, 'Isolating vocals from background…');
                log('info', 'Source separation enabled — running Demucs', { jobId: jobId.substring(0, 8) });
                const sep = await separateVocals(audioPath, (stage) => {
                    if (stage && stage.length < 120) {
                        updateProgress(jobId, 'separating', 43, `Demucs: ${stage}`);
                    }
                });
                // Replace audioPath with the isolated vocals — all downstream stages
                // (GCS upload, STT, Gemini) now see a clean narrator track.
                audioPath = sep.vocals;
                separationWorkDir = sep.workDir;
                cleanupPaths.push(sep.vocals, sep.noVocals);
                log('info', 'Source separation done — using isolated vocals track', {
                    jobId: jobId.substring(0, 8),
                    vocalsPath: sep.vocals,
                });
            } catch (sepErr) {
                // Don't fail the job — fall back to original audio if separation breaks
                log('warn', 'Source separation failed, continuing with original audio', {
                    jobId: jobId.substring(0, 8),
                    error: sepErr.message,
                });
            }
        }

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
        // TRANSCRIPTION DISPATCHER — three modes:
        //   • vertex  : Gemini-on-Vertex only (estimated timestamps)
        //   • auto    : STT cascade with Gemini as fallback (default)
        //   • hybrid  : STT + Gemini in parallel, merged via alignment.
        //               Gemini supplies text (better word coverage),
        //               STT supplies waveform-accurate word timings.
        // -------------------------------------------------------
        const transcriptionMode = (process.env.TRANSCRIPTION_MODE || 'auto').toLowerCase();
        let jsonResponse = null;
        let usedSTT = false;
        let transcriptionSource = 'unknown';

        if (transcriptionMode === 'vertex') {
            log('info', 'TRANSCRIPTION_MODE=vertex — Gemini-on-Vertex only', { jobId: jobId.substring(0, 8) });
            updateProgress(jobId, 'transcribing', 75, `Transcribing with ${GEMINI_MODEL}...`);
            jsonResponse = await runVertexGemini(audioPath, gcsUri, audioDuration, wordLimit, vocabulary, language, jobId);
            transcriptionSource = 'gemini';
        }
        else if (transcriptionMode === 'hybrid') {
            log('info', 'TRANSCRIPTION_MODE=hybrid — running STT and Gemini in parallel', { jobId: jobId.substring(0, 8) });
            updateProgress(jobId, 'transcribing', 50, 'Running STT and Gemini in parallel (cross-validation)...');

            const [sttResult, geminiResult] = await Promise.allSettled([
                runSTTCascade(audioPath, gcsUri, language, jobId),
                runVertexGemini(audioPath, gcsUri, audioDuration, wordLimit, vocabulary, language, jobId),
            ]);

            const sttData = sttResult.status === 'fulfilled' ? sttResult.value : null;
            const sttWords = sttData?.words;
            const geminiJson = geminiResult.status === 'fulfilled' ? geminiResult.value : null;

            if (sttResult.status === 'rejected') {
                log('warn', 'Hybrid: STT cascade failed', { jobId: jobId.substring(0, 8), error: sttResult.reason?.message });
            }
            if (geminiResult.status === 'rejected') {
                log('warn', 'Hybrid: Gemini failed', { jobId: jobId.substring(0, 8), error: geminiResult.reason?.message });
            }

            if (sttWords && geminiJson) {
                log('info', `Hybrid: merging Gemini text with ${sttWords.length} STT word timings (model: ${sttData.model})`, { jobId: jobId.substring(0, 8) });
                jsonResponse = alignJsonTimestamps(geminiJson, sttWords);
                // Gap-fill pass: insert STT-only cues into sections where Gemini missed
                // speech entirely. alignJsonTimestamps fixes timestamps but doesn't add
                // cues, so missed-by-Gemini speech stayed missing pre-fix.
                jsonResponse = fillSttGaps(jsonResponse, sttWords, 3.0, wordLimit);
                // Dead-zone retry: even after fillSttGaps there can be gaps where BOTH
                // Gemini AND the initial STT pass missed speech (STT word coverage is
                // context-dependent — running it on just the gap clip often recovers
                // words the full-audio pass dropped). Verified on bodycam audio:
                // recovered 21 cues across 3 gaps that were missing in main output.
                updateProgress(jobId, 'transcribing', 88, 'Checking for dead zones and recovering missed speech...');
                jsonResponse = await fillDeadZonesViaRetry(jsonResponse, audioPath, language, jobId, 5);
                usedSTT = true;
                transcriptionSource = `hybrid (${sttData.model} + ${GEMINI_MODEL})`;
            } else if (sttWords) {
                log('warn', `Hybrid: Gemini failed, using STT (${sttData.model}) alone`, { jobId: jobId.substring(0, 8) });
                jsonResponse = buildSegmentsFromSTT(sttWords, wordLimit);
                // Dead-zone retry: even on STT-only fallback, gaps >5s often contain
                // speech the initial pass dropped (context-dependent STT behavior).
                // Without this, when Gemini fails the output has multi-second holes.
                updateProgress(jobId, 'transcribing', 88, 'Checking for dead zones and recovering missed speech...');
                jsonResponse = await fillDeadZonesViaRetry(jsonResponse, audioPath, language, jobId, 5);
                usedSTT = true;
                transcriptionSource = sttData.model;
            } else if (geminiJson) {
                log('warn', 'Hybrid: STT failed, using Gemini alone (timestamps will be estimated)', { jobId: jobId.substring(0, 8) });
                jsonResponse = geminiJson;
                transcriptionSource = 'gemini (STT unavailable)';
            } else {
                throw new Error('Both STT and Gemini failed in hybrid mode. Check your API credentials and quota.');
            }
        }
        else {
            // auto mode: STT cascade first, Gemini as fallback
            log('info', 'TRANSCRIPTION_MODE=auto — STT cascade, Gemini fallback', { jobId: jobId.substring(0, 8) });

            let sttData = null;
            try {
                sttData = await runSTTCascade(audioPath, gcsUri, language, jobId);
            } catch (e) {
                log('warn', 'STT cascade threw', { jobId: jobId.substring(0, 8), error: e.message });
            }

            if (sttData?.words?.length > 0) {
                log('info', `STT transcription successful via ${sttData.model}`, {
                    jobId: jobId.substring(0, 8),
                    wordCount: sttData.words.length,
                    model: sttData.model,
                });
                jsonResponse = buildSegmentsFromSTT(sttData.words, wordLimit);
                // Dead-zone retry: find gaps >5s, re-run chirp_2 on those clips
                // alone, splice recovered cues back in. Catches speech that the
                // main pass dropped due to context-related model failures.
                // Lowered from 10s → 5s after gap-probe testing showed 5-9s gaps
                // routinely contain 1-3 missed cues on bodycam audio.
                updateProgress(jobId, 'transcribing', 85, 'Checking for dead zones and recovering missed speech...');
                jsonResponse = await fillDeadZonesViaRetry(jsonResponse, audioPath, language, jobId, 5);
                usedSTT = true;
                transcriptionSource = sttData.model;
            } else {
                log('info', 'STT cascade returned no words — falling back to Gemini', { jobId: jobId.substring(0, 8) });
                updateProgress(jobId, 'transcribing', 75, `Transcribing with ${GEMINI_MODEL}...`);
                jsonResponse = await runVertexGemini(audioPath, gcsUri, audioDuration, wordLimit, vocabulary, language, jobId);
                transcriptionSource = 'gemini (fallback)';
            }
        }

        updateProgress(jobId, 'formatting', 90, 'Formatting SRT output...');

        if (!jsonResponse || jsonResponse.trim() === '[]') {
            log('warn', 'No speech detected in audio', { jobId });
            throw new Error('No human speech detected in the audio file. The file may contain only music, instrumental audio, or be too quiet. Please ensure your audio contains clear spoken words.');
        }

        // -------------------------------------------------------
        // STEP 4: Convert transcription JSON → SRT
        // jsonResponse is from STT (primary) or Gemini fallback.
        // -------------------------------------------------------
        log('info', `Transcription source: ${transcriptionSource}`, { jobId: jobId.substring(0, 8) });

        const { srt: srtOutput, timingReport } = jsonToSrt(jsonResponse, wordLimit, audioDuration);

        if (!srtOutput || srtOutput.includes('[Error]')) {
            log('error', 'SRT conversion failed', { jsonResponse: jsonResponse.substring(0, 200) });
            throw new Error('Failed to generate subtitles. The audio may be unclear or too short.');
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        // Persist the EXACT audio the AI saw (the transcoded WAV) so playback in the UI
        // is bit-identical with what produced the timestamps — guarantees player sync.
        let storedAudioName = null;
        try {
            storedAudioName = `${jobId}.wav`;
            const storedAudioPath = path.join(STORED_AUDIO_DIR, storedAudioName);
            await fsPromises.copyFile(audioPath, storedAudioPath);
            log('debug', 'Stored audio for replay', { file: storedAudioName });
        } catch (e) {
            log('warn', 'Failed to store audio for replay', { error: e.message });
            storedAudioName = null;
        }

        updateProgress(jobId, 'complete', 100, `Completed in ${duration}s`);

        log('info', `Job completed successfully`, {
            jobId: jobId.substring(0, 8),
            duration: duration + 's',
            subtitleCount: srtOutput.split('\n\n').length
        });

        const result = { srt: srtOutput, timingReport, jobId, duration, language, audioFile: storedAudioName };
        // Stash so a reloaded client can recover via GET /api/result/:jobId
        jobResults.set(jobId, { status: 'ok', completedAt: Date.now(), result });
        setTimeout(() => jobResults.delete(jobId), RESULT_TTL_MS);

        res.json(result);

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

        const errPayload = {
            error: errorType,
            message: userMessage,
            details: isProduction ? undefined : error.message,
            jobId,
            timestamp: new Date().toISOString()
        };
        // Cache error so a reloaded client can see what went wrong
        jobResults.set(jobId, { status: 'error', completedAt: Date.now(), error: errPayload });
        setTimeout(() => jobResults.delete(jobId), RESULT_TTL_MS);

        if (!res.headersSent) {
            res.status(statusCode).json(errPayload);
        }

    } finally {
        // Cleanup
        for (const p of cleanupPaths) {
            await deleteLocalFile(p);
        }
        // Cleanup Demucs work dir (vocals.wav + no_vocals.wav are inside it)
        if (separationWorkDir) {
            try { await fsPromises.rm(separationWorkDir, { recursive: true, force: true }); }
            catch (e) { log('warn', 'Failed to remove separation workdir', { dir: separationWorkDir, error: e.message }); }
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

// ═══════════════════════════════════════════════════════════
//   HISTORY + STORED AUDIO (persistent server-side state)
// ═══════════════════════════════════════════════════════════

const STORED_AUDIO_DIR = path.join(__dirname, 'stored_audio');
const HISTORY_FILE = path.join(__dirname, 'history.json');
const HISTORY_MAX_ENTRIES = 50;

if (!fs.existsSync(STORED_AUDIO_DIR)) {
    fs.mkdirSync(STORED_AUDIO_DIR, { recursive: true });
}

function readHistory() {
    try {
        const raw = fs.readFileSync(HISTORY_FILE, 'utf8');
        const arr = JSON.parse(raw);
        return Array.isArray(arr) ? arr : [];
    } catch {
        return [];
    }
}

function writeHistory(arr) {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(arr, null, 2));
}

function deleteStoredAudio(audioFile) {
    if (!audioFile) return;
    const safe = path.basename(audioFile);
    const p = path.join(STORED_AUDIO_DIR, safe);
    if (fs.existsSync(p)) {
        try { fs.unlinkSync(p); } catch (e) {
            log('warn', 'Failed to delete stored audio', { file: safe, error: e.message });
        }
    }
}

// GET — list all entries (most recent first; matches frontend expectation)
app.get('/api/history', (req, res) => {
    res.json(readHistory());
});

// POST — append a new entry. Caps at HISTORY_MAX_ENTRIES; oldest dropped first.
app.post('/api/history', (req, res) => {
    const { name, content, words, language, audioFile } = req.body || {};
    if (!name || !content) {
        return res.status(400).json({ error: 'bad_request', message: 'name and content are required' });
    }
    const h = readHistory();
    h.unshift({
        id: Date.now(),
        name: String(name).slice(0, 200),
        content: String(content),
        words: typeof words === 'number' ? words : parseInt(words) || 8,
        language: language || 'auto',
        audioFile: audioFile || null,
        date: new Date().toISOString()
    });
    // Cap + cascade-delete audio of dropped entries so disk doesn't bloat
    while (h.length > HISTORY_MAX_ENTRIES) {
        const dropped = h.pop();
        deleteStoredAudio(dropped.audioFile);
    }
    writeHistory(h);
    res.json({ ok: true, count: h.length });
});

// DELETE one entry by id
app.delete('/api/history/:id', (req, res) => {
    const id = Number(req.params.id);
    let h = readHistory();
    const dropped = h.find(x => x.id === id);
    h = h.filter(x => x.id !== id);
    if (dropped) deleteStoredAudio(dropped.audioFile);
    writeHistory(h);
    res.json({ ok: true });
});

// DELETE everything
app.delete('/api/history', (req, res) => {
    const h = readHistory();
    h.forEach(item => deleteStoredAudio(item.audioFile));
    writeHistory([]);
    res.json({ ok: true });
});

// Resume in-flight or recently-finished jobs after a page reload.
// Returns: { status: 'running' | 'ok' | 'error', progress?, result?, error? }
app.get('/api/result/:jobId', (req, res) => {
    const jobId = req.params.jobId;
    const cached = jobResults.get(jobId);
    if (cached) {
        return res.json({ status: cached.status, ...(cached.result ? { result: cached.result } : {}), ...(cached.error ? { error: cached.error } : {}), completedAt: cached.completedAt });
    }
    const progress = activeJobs.get(jobId);
    if (progress) {
        return res.json({ status: 'running', progress });
    }
    res.status(404).json({ status: 'unknown', message: 'Job not found (may have expired)' });
});

// Stream stored audio (used by the in-app player so playback stays in sync with the SRT)
app.get('/api/audio/:filename', (req, res) => {
    const safe = path.basename(req.params.filename); // strip any path traversal attempts
    const p = path.join(STORED_AUDIO_DIR, safe);
    if (!fs.existsSync(p)) {
        return res.status(404).json({ error: 'not_found', message: 'Audio not found' });
    }
    res.sendFile(p);
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

// ═══════════════════════════════════════════════════════════
//   TRANSCRIPTION HELPERS (used by vertex/auto/hybrid dispatcher)
// ═══════════════════════════════════════════════════════════

/**
 * Run the STT cascade: Whisper-large-v3 (HF) → chirp_2 (Google V2) → latest_long (Google V1).
 * Returns the first tier that produces words. Throws nothing — returns null if all tiers fail
 * so the caller can decide how to handle missing STT in hybrid vs auto modes.
 *
 * @returns {Promise<{words: Array, model: string} | null>}
 */
async function runSTTCascade(audioPath, gcsUri, language, jobId) {
    // Tier 1: HuggingFace Whisper-large-v3 (highest word coverage; chunked at silence
    // boundaries for files larger than HF's ~25MB upload cap). Gated by
    // ENABLE_WHISPER_FALLBACK so users can skip it entirely when HF is unreliable.
    if (hfClient && ENABLE_WHISPER_FALLBACK) {
        try {
            updateProgress(jobId, 'transcribing', 55, 'Transcribing with Whisper-large-v3...');
            const words = await getWordTimestampsWhisperHF(audioPath);
            if (words && words.length > 0) {
                return { words, model: 'whisper-large-v3 (HF)' };
            }
        } catch (e) {
            log('warn', 'HF Whisper unavailable — trying chirp_2', {
                jobId: jobId.substring(0, 8), error: e.message,
            });
        }
    } else if (hfClient && !ENABLE_WHISPER_FALLBACK) {
        log('info', 'Skipping HF Whisper (ENABLE_WHISPER_FALLBACK=false) — going straight to chirp_2',
            { jobId: jobId.substring(0, 8) });
    }

    // Tier 2: chirp_2 (Google's newest universal speech model — V2 API).
    try {
        updateProgress(jobId, 'transcribing', 60, 'Analyzing speech with chirp_2 model...');
        const words = await getWordTimestampsChirp2(gcsUri, language);
        if (words && words.length > 0) {
            return { words, model: 'chirp_2' };
        }
    } catch (e) {
        log('warn', 'chirp_2 failed — trying latest_long', {
            jobId: jobId.substring(0, 8), error: e.message,
        });
    }

    // Tier 3: latest_long (V1 API — reliable baseline).
    try {
        updateProgress(jobId, 'transcribing', 65, 'Analyzing speech with latest_long model...');
        const words = await getWordTimestamps(gcsUri, language);
        if (words && words.length > 0) {
            return { words, model: 'latest_long' };
        }
    } catch (e) {
        log('warn', 'latest_long failed', {
            jobId: jobId.substring(0, 8), error: e.message,
        });
    }

    return null;
}

/**
 * Run Gemini-on-Vertex for transcription. Uses chunked path for long audio
 * (>VERTEX_CHUNK_THRESHOLD_SEC) to dodge silent truncation. Returns the cleaned
 * JSON string ([{start, end, text}, ...]). Throws on safety block, MAX_TOKENS
 * with empty content, or empty response. Caller wraps in try/catch when running
 * in parallel with STT (hybrid mode).
 *
 * @returns {Promise<string>} JSON string of segments
 */
async function runVertexGemini(audioPath, gcsUri, audioDuration, wordLimit, vocabulary, language, jobId) {
    // Long audio: chunk it. Vertex Gemini-2.5-Pro silently truncates audio >~15 min
    // in a single request — chunking forces full coverage.
    if (audioDuration > VERTEX_CHUNK_THRESHOLD_SEC) {
        log('info', `Audio ${audioDuration.toFixed(1)}s > ${VERTEX_CHUNK_THRESHOLD_SEC}s — using Vertex chunking`,
            { jobId: jobId.substring(0, 8) });
        return await transcribeVertexInChunks(
            audioPath, audioDuration, wordLimit, vocabulary, language, jobId
        );
    }

    // Single-shot path for shorter audio
    let response = null;
    for (let attempt = 1; attempt <= 2; attempt++) {
        try {
            log('info', `Gemini attempt ${attempt}`, { jobId: jobId.substring(0, 8), model: GEMINI_MODEL });
            response = await aiClient.models.generateContent({
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
                        { text: buildEnhancedPrompt(wordLimit, vocabulary, audioDuration, language) },
                        { fileData: { mimeType: 'audio/wav', fileUri: gcsUri } }
                    ]
                }]
            });
            break;
        } catch (e) {
            log('warn', `Gemini attempt ${attempt} failed`, { error: e.message });
            if (attempt >= 2) throw new Error(`Gemini transcription failed: ${e.message}`);
            await new Promise(r => setTimeout(r, 15000));
        }
    }

    if (response.promptFeedback?.blockReason) {
        throw new Error(`Gemini content blocked by safety filters: ${response.promptFeedback.blockReason}`);
    }
    if (!response.candidates || response.candidates.length === 0) {
        throw new Error('Gemini returned no candidates');
    }
    const candidate = response.candidates[0];
    if (candidate.finishReason === 'SAFETY') {
        throw new Error('Gemini stopped due to safety');
    }
    if (candidate.finishReason === 'MAX_TOKENS') {
        log('warn', 'Gemini hit MAX_TOKENS — transcription may be incomplete', { jobId: jobId.substring(0, 8) });
    }

    const rawText = response?.text || candidate?.content?.parts?.[0]?.text || '';
    const cleaned = cleanOutput(rawText);
    if (!cleaned || cleaned.trim().length === 0) {
        throw new Error('Gemini returned empty response after cleaning');
    }
    return cleaned;
}

/**
 * Dead-zone retry pass: detect inter-cue gaps >minGapSec, extract just those
 * audio clips, re-run chirp_2 on them, and splice recovered words back in.
 *
 * Why this works: chirp_2 sometimes drops speech on long-form audio when there's
 * cross-talk, quiet speech, or background noise. Feeding it ONLY the missed
 * section (without surrounding context) often recovers the lost words —
 * confirmed empirically on voice.mp3 where gaps at 15:04 and 15:39 yielded
 * 45 new words on retry that were missed in the main pass.
 *
 * Processes gaps in parallel via Promise.allSettled. Failures on individual
 * gaps are logged but don't fail the whole pass.
 *
 * @param {string} jsonString - segments JSON from main STT pass
 * @param {string} audioPath - local WAV path (already transcoded)
 * @param {string} language - language hint (e.g. 'auto', 'en-US')
 * @param {string} jobId - for logging
 * @param {number} minGapSec - min gap size to trigger retry (default 10s)
 * @returns {Promise<string>} JSON with recovered segments spliced in
 */
async function fillDeadZonesViaRetry(jsonString, audioPath, language, jobId, minGapSec = 10) {
    try {
        let cleaned = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();
        const s = cleaned.indexOf('[');
        const e = cleaned.lastIndexOf(']');
        if (s === -1 || e === -1) return jsonString;
        cleaned = cleaned.substring(s, e + 1).replace(/,\s*([}\]])/g, '$1');

        const segments = JSON.parse(cleaned);
        if (!Array.isArray(segments) || segments.length < 2) return jsonString;

        segments.sort((a, b) => parseTimestamp(a.start) - parseTimestamp(b.start));

        // Find inter-cue gaps >= minGapSec
        const gaps = [];
        for (let i = 0; i < segments.length - 1; i++) {
            const endA = parseTimestamp(segments[i].end);
            const startB = parseTimestamp(segments[i + 1].start);
            const gapSize = startB - endA;
            if (gapSize >= minGapSec) {
                gaps.push({ start: endA, end: startB, dur: gapSize });
            }
        }

        if (gaps.length === 0) return jsonString;

        log('info', `Found ${gaps.length} dead zones >=${minGapSec}s — retrying chirp_2 on each`,
            { jobId: jobId.substring(0, 8), gaps: gaps.map(g => `${g.start.toFixed(0)}-${g.end.toFixed(0)}s`).join(',') });

        // Process gaps in parallel
        const retryPromises = gaps.map(async (gap) => {
            let localPath = null;
            let gcsName = null;
            try {
                // Small buffer (0.2s) on each side helps catch words right at the boundary
                const bufferedStart = Math.max(0, gap.start - 0.2);
                const bufferedDur = gap.dur + 0.4;
                localPath = await extractWavChunk(audioPath, bufferedStart, bufferedDur);
                gcsName = `srt-ai/deadzone-${jobId.substring(0, 8)}-${gap.start.toFixed(0)}-${Date.now()}.wav`;
                await gcsBucket.upload(localPath, {
                    destination: gcsName,
                    metadata: { contentType: 'audio/wav' },
                });
                const gcsUri = `gs://${GCS_BUCKET_NAME}/${gcsName}`;
                const words = await getWordTimestampsChirp2(gcsUri, language);

                if (!words || words.length === 0) return null;

                // Offset timestamps to global audio time
                for (const w of words) {
                    w.startTime += bufferedStart;
                    w.endTime += bufferedStart;
                }

                // Build cues from these words
                const subJson = buildSegmentsFromSTT(words, 8);
                const subSegs = JSON.parse(subJson);
                return Array.isArray(subSegs) && subSegs.length > 0 ? subSegs : null;
            } catch (err) {
                log('warn', `Dead-zone retry failed at ${gap.start.toFixed(0)}s`,
                    { jobId: jobId.substring(0, 8), error: err.message });
                return null;
            } finally {
                if (localPath) try { await fsPromises.unlink(localPath); } catch {}
                if (gcsName) try { await gcsBucket.file(gcsName).delete(); } catch {}
            }
        });

        const results = await Promise.allSettled(retryPromises);
        const newSegments = [];
        for (let i = 0; i < results.length; i++) {
            if (results[i].status === 'fulfilled' && results[i].value) {
                newSegments.push(...results[i].value);
                log('info', `Dead zone @${gaps[i].start.toFixed(0)}s: recovered ${results[i].value.length} cues`,
                    { jobId: jobId.substring(0, 8) });
            }
        }

        if (newSegments.length === 0) {
            log('info', 'Dead-zone retry: no words recovered from any gap', { jobId: jobId.substring(0, 8) });
            return jsonString;
        }

        const merged = [...segments, ...newSegments].sort((a, b) =>
            parseTimestamp(a.start) - parseTimestamp(b.start));

        log('info', `Dead-zone retry complete: ${segments.length} → ${merged.length} cues (+${newSegments.length})`,
            { jobId: jobId.substring(0, 8) });
        return JSON.stringify(merged);

    } catch (err) {
        log('warn', 'fillDeadZonesViaRetry failed', { error: err.message });
        return jsonString;
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

    // Video-editing standard: target wordLimit words per cue, allow up to +2 if
    // needed to avoid ending on a weak/incomplete word (so phrases like "far more"
    // don't get split). Hard caps prevent runaway cues and Premiere wrap-onto-2-lines.
    const TARGET_WORDS = wordLimit;
    const MAX_WORDS = wordLimit + 2;        // soft extend up to +2 words (e.g. 10)
    const MAX_CHARS = 60;                   // single-line in Premiere at standard sizes
    // Hard caps: only exceeded when a sentence-end is within 1-3 words and
    // we don't want to orphan it into a tiny next cue.
    const MAX_WORDS_HARD = wordLimit + 4;   // e.g. 12 — absolute ceiling
    const MAX_CHARS_HARD = 75;              // 1 line on Premiere at smaller font

    // Words that should NOT end a cue (they expect the next word to complete the
    // phrase). If we hit TARGET_WORDS but the last word is weak, we keep extending
    // up to MAX_WORDS / MAX_CHARS.
    const WEAK_TRAILING = new Set([
        // articles + prepositions
        'the', 'a', 'an', 'of', 'in', 'at', 'to', 'on', 'for', 'with', 'by',
        'from', 'as', 'into', 'onto', 'about', 'over', 'under',
        // copula / aux
        'is', 'was', 'are', 'were', 'be', 'been', 'being', 'am', 'has', 'have',
        'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
        'may', 'might', 'must', 'shall',
        // pronouns / possessives
        'he', 'she', 'it', 'they', 'we', 'you', 'i', 'his', 'her', 'their',
        'your', 'my', 'our', 'this', 'that', 'these', 'those',
        // conjunctions / connectives
        'and', 'but', 'or', 'so', 'if', 'than', 'that', 'because', 'though',
        // intensifiers that pair with the next word
        'very', 'too', 'quite', 'just', 'really', 'far', 'more', 'most',
        'less', 'least', 'much', 'many', 'some', 'any', 'all', 'no', 'not',
    ]);

    // Words that almost always begin a new clause/sentence in English.
    // When STT shows even a short gap before one of these, that's a real break.
    const CLAUSE_STARTERS = new Set([
        'but', 'and', 'or', 'so', 'because', 'however', 'although',
        'while', 'then', 'though', 'yet', 'still', 'also', 'plus',
        'meanwhile', 'instead', 'otherwise', 'therefore', 'thus',
    ]);

    // Clamp pathologically long word endpoints. latest_long (and occasionally
    // chirp_2) report word.endTime that includes trailing silence — e.g. "him."
    // gets endTime 12s after startTime because STT extends the endpoint to the
    // next speech onset. Cap each word's reported duration to a phonetically
    // reasonable max based on character length, so a single "yes" cue can't
    // stretch for 12 seconds in the final SRT.
    for (let i = 0; i < sttWords.length; i++) {
        const w = sttWords[i];
        const charLen = (w.word || '').replace(/[^a-zA-Z0-9']/g, '').length || 1;
        // 200ms/char + 600ms floor → "yes" caps at 1.2s, "hello" at 1.6s,
        // "antidisestablishmentarianism" at ~5.6s. Drawn-out vowels can exceed
        // this slightly but a clipped cue is far better than a 12s "With him."
        const maxWordDur = Math.max(0.6, charLen * 0.20);
        const reportedDur = w.endTime - w.startTime;
        if (reportedDur > maxWordDur) {
            w.endTime = w.startTime + maxWordDur;
        }
    }

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

        // PRE-PUSH cap check: if adding this word would exceed hard caps AND we
        // already have words in the group, flush the existing group first (without
        // the new word), then start a fresh group with this word.
        //
        // EXCEPTION: if a sentence-end (.!?) is within reach in the next 1-3 words,
        // allow up to MAX_CHARS_HARD / MAX_WORDS_HARD so the complete sentence
        // lands in one cue. Splitting "another disturbing event at the same horror
        // house | takes place." is worse than a slightly-over-50-char single cue.
        if (group.length > 0) {
            const tentativeText = group.map(g => g.word).join(' ') + ' ' + w.word;
            const tentativeChars = tentativeText.length;
            const tentativeWords = group.length + 1;
            const overSoftCap = tentativeChars > MAX_CHARS || tentativeWords > MAX_WORDS;

            if (overSoftCap) {
                // Look ahead: does a sentence-end appear within next 3 words?
                let sentenceWithinReach = false;
                let projectedChars = tentativeChars;
                let projectedWords = tentativeWords;
                if (SENTENCE_END.test(w.word)) {
                    sentenceWithinReach = true;
                } else {
                    for (let look = 1; look <= 3 && (i + look) < words.length; look++) {
                        const lw = words[i + look];
                        projectedChars += 1 + lw.word.length;
                        projectedWords += 1;
                        if (projectedChars > MAX_CHARS_HARD || projectedWords > MAX_WORDS_HARD) break;
                        if (SENTENCE_END.test(lw.word)) {
                            sentenceWithinReach = true;
                            break;
                        }
                    }
                }

                const withinHardCaps = tentativeChars <= MAX_CHARS_HARD && tentativeWords <= MAX_WORDS_HARD;
                if (!(sentenceWithinReach && withinHardCaps)) {
                    flush();
                }
                // else: defer flush; the sentence-end will flush on a later iteration.
            }
        }

        group.push(w);

        const cueText = group.map(g => g.word).join(' ');
        const cueChars = cueText.length;

        const naturalPause = next && (next.startTime - w.endTime) >= PAUSE_THRESHOLD;
        const hitTarget = group.length >= TARGET_WORDS;
        // Iteration-level caps use HARD values — soft caps (MAX_WORDS/MAX_CHARS)
        // are enforced by the PRE-PUSH check, which allows extending past them when
        // a sentence-end is within reach. Using soft caps here would override that
        // bypass and chop sentences mid-flight.
        const hitMaxWords = group.length >= MAX_WORDS_HARD;
        const hitMaxChars = cueChars >= MAX_CHARS_HARD;
        const isLast = !next;
        const sentenceEnd = SENTENCE_END.test(w.word);
        const clauseEnd = CLAUSE_END.test(w.word) && group.length >= MIN_WORDS_FOR_SOFT_BREAK;

        // Is the current last word "weak" (article/prep/aux/intensifier)?
        // If yes, flushing here would orphan an incomplete phrase.
        const lastClean = w.word.toLowerCase().replace(/[^a-z']/g, '');
        const endsOnWeak = WEAK_TRAILING.has(lastClean);

        // Always flush after sentence-ending punctuation — never carry a
        // sentence across a subtitle break.
        if (sentenceEnd && !isLast) {
            flush();
            continue;
        }

        // Hard caps: always flush regardless of weak-trailing rule.
        if (isLast || hitMaxWords || hitMaxChars) {
            flush();
            continue;
        }

        // Natural break points: pause or clause-end punctuation.
        // EXCEPTION: if a sentence-end (.!?) is within reach in next 1-3 words
        // AND we'd stay within hard caps, defer the flush so the complete
        // sentence lands in one cue even if the speaker breathed mid-sentence.
        if (naturalPause || clauseEnd) {
            const headroom = MAX_WORDS_HARD - group.length;
            let sentenceWithinReach = false;
            if (headroom > 0) {
                let projectedChars = cueChars;
                for (let look = 1; look <= headroom && (i + look) < words.length; look++) {
                    const lw = words[i + look];
                    projectedChars += 1 + lw.word.length;
                    if (projectedChars > MAX_CHARS_HARD) break;
                    if (SENTENCE_END.test(lw.word)) {
                        sentenceWithinReach = true;
                        break;
                    }
                }
            }
            if (!sentenceWithinReach) {
                flush();
                continue;
            }
            // else: sentence completes within reach — defer the pause-flush.
        }

        // Target-hit: flush only if NOT ending on a weak word.
        // If ending on weak, keep extending until MAX or non-weak word.
        if (hitTarget && !endsOnWeak) {
            // SENTENCE-COMPLETION LOOK-AHEAD: before flushing at target,
            // peek forward up to (MAX_WORDS - group.length) words. If a
            // sentence-ender (.!?) is within reach AND the projected char
            // count stays under MAX_CHARS, keep extending so the complete
            // sentence lives in one cue.
            //
            // This prevents splits like:
            //   ❌ "another disturbing event at the same horror house"  (8w, target hit)
            //   ❌ "takes place."                                       (2w, orphan)
            // And produces instead:
            //   ✅ "another disturbing event at the same horror house takes place."  (10w, complete)
            // Look-ahead uses HARD caps so a single complete sentence isn't
            // chopped just because it pushes 1-2 chars over MAX_CHARS soft cap.
            const headroom = MAX_WORDS_HARD - group.length;
            let sentenceWithinReach = false;
            if (headroom > 0) {
                let projectedChars = cueChars;
                for (let look = 1; look <= headroom && (i + look) < words.length; look++) {
                    const lw = words[i + look];
                    projectedChars += 1 + lw.word.length;
                    if (projectedChars > MAX_CHARS_HARD) break;
                    if (SENTENCE_END.test(lw.word)) {
                        sentenceWithinReach = true;
                        break;
                    }
                }
            }
            if (!sentenceWithinReach) {
                flush();
            }
            // else: keep building; sentence-end will trigger flush on later iteration
        }
    }

    return JSON.stringify(segments);
}

/**
 * Fill gaps in Gemini-aligned output using chirp_2 word timestamps.
 *
 * alignJsonTimestamps only fixes timestamps — it can't add cues for speech
 * that Gemini missed in its text output. This function scans the merged
 * output for inter-cue gaps > minGapSec, looks up any STT words in those
 * gaps, and inserts them as new cues built by buildSegmentsFromSTT.
 *
 * Without this, mid-sentence drops (verified at 5:28, 12:51 in voice.mp3)
 * stay missing in the final SRT even though chirp_2 caught the words.
 *
 * @param {string} jsonString - aligned JSON segments from alignJsonTimestamps
 * @param {Array<{word,startTime,endTime}>} sttWords - chirp_2 word list
 * @param {number} minGapSec - minimum gap size to fill (default 3.0s)
 * @param {number} wordLimit - words per inserted cue
 * @returns {string} JSON with gap-fill segments inserted
 */
function fillSttGaps(jsonString, sttWords, minGapSec = 3.0, wordLimit = 8) {
    if (!sttWords || sttWords.length === 0) return jsonString;
    try {
        let cleaned = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();
        const s = cleaned.indexOf('[');
        const e = cleaned.lastIndexOf(']');
        if (s === -1 || e === -1) return jsonString;
        cleaned = cleaned.substring(s, e + 1).replace(/,\s*([}\]])/g, '$1');

        const segments = JSON.parse(cleaned);
        if (!Array.isArray(segments) || segments.length === 0) return jsonString;

        // Sort segments by start time
        segments.sort((a, b) => parseTimestamp(a.start) - parseTimestamp(b.start));

        const result = [];
        let prevEnd = 0;
        let totalInserted = 0;

        for (let i = 0; i <= segments.length; i++) {
            const cur = segments[i];
            const curStart = cur ? parseTimestamp(cur.start) : Infinity;
            const gap = curStart - prevEnd;

            if (gap >= minGapSec) {
                // Find STT words in this gap (with small buffer to avoid stealing edge words)
                const wordsInGap = sttWords.filter(w =>
                    w.startTime >= prevEnd + 0.05 &&
                    w.endTime <= curStart - 0.05
                );

                if (wordsInGap.length > 0) {
                    const fillJson = buildSegmentsFromSTT(wordsInGap, wordLimit);
                    try {
                        const fillSegs = JSON.parse(fillJson);
                        if (Array.isArray(fillSegs) && fillSegs.length > 0) {
                            result.push(...fillSegs);
                            totalInserted += fillSegs.length;
                        }
                    } catch { /* skip malformed fill */ }
                }
            }

            if (cur) {
                result.push(cur);
                prevEnd = parseTimestamp(cur.end);
            }
        }

        if (totalInserted > 0) {
            log('info', `STT gap-fill: inserted ${totalInserted} cues into ${segments.length}-segment alignment`);
        }
        return JSON.stringify(result);
    } catch (err) {
        log('warn', '[fillSttGaps] failed, keeping aligned output as-is', { error: err.message });
        return jsonString;
    }
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

        // ─── Step 1: Build flat Gemini-word list, tagged with originating segment ───
        // Each entry: { text, segIdx, posInSeg, posInDoc }
        const aiWords = [];
        segments.forEach((seg, segIdx) => {
            const tokens = (seg.text || '').replace(/[^a-zA-Z0-9'\s]/g, '').split(/\s+/).filter(Boolean);
            tokens.forEach((tok, posInSeg) => {
                aiWords.push({ text: norm(tok), segIdx, posInSeg, posInDoc: aiWords.length });
            });
        });
        if (aiWords.length === 0 || sttWords.length === 0) return jsonString;

        // ─── Step 2: Build STT word index for O(1) lookups: word → [sttIdx, …] ───
        const sttIndex = new Map();
        for (let i = 0; i < sttWords.length; i++) {
            const w = norm(sttWords[i].word);
            if (!w) continue;
            if (!sttIndex.has(w)) sttIndex.set(w, []);
            sttIndex.get(w).push(i);
        }

        // ─── Step 3: Anchor pass — for each Gemini word, find best STT match ───
        // Strategy: bigram match preferred (current + next word both match), unigram fallback.
        // Search is constrained to a sliding window around the previously-matched STT index
        // so we stay monotonic and don't re-match an early "the" for a late "the".
        // Crucially: STT may have transcribed a *different* speaker than Gemini (narrator
        // vs. body-cam). When that's the case for a given word, we simply skip it — the
        // segment gets timestamps interpolated from anchors in its neighbours.
        const anchors = []; // [{ aiIdx, sttIdx, score }]  score: 2=bigram, 1=unigram
        let lastSttIdx = 0;          // monotonic floor — anchors only go forward
        const WIN_BACK = 30;         // allow some backtrack for missed words
        const WIN_FWD  = 200;        // forward search window (200 STT words ≈ ~1 min audio)

        for (let ai = 0; ai < aiWords.length; ai++) {
            const target = aiWords[ai].text;
            const nextTarget = aiWords[ai + 1]?.text;
            const candidates = sttIndex.get(target);
            if (!candidates || candidates.length === 0) continue;

            // Find candidate STT indices within the search window
            const lo = Math.max(0, lastSttIdx - WIN_BACK);
            const hi = Math.min(sttWords.length, lastSttIdx + WIN_FWD);

            let bestSttIdx = -1;
            let bestScore = 0;
            for (const cand of candidates) {
                if (cand < lo || cand >= hi) continue;
                // Score: bigram match preferred. Closer-to-lastSttIdx as tiebreaker.
                let score = 1;
                if (nextTarget && cand + 1 < sttWords.length && norm(sttWords[cand + 1].word) === nextTarget) {
                    score = 2;
                }
                if (score > bestScore || (score === bestScore && bestSttIdx !== -1 && cand < bestSttIdx)) {
                    bestScore = score;
                    bestSttIdx = cand;
                }
            }

            if (bestSttIdx !== -1) {
                anchors.push({ aiIdx: ai, sttIdx: bestSttIdx, score: bestScore });
                lastSttIdx = bestSttIdx + 1;
            }
        }

        if (anchors.length === 0) {
            // No alignment possible — keep AI timestamps as-is
            return jsonString;
        }

        // ─── Step 4: For each segment, derive start/end from anchors ───
        // Find anchors whose aiIdx falls inside the segment's Gemini-word range.
        // If 1+ anchors inside: use first/last anchor times directly.
        // If 0 anchors: interpolate from nearest before/after anchors (preserves monotonicity).
        // Build per-segment Gemini-word index ranges for quick anchor filtering
        const segRange = segments.map(() => ({ firstAiIdx: -1, lastAiIdx: -1 }));
        for (const w of aiWords) {
            const r = segRange[w.segIdx];
            if (r.firstAiIdx === -1) r.firstAiIdx = w.posInDoc;
            r.lastAiIdx = w.posInDoc;
        }

        // Binary helpers to find nearest anchor before/after a given aiIdx
        const findAnchorIn = (firstAi, lastAi) => {
            // anchors sorted by aiIdx ascending (built that way)
            const inside = anchors.filter(a => a.aiIdx >= firstAi && a.aiIdx <= lastAi);
            return inside;
        };
        const findAnchorBefore = (aiIdx) => {
            for (let i = anchors.length - 1; i >= 0; i--) if (anchors[i].aiIdx < aiIdx) return anchors[i];
            return null;
        };
        const findAnchorAfter = (aiIdx) => {
            for (let i = 0; i < anchors.length; i++) if (anchors[i].aiIdx > aiIdx) return anchors[i];
            return null;
        };

        let lastSegEndSec = 0;
        for (let si = 0; si < segments.length; si++) {
            const seg = segments[si];
            const { firstAiIdx, lastAiIdx } = segRange[si];
            if (firstAiIdx === -1) continue; // empty text

            const inSeg = findAnchorIn(firstAiIdx, lastAiIdx);

            let startSec, endSec;
            if (inSeg.length > 0) {
                // Use first/last anchored words' STT times
                const firstAnc = inSeg[0];
                const lastAnc = inSeg[inSeg.length - 1];
                startSec = sttWords[firstAnc.sttIdx].startTime;
                endSec = sttWords[lastAnc.sttIdx].endTime;
                // If only 1 anchor and it's not at the segment's first word, the first
                // word started slightly earlier — back off by (anchor-pos × 0.25s per word)
                if (inSeg.length === 1) {
                    const wordsBeforeAnchor = firstAnc.aiIdx - firstAiIdx;
                    if (wordsBeforeAnchor > 0) startSec = Math.max(lastSegEndSec, startSec - wordsBeforeAnchor * 0.25);
                    const wordsAfterAnchor = lastAiIdx - lastAnc.aiIdx;
                    if (wordsAfterAnchor > 0) endSec = endSec + wordsAfterAnchor * 0.25;
                }
            } else {
                // No anchors in this segment — interpolate between neighbors
                const before = findAnchorBefore(firstAiIdx);
                const after = findAnchorAfter(lastAiIdx);
                if (before && after) {
                    // Linear interpolate by Gemini-word position
                    const tBefore = sttWords[before.sttIdx].endTime;
                    const tAfter = sttWords[after.sttIdx].startTime;
                    const aiSpan = after.aiIdx - before.aiIdx;
                    if (aiSpan > 0) {
                        const ratio1 = (firstAiIdx - before.aiIdx) / aiSpan;
                        const ratio2 = (lastAiIdx - before.aiIdx) / aiSpan;
                        startSec = tBefore + (tAfter - tBefore) * ratio1;
                        endSec = tBefore + (tAfter - tBefore) * ratio2;
                    } else {
                        startSec = tBefore;
                        endSec = tAfter;
                    }
                } else if (before) {
                    // Extrapolate forward from last known anchor — add ~0.3s per word
                    startSec = sttWords[before.sttIdx].endTime + (firstAiIdx - before.aiIdx) * 0.3;
                    endSec = startSec + (lastAiIdx - firstAiIdx + 1) * 0.3;
                } else if (after) {
                    // Extrapolate backward from first anchor
                    endSec = sttWords[after.sttIdx].startTime - (after.aiIdx - lastAiIdx) * 0.3;
                    startSec = endSec - (lastAiIdx - firstAiIdx + 1) * 0.3;
                } else {
                    // No anchors at all — keep AI timestamps
                    continue;
                }
            }

            // Enforce monotonicity: never go backward from previous segment's end
            if (startSec < lastSegEndSec) startSec = lastSegEndSec;
            if (endSec <= startSec) endSec = startSec + 0.5;

            seg.start = formatTimestamp(startSec);
            seg.end = formatTimestamp(endSec);
            lastSegEndSec = endSec;
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

ABSOLUTE PROHIBITIONS (read carefully — violating these breaks the pipeline):
- DO NOT add narrator-style commentary describing what is happening in the audio.
  Forbidden examples: "The suspect takes her time", "She moves out of the hotel",
  "The officers are losing patience", "Now the drama begins", "Just after that...".
- DO NOT summarize, recap, or re-transcribe any portion of the audio. Each
  spoken utterance must appear EXACTLY ONCE in the output array.
- DO NOT describe scenes, actions, emotions, or events — transcribe ONLY the
  literal words a human voice says.
- If the audio itself contains a YouTube/documentary voice-over, transcribe it
  VERBATIM as you hear it. NEVER add your own narration on top of it.
- DO NOT include any cue whose text wasn't actually spoken in the audio.

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

/**
 * Pre-filter pass: drop garbage and duplicate cues before timing fixes.
 *
 * Catches three pathologies observed in chirp_2 output:
 *   1. Duplicate-start segments — when chirp_2 emits both a short and long
 *      version of the same utterance at the same timestamp. Keep the longer.
 *   2. Repeated-token garbage — cues whose text is just one token repeated
 *      ("0 0 0 0 0 0 0 0"). chirp_2 emits these on phone-dial tones, beeps,
 *      or distorted non-speech audio.
 *   3. Empty/whitespace text.
 */
function dedupAndFilterSegments(segments) {
    if (!segments || segments.length === 0) return [];

    // 1. Drop empty + repeated-token garbage
    const filtered = segments.filter(seg => {
        const text = (seg.text || '').trim();
        if (!text) return false;
        const tokens = text.split(/\s+/);
        // If 3+ tokens and all the same (case-insensitive, ignoring punctuation),
        // treat as STT garbage on non-speech audio.
        if (tokens.length >= 3) {
            const norm = tokens.map(t => t.toLowerCase().replace(/[^a-z0-9]/g, ''));
            const allSame = norm.every(t => t === norm[0]);
            if (allSame) {
                log('debug', `dedupAndFilter: dropping repeated-token garbage cue "${text}"`);
                return false;
            }
        }
        return true;
    });

    // 2. Sort by start time, then by text length DESC so longer text wins dedup
    filtered.sort((a, b) => {
        const sa = parseTimestamp(a.start);
        const sb = parseTimestamp(b.start);
        if (sa !== sb) return sa - sb;
        return (b.text?.length || 0) - (a.text?.length || 0);
    });

    // 3. Drop duplicate-start segments: when two cues start within 100ms,
    //    keep the longer-text one. Common when chirp_2 emits prefix duplicates
    //    (e.g., "Right" + "Right. Would you say...").
    const result = [];
    for (let i = 0; i < filtered.length; i++) {
        const cur = filtered[i];
        const curStart = parseTimestamp(cur.start);

        // If previous result starts within 100ms and is a strict prefix of
        // current text, replace previous. Otherwise just append.
        if (result.length > 0) {
            const prev = result[result.length - 1];
            const prevStart = parseTimestamp(prev.start);
            if (Math.abs(curStart - prevStart) < 0.1) {
                // Same-start duplicate. Keep whichever has more text.
                if ((cur.text?.length || 0) > (prev.text?.length || 0)) {
                    log('debug', `dedupAndFilter: replacing duplicate-start "${prev.text}" with "${cur.text}"`);
                    result[result.length - 1] = cur;
                } else {
                    log('debug', `dedupAndFilter: dropping duplicate-start "${cur.text}"`);
                }
                continue;
            }
        }
        result.push(cur);
    }

    // 4. Drop cross-timeline duplicates: same text appearing >30s apart.
    //    Almost always a Gemini "recap" hallucination, not real replayed audio.
    //    Verified on bodycam audio where Gemini emitted the same dialogue twice
    //    with the second copy stamped near the audio end.
    const seenText = new Map();          // normalizedText → first occurrence start (sec)
    const finalResult = [];
    let crossDupDropped = 0;
    for (const seg of result) {
        const key = (seg.text || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
        // Don't dedup very short cues — "Yes." / "Okay." legitimately repeat
        if (key.length < 12) { finalResult.push(seg); continue; }
        const firstStart = seenText.get(key);
        const curStart = parseTimestamp(seg.start);
        if (firstStart !== undefined && curStart - firstStart > 30) {
            log('debug', `dedupAndFilter: dropping cross-timeline duplicate "${seg.text}" (orig @${firstStart.toFixed(1)}s, dup @${curStart.toFixed(1)}s)`);
            crossDupDropped++;
            continue;
        }
        if (firstStart === undefined) seenText.set(key, curStart);
        finalResult.push(seg);
    }

    if (filtered.length !== segments.length || finalResult.length !== filtered.length) {
        log('info', `dedupAndFilter: ${segments.length} → ${finalResult.length} cues (dropped ${segments.length - finalResult.length}, of which ${crossDupDropped} cross-timeline dups)`);
    }
    return finalResult;
}

/**
 * Enforce a minimum cue display duration so single-word/filler cues
 * (chirp_2 acoustic endpoints of ~40-120ms) extend long enough to read.
 * Never extends into the next cue's start — preserves non-overlap.
 *
 * @param {number} minMs - target minimum cue duration (e.g., 700ms)
 */
function enforceMinDisplayDuration(segments, minMs = 700) {
    if (!segments || segments.length === 0) return segments;
    const minSec = minMs / 1000;
    const result = [];
    for (let i = 0; i < segments.length; i++) {
        const seg = { ...segments[i] };
        const startSec = parseTimestamp(seg.start);
        const endSec = parseTimestamp(seg.end);
        const curDur = endSec - startSec;

        if (curDur < minSec && i < segments.length - 1) {
            const nextStart = parseTimestamp(segments[i + 1].start);
            // Extend up to minSec or just before next cue, whichever is smaller.
            // 50ms safety buffer so we never collide with the next cue's start.
            const maxAllowedEnd = nextStart - 0.05;
            const targetEnd = Math.min(startSec + minSec, maxAllowedEnd);
            if (targetEnd > endSec) {
                seg.end = formatTimestamp(targetEnd);
            }
        } else if (curDur < minSec && i === segments.length - 1) {
            // Last cue: safe to extend to full minSec
            seg.end = formatTimestamp(startSec + minSec);
        }
        result.push(seg);
    }
    return result;
}

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

        // These caps must mirror buildSegmentsFromSTT's HARD caps so the post-processor
        // doesn't re-split cues that the upstream segmenter just carefully kept whole.
        // Previously: maxWords=12 / maxChars=50, which chopped 67-char complete-sentence
        // cues like "Finally, the court decides that the trial will begin in August 2024."
        // back into two pieces.
        const maxWords = (parseInt(wordLimit) || 8) + 4;   // matches MAX_WORDS_HARD
        const maxChars = 75;                                // matches MAX_CHARS_HARD

        // STEP 1: Parse, validate timestamps, and smart-split long segments
        // Drop any cue whose start is beyond actual audio length — these are
        // pure Gemini hallucinations (the model invents content with timestamps
        // past the audio end, especially after chunked transcription). Verified
        // on 282s audio where Gemini emitted cues stamped up to 5:55 (354s).
        // A small margin (2s) tolerates rounding error from ffprobe.
        const AUDIO_END_TOLERANCE = 2.0;
        const audioEndLimit = audioDurationSec ? audioDurationSec + AUDIO_END_TOLERANCE : Infinity;
        let droppedBeyondAudio = 0;
        const filteredSegments = segments.filter(seg => {
            const startSec = parseTimestamp(seg.start);
            if (startSec > audioEndLimit) {
                droppedBeyondAudio++;
                return false;
            }
            return true;
        });
        if (droppedBeyondAudio > 0) {
            log('warn', `jsonToSrt: dropped ${droppedBeyondAudio} cues beyond audio duration (${audioDurationSec?.toFixed(1)}s)`);
        }

        filteredSegments.forEach((seg) => {
            let tStart = parseTimestamp(seg.start);
            let tEnd = parseTimestamp(seg.end);

            // Also clamp end to audio duration (cues that START before end but
            // EXTEND beyond it get trimmed back).
            if (audioDurationSec && tEnd > audioDurationSec + AUDIO_END_TOLERANCE) {
                tEnd = audioDurationSec;
            }

            // VALIDATION: Ensure end > start
            if (tEnd <= tStart) {
                const words = seg.text.trim().split(/\s+/);
                tEnd = tStart + Math.max(0.5, words.length * 0.25);
            }

            // VALIDATION: clamp cue durations to realistic speaking-rate.
            // Body-cam testing showed Gemini emitting 25-35s cues for 4-6 word
            // phrases because it extended cue.end across silence to the next
            // utterance. That makes a short phrase visually hang on screen for
            // half a minute. Cap at ~1.2s/word with a 4s floor (so even very
            // short cues get a readable minimum) and a 12s absolute ceiling
            // (no single subtitle should ever last 12+ seconds — split it).
            const duration = tEnd - tStart;
            const wordCount = seg.text.trim().split(/\s+/).filter(Boolean).length || 1;
            const sanityCap = Math.max(4.0, Math.min(12.0, wordCount * 1.2));
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

        // STEP 1.5: Drop chirp_2 pathologies — duplicate-start prefix cues and
        // repeated-token garbage (e.g., "0 0 0 0 0 0 0 0" on non-speech audio).
        finalSegments = dedupAndFilterSegments(finalSegments);

        // STEP 2: Fix overlapping timestamps FIRST so adjacent cues are non-overlapping
        // before merging. (Whisper word boundaries often overlap by 1-50ms; if we
        // merge first, mergeCloseSegments skips them because gap<0, and overlap
        // remains in the final SRT.)
        finalSegments = compactOverlaps(finalSegments, 300);

        // STEP 3: Merge very close segments (< 150ms gap) to prevent visual breaks
        finalSegments = mergeCloseSegments(finalSegments, 150, maxWords, maxChars);

        // STEP 4: Bridge tiny gaps (< 250ms) for visual continuity
        let bridgedSegments = bridgeGaps(finalSegments, 250);

        // STEP 5: Enforce minimum 700ms cue duration for readability —
        // chirp_2's acoustic word endpoints can be 40-120ms long, which flashes
        // on screen too briefly to read. Extends up to next cue's start (never
        // overlaps). Most impactful UX fix on STT-driven output.
        bridgedSegments = enforceMinDisplayDuration(bridgedSegments, 700);

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

app.listen(PORT, async () => {
    log('info', '══════════════════════════════════════════');
    log('info', `🎬 SRT-AI Server v2.0.0 (Production Ready)`);
    log('info', `📍 URL:   http://localhost:${PORT}`);
    log('info', `🤖 Model: ${GEMINI_MODEL}`);
    log('info', `⚙️  Environment: ${NODE_ENV}`);
    log('info', `⏱️  Timeout: ${REQUEST_TIMEOUT_MS / 1000}s`);
    log('info', `📊 Rate Limit: ${RATE_LIMIT_REQUESTS} req/hour`);
    log('info', `📝 Log Level: ${LOG_LEVEL}`);

    // Capability probe: source separation (Demucs)
    if (process.env.ENABLE_SOURCE_SEPARATION === 'true') {
        const ok = await isSourceSeparationAvailable();
        if (ok) {
            log('info', `🔊 Source separation: enabled (Demucs ready)`);
        } else {
            log('warn', `🔊 Source separation: enabled in env but Demucs NOT installed.`);
            log('warn', `   Install with: python3 -m pip install --user demucs`);
            log('warn', `   Pipeline will fall back to original audio per request.`);
        }
    } else {
        log('info', `🔊 Source separation: disabled (set ENABLE_SOURCE_SEPARATION=true to enable)`);
    }

    log('info', '══════════════════════════════════════════');
});


