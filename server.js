import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import os from 'os';
import path from 'path';
import { promises as fsPromises } from 'fs';
import crypto from 'crypto';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import { fileURLToPath } from 'url';

// --- NEW SDK IMPORT ---
import { GoogleGenAI } from '@google/genai';

// Load environment variables
dotenv.config();

console.log('--- SERVER STARTING (Gemini 2.x Architecture) ---');

// ═══════════════════════════════════════════════════════════
//   CONFIGURATION
// ═══════════════════════════════════════════════════════════

const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';
const isProduction = NODE_ENV === 'production';

// Default to Gemini 2.5 Pro (Best quality)
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-pro';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Validate API Key
if (!GEMINI_API_KEY) {
    console.error('\n[FATAL ERROR] GEMINI_API_KEY is missing.');
    console.error('Please add it to your .env file.\n');
    process.exit(1);
}

// ═══════════════════════════════════════════════════════════
//   SETUP & INITIALIZATION
// ═══════════════════════════════════════════════════════════

// 1. Initialize Google GenAI Client
const aiClient = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// 2. Setup FFmpeg
let ffmpegPath = ffmpegStatic || 'ffmpeg';
ffmpeg.setFfmpegPath(ffmpegPath);
console.log(`[Setup] FFmpeg Path: ${ffmpegPath}`);

// 3. Express App & Middleware
const app = express();

if (isProduction) {
    app.set('trust proxy', 1);
}

app.use(cors({
    origin: true,
    credentials: true
}));

app.use(express.json({ limit: '1mb' }));

// 4. Static Files (Frontend)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const staticRoot = process.env.STATIC_ROOT || __dirname;
app.use(express.static(staticRoot));

// 5. Upload Configuration (Supports 1+ hour files)
const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2GB

const upload = multer({
    storage: multer.diskStorage({
        destination: os.tmpdir(),
        filename: (req, file, cb) => {
            const ext = path.extname(file.originalname) || '.bin';
            const safeName = `srt-ai-${crypto.randomUUID()}${ext}`;
            cb(null, safeName);
        }
    }),
    limits: {
        fileSize: MAX_FILE_SIZE,
    }
});

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
    
    console.log(`[Progress] ${jobId.substring(0, 8)}... | ${stage} | ${percent}% | ${message}`);
}

// ═══════════════════════════════════════════════════════════
//   API ROUTES
// ═══════════════════════════════════════════════════════════

/**
 * Health Check
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: 'online',
        system: 'Gemini 2.x Architecture',
        model: GEMINI_MODEL,
        time: new Date().toISOString()
    });
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
app.post('/api/transcribe', upload.single('file'), async (req, res) => {
    
    // 1. Validation
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    // Generate unique job ID
    const jobId = req.body.jobId || crypto.randomUUID();
    const startTime = Date.now();
    const cleanupPaths = [req.file.path];
    let geminiFileName = null;

    try {
        console.log(`\n[Job ${jobId.substring(0, 8)}] New Request: ${req.file.originalname}`);
        updateProgress(jobId, 'upload', 5, 'File received, starting processing...');

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

        // -------------------------------------------------------
        // STEP 2: Upload to Google AI (New SDK)
        // -------------------------------------------------------
        updateProgress(jobId, 'uploading', 45, 'Uploading to Google AI Studio...');
        
        const uploadResult = await aiClient.files.upload({
            file: audioPath,
            config: { 
                mimeType: 'audio/wav',
                displayName: `SRT_Job_${Date.now()}`
            }
        });

        geminiFileName = uploadResult.name;
        updateProgress(jobId, 'uploading', 50, 'Upload complete, waiting for processing...');

        // -------------------------------------------------------
        // STEP 3: Wait for Processing (supports 1+ hour files)
        // -------------------------------------------------------
        let fileState = uploadResult.state;
        let waitAttempts = 0;
        const maxWaitAttempts = 180; // 30 minutes max
        
        while (fileState === 'PROCESSING') {
            waitAttempts++;
            if (waitAttempts > maxWaitAttempts) {
                throw new Error('Gemini processing timeout - file may be too long');
            }
            
            // Map waiting progress from 50-70%
            const waitPercent = 50 + Math.min(20, Math.round(waitAttempts * 0.5));
            const elapsed = waitAttempts * 10;
            updateProgress(jobId, 'processing', waitPercent, `Gemini processing audio... (${elapsed}s elapsed)`);
            
            await new Promise(resolve => setTimeout(resolve, 10000));
            
            const fileStatus = await aiClient.files.get({ name: geminiFileName });
            fileState = fileStatus.state;
            
            if (fileState === 'FAILED') {
                throw new Error('Gemini failed to process the audio file.');
            }
        }
        
        updateProgress(jobId, 'processing', 70, 'Audio processed, generating transcription...');

        // -------------------------------------------------------
        // STEP 4: Generate Subtitles (JSON-First Architecture)
        // -------------------------------------------------------
        updateProgress(jobId, 'transcribing', 75, `Transcribing with ${GEMINI_MODEL}...`);
        
        const wordLimit = parseInt(req.body.wordLimit) || 6;
        const vocabulary = req.body.vocabulary ? req.body.vocabulary.split(',').map(v => v.trim()).filter(v => v) : [];
        
        const response = await aiClient.models.generateContent({
            model: GEMINI_MODEL,
            config: {
                temperature: 0.0,
                topP: 0.95,
                maxOutputTokens: 65536,
            },
            contents: [
                {
                    role: 'user',
                    parts: [
                        { text: buildEnhancedPrompt(wordLimit, vocabulary) },
                        { 
                            fileData: { 
                                mimeType: 'audio/wav', 
                                fileUri: uploadResult.uri 
                            } 
                        }
                    ]
                }
            ]
        });

        updateProgress(jobId, 'formatting', 90, 'Formatting SRT output...');

        // -------------------------------------------------------
        // STEP 5: Convert JSON to SRT (Perfect Formatting)
        // -------------------------------------------------------
        const jsonResponse = cleanOutput(response.text);
        const srtOutput = jsonToSrt(jsonResponse);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        
        updateProgress(jobId, 'complete', 100, `Completed in ${duration}s`);
        
        console.log(`[Success] Job ${jobId.substring(0, 8)} completed in ${duration}s`);
        res.json({ srt: srtOutput, jobId, duration });

    } catch (error) {
        console.error('[Error] Processing failed:', error);
        updateProgress(jobId, 'error', 0, error.message);
        
        const statusCode = error.message.includes('quota') ? 429 : 500;
        res.status(statusCode).json({ 
            error: 'Transcription failed',
            details: error.message,
            jobId
        });

    } finally {
        // Cleanup
        for (const p of cleanupPaths) {
            await deleteLocalFile(p);
        }

        if (geminiFileName) {
            try {
                await aiClient.files.delete({ name: geminiFileName });
            } catch (e) {
                // Ignore cleanup errors
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
//   HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

function buildEnhancedPrompt(wordLimit, vocabulary = []) {
    const vocabString = vocabulary.length > 0 
        ? `\n\nVOCABULARY LIST (Prioritize these spellings): ${vocabulary.join(', ')}` 
        : '';

    return `You are a forensic transcription engine. Your task is to transcribe audio to a JSON array.
${vocabString}

CRITICAL: You MUST return ONLY a valid JSON array. Nothing else. Start with [ and end with ].

EXAMPLE FORMAT (copy this structure exactly):

[{"start":"00:00:00,000","end":"00:00:03,500","text":"Hello, this is the first sentence."},{"start":"00:00:04,000","end":"00:00:07,200","text":"Thank you for that introduction."}]

REQUIREMENTS:
1. Start with [ character
2. End with ] character  
3. Each object must have: "start", "end", "text"
4. Timestamps: HH:MM:SS,mmm format (example: "00:00:05,250")
5. Maximum ${wordLimit} words per "text" field
6. Each segment 1-3 seconds duration
7. Break at natural pauses
8. Use exact vocabulary spellings if provided
9. Do NOT include speaker labels or speaker identification

DO NOT:
- Add markdown code blocks
- Add explanations before or after
- Add any text outside the JSON array
- Use newlines inside the JSON (keep it compact)
- Include speaker labels like "Speaker 1" or "[Speaker]"

OUTPUT NOW (JSON array only):`;
}

function transcodeToWav(inputPath, onProgress = () => {}) {
    const outputPath = path.join(os.tmpdir(), `srt-ai-hq-${crypto.randomUUID()}.wav`);
    
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error('Transcoding timed out - file may be too large'));
        }, 60 * 60 * 1000);

        ffmpeg(inputPath)
            .noVideo()
            .audioChannels(2)
            .audioFrequency(44100)
            .audioCodec('pcm_s16le')
            .format('wav')
            .audioFilters([
                'highpass=f=80',
                'lowpass=f=12000',
                'loudnorm=I=-16:TP=-1.5:LRA=11',
                'afftdn=nf=-25'
            ])
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
    } catch (e) {
        // Ignore
    }
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

function jsonToSrt(jsonString) {
    if (!jsonString || !jsonString.trim()) {
        return `1\n00:00:00,000 --> 00:00:05,000\n[Error] Empty transcription response.`;
    }
    
    try {
        let cleanJson = jsonString.trim();
        
        const jsonStart = cleanJson.indexOf('[');
        const jsonEnd = cleanJson.lastIndexOf(']');
        
        if (jsonStart === -1 || jsonEnd === -1 || jsonEnd <= jsonStart) {
            throw new Error('No JSON array found');
        }
        
        cleanJson = cleanJson.substring(jsonStart, jsonEnd + 1);
        cleanJson = cleanJson.replace(/,\s*([}\]])/g, '$1');
        
        const data = JSON.parse(cleanJson);
        
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Invalid or empty array');
        }
        
        return data.map((entry, index) => {
            const sequence = index + 1;
            const start = entry.start || '00:00:00,000';
            const end = entry.end || '00:00:00,000';
            const text = (entry.text || '').trim();
            
            if (!text) return null;
            
            return `${sequence}\n${start} --> ${end}\n${text}`;
        }).filter(Boolean).join('\n\n');
        
    } catch (e) {
        console.error('[Error] JSON parsing failed:', e.message);
        
        // Try recovery
        try {
            const objects = [];
            const regex = /\{"start":"([^"]+)","end":"([^"]+)","text":"([^"]+)"\}/g;
            let match;
            while ((match = regex.exec(jsonString)) !== null) {
                objects.push({ start: match[1], end: match[2], text: match[3] });
            }
            
            if (objects.length > 0) {
                return objects.map((entry, index) => {
                    const sequence = index + 1;
                    return `${sequence}\n${entry.start} --> ${entry.end}\n${entry.text}`;
                }).join('\n\n');
            }
        } catch (recoveryErr) {}
        
        return `1\n00:00:00,000 --> 00:00:05,000\n[Error] Failed to parse transcription.`;
    }
}

// ═══════════════════════════════════════════════════════════
//   START SERVER
// ═══════════════════════════════════════════════════════════

app.listen(PORT, () => {
    console.log(`\n══════════════════════════════════════════`);
    console.log(`  🎬 SRT-AI Server (Gemini 2.x)`);
    console.log(`  📍 URL:   http://localhost:${PORT}`);
    console.log(`  🤖 Model: ${GEMINI_MODEL}`);
    console.log(`  📊 Progress Tracking: Enabled`);
    console.log(`══════════════════════════════════════════\n`);
});

