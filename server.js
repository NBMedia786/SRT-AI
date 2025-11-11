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
import { GoogleGenerativeAI } from '@google/generative-ai';
import { fileURLToPath } from 'url';

dotenv.config();

if (ffmpegStatic) {
    ffmpeg.setFfmpegPath(ffmpegStatic);
} else {
    console.warn('[startup] ffmpeg-static binary not found. Ensure ffmpeg is available on PATH.');
}

const PORT = process.env.PORT || 3006;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-pro';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
    console.warn('[startup] GEMINI_API_KEY is not set. Transcription requests will fail.');
}

const app = express();
app.use(cors());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const staticRoot = process.env.STATIC_ROOT
    ? path.resolve(__dirname, process.env.STATIC_ROOT)
    : __dirname;

app.use(express.static(staticRoot));

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 750 * 1024 * 1024, // ~750MB
    },
});

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY || '');

app.post('/api/transcribe', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    if (!GEMINI_API_KEY) {
        return res.status(500).json({ error: 'Missing GEMINI_API_KEY configuration.' });
    }

    const userPrompt = (req.body?.prompt || '').toString().trim();
    const rawWordLimit = Number.parseInt(req.body?.wordLimit ?? '6', 10);
    const wordLimit = Number.isFinite(rawWordLimit) ? rawWordLimit : 6;
    const combinedPrompt = userPrompt || buildPrompt('', wordLimit);

    const tempInputPath = await writeBufferToTempFile(req.file.buffer, req.file.originalname || 'upload');

    let audioPath = tempInputPath;
    const cleanupPaths = [tempInputPath];

    try {
        const isVideo = req.file.mimetype?.startsWith('video');
        const isWav = req.file.mimetype === 'audio/wav' || path.extname(req.file.originalname || '').toLowerCase() === '.wav';

        if (isVideo || !isWav) {
            const wavPath = await transcodeToWav(tempInputPath);
            audioPath = wavPath;
            cleanupPaths.push(wavPath);
        }

        const audioBase64 = await fsPromises.readFile(audioPath, { encoding: 'base64' });

        const model = genAI.getGenerativeModel({ model: GEMINI_MODEL });
        const result = await model.generateContent([
            { text: combinedPrompt },
            {
                inlineData: {
                    mimeType: 'audio/wav',
                    data: audioBase64,
                },
            },
        ]);

        const responseText = extractText(result);
        const cleanedResponse = stripCodeFences(responseText);

        // --- THIS IS THE DEBUGGING LINE ---
        console.log('--- RAW GEMINI RESPONSE ---', cleanedResponse, '-----------------------------');

        const srt = cleanedResponse.trim();

        if (!srt) {
            throw new Error('Gemini returned an empty response.');
        }

        res.json({ srt });
    } catch (error) {
        console.error('[transcribe] failed', error);
        res.status(500).json({ error: error.message || 'Transcription failed.' });
    } finally {
        await Promise.allSettled(cleanupPaths.map(deleteIfExists));
    }
});

app.use((req, res, next) => {
    if (req.path.startsWith('/api')) {
        return next();
    }

    if (req.method !== 'GET') {
        return next();
    }

    res.sendFile(path.join(staticRoot, 'index.html'));
});

function buildPrompt(extraInstructions = '', wordLimit = 6) {
    const sanitizedInstructions = (extraInstructions || '').trim();
    const effectiveLimit = Math.max(3, Math.min(20, Number(wordLimit) || 6));
    let prompt = `You are an AI assistant specialized in generating professional-quality subtitle files (SRT) from video or audio transcriptions.

Input: 

- The raw transcription text extracted from a video/audio file.
- The preferred word limit per subtitle line (a number).
- Instructions to automatically break lines at punctuation marks (.,?,,) regardless of word limit.
- Instructions to intelligently adjust line lengths so that if a phrase exceeds or falls short of the word limit by a small margin, the adjustment is applied only to that phrase to maintain readability and natural flow.

Task:  

Format the transcription into a valid SRT subtitle format, including:

- Subtitle sequence numbers.
- Start and end timestamps for each subtitle (based on original transcription timing if available, or approximate reasonable splitting).
- Subtitle text formatted according to the word limit and punctuation-based line breaks described.
- Ensure the SRT file is clean, readable, and professional for use in video players.
- Each subtitle entry must contain exactly one line of dialogue text with no internal line breaks.

---

Transcription Text: """<Transcribe the attached media to obtain the raw transcript>"""

Word limit per line: ${effectiveLimit}

Please generate the formatted SRT file text accordingly.`;

    if (sanitizedInstructions) {
        prompt += `\n\nAdditional user instructions:\n${sanitizedInstructions}`;
    }

    return prompt;
}

function extractText(result) {
    if (!result?.response?.candidates?.length) return '';
    const parts = result.response.candidates[0].content?.parts || [];
    return parts
        .map(part => part.text || part.citationMetadata?.citation || '')
        .join('');
}

function stripCodeFences(text = '') {
    return (text || '')
        .replace(/```\s*srt/gi, '')
        .replace(/```/g, '')
        .replace(/\r\n/g, '\n');
}

async function writeBufferToTempFile(buffer, originalName = 'upload') {
    const extension = path.extname(originalName) || '.bin';
    const tempPath = path.join(os.tmpdir(), `${cryptoSafeName()}${extension}`);
    await fsPromises.writeFile(tempPath, buffer);
    return tempPath;
}

async function transcodeToWav(inputPath) {
    const outputPath = path.join(os.tmpdir(), `${cryptoSafeName()}.wav`);
    await new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .inputOptions(['-y'])
            .audioChannels(1)
            .audioFrequency(16000)
            .format('wav')
            .on('end', resolve)
            .on('error', reject)
            .save(outputPath);
    });
    return outputPath;
}

async function deleteIfExists(filePath) {
    try {
        await fsPromises.unlink(filePath);
    } catch (error) {
        if (error.code !== 'ENOENT') {
            console.warn('[cleanup] unable to delete', filePath, error.message);
        }
    }
}

function cryptoSafeName() {
    return typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2);
}

app.listen(PORT, () => {
    console.log(`AI SRT backend running on http://localhost:${PORT}`);
});

export default app;
