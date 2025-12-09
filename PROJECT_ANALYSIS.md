# SRT-AI Project - Deep Analysis & Flow Documentation

## 📋 Project Overview

**SRT-AI** is an AI-powered web application that automatically generates professional SRT (SubRip Subtitle) files from video or audio files using Google's Gemini AI. The application provides a user-friendly interface for uploading media, customizing subtitle generation parameters, and downloading the resulting subtitle files.

### Key Capabilities
- **Automatic Transcription**: Converts video/audio to text using Google Gemini AI
- **Intelligent Subtitle Generation**: Creates properly timed SRT files with customizable word limits
- **Media Format Support**: Handles various video (MP4, MOV) and audio (MP3, WAV) formats
- **Real-time Preview**: Allows users to preview and adjust subtitles before downloading
- **Custom Instructions**: Supports user-provided context for better transcription accuracy

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- **Node.js** (ES Modules)
- **Express.js** - Web server framework
- **Multer** - File upload handling (memory storage)
- **FFmpeg** (via fluent-ffmpeg) - Media transcoding
- **Google Generative AI SDK** - AI transcription service
- **CORS** - Cross-origin resource sharing

**Frontend:**
- **Vanilla JavaScript** - No framework dependencies
- **Tailwind CSS** (CDN) - Styling
- **HTML5** - Structure

**External Services:**
- **Google Gemini API** - AI transcription and subtitle generation
- **Google AI File Manager** - File upload/management for Gemini

### Project Structure
```
SRT-AI/
├── server.js          # Express backend server
├── index.html         # Frontend application (single-page)
├── package.json       # Dependencies and scripts
├── config/            # Configuration directory (empty)
└── node_modules/      # Dependencies
```

---

## 🔄 Detailed Application Flow

### Phase 1: Initialization & Setup

#### 1.1 Server Startup (`server.js`)
```
1. Load environment variables (dotenv)
   - PORT (default: 3006)
   - GEMINI_API_KEY (required)
   - GEMINI_MODEL (default: 'gemini-2.5-pro')
   - STATIC_ROOT (optional, defaults to __dirname)

2. Configure FFmpeg
   - Set FFmpeg binary path from ffmpeg-static package
   - Warn if FFmpeg not available

3. Initialize Express app
   - Enable CORS
   - Configure static file serving
   - Set up Multer for file uploads (750MB limit, memory storage)

4. Initialize Google AI services
   - GoogleGenerativeAI instance
   - GoogleAIFileManager instance

5. Start server on configured PORT
```

#### 1.2 Frontend Load (`index.html`)
```
1. Load Tailwind CSS from CDN
2. Initialize DOM elements and event listeners
3. Set up state variables:
   - selectedFile
   - originalSrtContent
   - parsedSrtBlocks
   - currentWordLimit (default: 6)
   - logEntries
4. Determine API base URL (from window.SRT_API_BASE or current origin)
5. Display upload view
```

---

### Phase 2: File Selection & Configuration

#### 2.1 User Interaction - File Selection
```
User can select file via:
├── Drag & Drop (dropZone element)
├── Browse Button (file input click)
└── Direct click on drop zone

Event Flow:
1. File selected → fileInput.change event
2. prepareFileForAnalysis(file) called
```

#### 2.2 File Preparation (`prepareFileForAnalysis`)
```
1. Store selected file in state
2. Extract original filename (without extension)
3. Update UI:
   - Display filename in selected-file-name label
   - Show post-upload-controls section
   - Enable analyze-button
   - Set word limit slider to current value
4. Add log entry: "Ready to analyze..."
```

#### 2.3 Configuration Options
```
User can configure:
├── Custom Instructions (prompt-input textarea)
│   └── Optional context for AI (e.g., "technical presentation", "formal language")
│
└── Words Per Line (pre-words-per-line-slider)
    ├── Range: 3-20 words
    ├── Default: 6 words
    └── Real-time value display
```

---

### Phase 3: Transcription Request

#### 3.1 User Initiates Analysis
```
User clicks "Analyze" button:
1. analyze-button.click event
2. handleFileUpload(selectedFile, wordLimit) called
```

#### 3.2 Frontend Processing (`handleFileUpload`)
```
1. Disable analyze button
2. Reset logs
3. Switch to processing view:
   - Hide upload-view
   - Hide result-view
   - Show processing-view (spinner animation)

4. Prepare request:
   - Validate word limit (3-20 range)
   - Get user instructions from prompt-input
   - Generate structured prompt (generateStructuredPrompt)
   - Log: "Preparing transcription request..."

5. Call API:
   - callTranscriptionApi(file, structuredPrompt, wordLimit, addLog)
```

#### 3.3 API Request (`callTranscriptionApi`)
```
1. Create FormData:
   - file: selected media file
   - prompt: structured prompt with instructions
   - wordLimit: number (3-20)

2. Send POST request to: ${apiBase}/api/transcribe
   - Log: "Contacting transcription service..."
   - Log: "Upload complete. Awaiting Gemini response..."

3. Handle response:
   - Parse JSON response
   - Extract SRT content
   - Log: "Gemini transcription parsed successfully."
   - Return SRT text
```

---

### Phase 4: Backend Processing

#### 4.1 API Endpoint (`/api/transcribe`)
```
POST /api/transcribe
Middleware: upload.single('file')
```

#### 4.2 Request Validation
```
1. Check file exists → 400 if missing
2. Check GEMINI_API_KEY → 500 if missing
3. Extract request parameters:
   - userPrompt: req.body.prompt
   - wordLimit: req.body.wordLimit (default: 6)
   - combinedPrompt: userPrompt || buildPrompt('', wordLimit)
```

#### 4.3 File Processing Pipeline
```
Step 1: Write Uploaded File to Temp Storage
├── writeBufferToTempFile(req.file.buffer, originalname)
│   └── Creates temp file in os.tmpdir() with random UUID name
│   └── Returns: tempInputPath
│
Step 2: Media Format Detection & Transcoding
├── Check file type:
│   ├── isVideo: mimetype starts with 'video'
│   └── isWav: mimetype === 'audio/wav' OR extension === '.wav'
│
└── If (isVideo OR !isWav):
    └── transcodeToWav(tempInputPath)
        ├── FFmpeg conversion:
        │   ├── Input: tempInputPath
        │   ├── Output: temp file with .wav extension
        │   ├── Audio: 1 channel, 16kHz sample rate
        │   └── Format: WAV
        └── Returns: wavPath

Step 3: Upload to Google Gemini
├── uploadAudioToGemini(audioPath, displayName)
│   ├── Read audio file buffer
│   ├── Upload via fileManager.uploadFile()
│   │   ├── mimeType: 'audio/wav'
│   │   └── displayName: original filename
│   └── waitForFileActivation(uploadResponse.file)
│       ├── Poll file status (max 20 attempts, 1.5s delay)
│       ├── States: PENDING → ACTIVE → (use)
│       └── Error if state = FAILED or timeout
│
└── Returns: geminiFile (with name, uri, mimeType)
```

#### 4.4 AI Transcription
```
1. Get Generative Model:
   └── genAI.getGenerativeModel({ model: GEMINI_MODEL })

2. Generate Content:
   └── model.generateContent([
       { text: combinedPrompt },
       {
           fileData: {
               mimeType: geminiFile.mimeType,
               fileUri: geminiFile.uri
           }
       }
   ])

3. Extract Response:
   ├── extractText(result)
   │   └── Gets text from result.response.candidates[0].content.parts
   │
   └── stripCodeFences(responseText)
       └── Removes ```srt and ``` markers, normalizes line endings

4. Validate:
   └── Throw error if response is empty

5. Return JSON:
   └── res.json({ srt: cleanedResponse })
```

#### 4.5 Cleanup (finally block)
```
1. Delete temporary files:
   └── Promise.allSettled([
       ...cleanupPaths.map(deleteIfExists),
       deleteGeminiFile(uploadedGeminiFileName)
   ])

2. Cleanup includes:
   ├── Original uploaded temp file
   ├── Transcoded WAV file (if created)
   └── Gemini uploaded file (via fileManager.deleteFile)
```

---

### Phase 5: Response Processing & Display

#### 5.1 Frontend Receives Response
```
1. callTranscriptionApi returns SRT text
2. Process response:
   ├── stripCodeFences() - Remove markdown code blocks
   └── normalizeSrtToSingleLine() - Ensure single line per subtitle

3. Store in state:
   └── originalSrtContent = processed SRT

4. Log: "Gemini returned transcription data."
```

#### 5.2 Display Results (`showResultView`)
```
1. Parse SRT:
   ├── parseSRT(originalSrtContent)
   │   └── Converts SRT string to array of subtitle blocks
   │   └── Each block: { index, startMs, endMs, text }
   │
   └── Store: parsedSrtBlocks

2. Update UI:
   ├── rawPreviewArea.value = originalSrtContent
   ├── If parsing successful:
   │   ├── Enable word limit slider
   │   ├── updatePreview(defaultWords) - Format with current word limit
   │   └── Log: "Successfully parsed X subtitle blocks"
   │
   └── If parsing failed:
       ├── Disable slider
       └── Log: "Parsing produced no caption blocks"

3. Switch views:
   ├── Hide processing-view
   ├── Hide upload-view
   └── Show result-view

4. Log: "Result view displayed. Ready for review and download."
```

#### 5.3 SRT Parsing (`parseSRT`)
```
Algorithm:
1. Normalize input (remove code fences, normalize line endings)
2. Split into lines
3. Iterate through lines:
   ├── Skip empty lines
   ├── Detect subtitle block:
   │   ├── Line 1: Index number (optional)
   │   ├── Line 2: Timestamp (HH:MM:SS,mmm --> HH:MM:SS,mmm)
   │   └── Lines 3+: Text content
   │
   ├── Parse timestamp:
   │   └── Convert to milliseconds (timecodeToMilliseconds)
   │
   └── Extract text (join multiple lines with spaces)

4. Return array of blocks: [{ index, startMs, endMs, text }, ...]
```

#### 5.4 Preview Formatting (`updatePreview`)
```
Triggered by:
├── Initial display (showResultView)
└── Word limit slider change (real-time)

Process:
1. Check if parsedSrtBlocks exists
2. Format SRT:
   └── formatSRT(parsedSrtBlocks, wordsPerLine)
       ├── resegmentSrtBlocks(parsedSrtBlocks, wordsPerLine)
       │   ├── For each subtitle block:
       │   │   ├── Split text into words
       │   │   ├── Calculate duration per word
       │   │   ├── Chunk words by preferred limit (chunkWords)
       │   │   ├── Allocate time proportionally
       │   │   └── Create new blocks with single-line text
       │   │
       │   └── compactOverlaps() - Ensure no timestamp overlaps
       │
       └── Convert blocks to SRT format string

3. Update previewArea.value (if exists)
4. Update slider value display
```

---

### Phase 6: User Interaction & Download

#### 6.1 Real-time Adjustment
```
Word Limit Slider (pre-words-per-line-slider):
├── Range: 3-20
├── On input change:
│   ├── Update currentWordLimit
│   ├── Update display value
│   ├── If in result view:
│   │   ├── Log: "Adjusting preview to X words per line..."
│   │   └── updatePreview(words) - Reformat immediately
│   │
│   └── No API call needed (client-side reformatting)
```

#### 6.2 Download SRT File
```
User clicks "Download .srt File":
├── downloadSRT()
│   ├── Get SRT content:
│   │   ├── From previewArea.value (if exists)
│   │   └── OR from rawPreviewArea.value
│   │
│   ├── Create Blob:
│   │   └── new Blob([srtContent], { type: 'text/plain;charset=utf-8' })
│   │
│   ├── Create download link:
│   │   ├── URL.createObjectURL(blob)
│   │   ├── Set download attribute: `${originalFilename}.srt`
│   │   ├── Trigger click
│   │   └── Cleanup: URL.revokeObjectURL()
│   │
│   └── File downloads to user's device
```

#### 6.3 Start Over
```
User clicks "Start Over":
├── showUploadView()
│   ├── Reset all state:
│   │   ├── selectedFile = null
│   │   ├── originalSrtContent = ''
│   │   ├── parsedSrtBlocks = []
│   │   ├── currentWordLimit = 6
│   │   └── Clear all UI inputs
│   │
│   ├── Switch views:
│   │   ├── Hide processing-view
│   │   ├── Hide result-view
│   │   └── Show upload-view
│   │
│   └── Reset logs
```

---

## 🔧 Key Functions & Algorithms

### Backend Functions

#### `buildPrompt(extraInstructions, wordLimit)`
```
Purpose: Generate structured prompt for Gemini AI

Structure:
├── Role definition: "AI assistant specialized in SRT generation"
├── Input description:
│   ├── Raw transcription text
│   ├── Word limit per line
│   ├── Punctuation-based line breaks
│   └── Intelligent length adjustment
│
├── Task description:
│   ├── Format into valid SRT
│   ├── Include sequence numbers
│   ├── Include timestamps
│   ├── Format text per word limit
│   └── Single line per subtitle entry
│
└── Additional user instructions (if provided)
```

#### `transcodeToWav(inputPath)`
```
Purpose: Convert any audio/video to WAV format for Gemini

Process:
1. Create output path in temp directory
2. Use FFmpeg:
   ├── Input: inputPath
   ├── Output: outputPath (.wav)
   ├── Audio channels: 1 (mono)
   ├── Sample rate: 16000 Hz
   └── Format: WAV
3. Return output path
```

#### `waitForFileActivation(fileMetadata)`
```
Purpose: Wait for Gemini to process uploaded file

Algorithm:
├── Max attempts: 20
├── Delay between attempts: 1500ms
├── Poll file status:
│   ├── If state === 'ACTIVE' → return file
│   ├── If state === 'FAILED' → throw error
│   └── If state === 'PENDING' → continue polling
│
└── If timeout → throw error
```

### Frontend Functions

#### `resegmentSrtBlocks(srtBlocks, wordsPerLine)`
```
Purpose: Re-segment subtitle blocks based on word limit

Algorithm:
For each subtitle block:
├── Split text into words
├── Calculate total duration
├── Calculate duration per word
├── Chunk words by preferred limit:
│   └── chunkWords(words, preferredWords)
│       └── Split into arrays of maxWords size
│
├── For each chunk:
│   ├── Calculate allocated duration (proportional)
│   ├── Set start/end timestamps
│   └── Create single-line caption
│
└── compactOverlaps() - Ensure no overlapping timestamps

Returns: Array of { startMs, endMs, lines: [singleLine] }
```

#### `chunkWords(words, maxWords)`
```
Purpose: Split word array into chunks

Algorithm:
├── Calculate chunk size: Math.max(1, Math.floor(maxWords))
├── Iterate through words:
│   └── Slice words[i : i + size]
│
└── Return: Array of word arrays
```

#### `compactOverlaps(blocks)`
```
Purpose: Ensure strictly increasing timestamps

Algorithm:
├── Sort blocks by startMs
├── Track previousEnd
├── For each block:
│   ├── startMs = max(block.startMs, previousEnd + 1)
│   ├── endMs = max(block.endMs, startMs + 200)
│   └── Update previousEnd = endMs
│
└── Return adjusted blocks
```

---

## 📊 Data Flow Diagram

```
┌─────────────┐
│   Browser   │
│  (Frontend) │
└──────┬──────┘
       │
       │ 1. User selects file
       │ 2. Configure options (word limit, instructions)
       │ 3. Click "Analyze"
       │
       ▼
┌─────────────────────────────────────┐
│  POST /api/transcribe               │
│  FormData:                          │
│  - file (binary)                    │
│  - prompt (string)                  │
│  - wordLimit (number)               │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Express Server                     │
│  ├── Multer: Store in memory        │
│  ├── Validate file & API key        │
│  └── Extract parameters             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  File Processing                    │
│  ├── Write to temp file             │
│  ├── Check format (video/audio)     │
│  └── Transcode to WAV (if needed)   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Google Gemini File Manager         │
│  ├── Upload WAV file                 │
│  └── Wait for activation            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Google Gemini AI                    │
│  ├── Generate transcription          │
│  ├── Generate SRT format             │
│  └── Return SRT text                 │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Response Processing                 │
│  ├── Extract text from response      │
│  ├── Strip code fences               │
│  └── Clean up temp files             │
└──────┬──────────────────────────────┘
       │
       │ JSON: { srt: "..." }
       │
       ▼
┌─────────────┐
│   Browser   │
│  (Frontend) │
│  ├── Parse SRT                       │
│  ├── Display preview                 │
│  └── Allow download                  │
└─────────────┘
```

---

## 🎯 Key Features & Behaviors

### 1. Intelligent Subtitle Generation
- **Word Limit Control**: Users can set preferred words per line (3-20)
- **Punctuation Awareness**: AI instructed to break at punctuation marks
- **Natural Timing**: Timestamps based on actual speech patterns
- **Single-Line Format**: Each subtitle entry contains exactly one line

### 2. Media Format Support
- **Video Formats**: MP4, MOV, and other video formats (transcoded to WAV)
- **Audio Formats**: MP3, WAV, and other audio formats
- **Automatic Transcoding**: Non-WAV files automatically converted
- **File Size Limit**: 750MB maximum

### 3. User Experience
- **Drag & Drop**: Intuitive file selection
- **Real-time Preview**: See subtitles before downloading
- **Live Logs**: Processing status updates
- **Error Handling**: Graceful fallbacks and error messages
- **Responsive Design**: Works on desktop and mobile

### 4. Client-Side Intelligence
- **No Re-transcription**: Word limit changes don't require new API calls
- **Instant Updates**: Slider changes update preview immediately
- **SRT Parsing**: Robust parsing handles various SRT formats
- **Timestamp Preservation**: Original timing maintained during resegmentation

---

## 🔐 Configuration & Environment

### Required Environment Variables
```env
GEMINI_API_KEY=your_api_key_here        # Required
PORT=3006                                # Optional (default: 3006)
GEMINI_MODEL=gemini-2.5-pro             # Optional (default: gemini-2.5-pro)
STATIC_ROOT=./public                     # Optional (default: __dirname)
```

### API Configuration
- **Base URL**: Determined from `window.SRT_API_BASE` or current origin
- **Fallback**: `http://localhost:8787` if origin not available
- **Endpoint**: `/api/transcribe` (POST)

---

## 🐛 Error Handling

### Backend Errors
1. **Missing File**: 400 Bad Request
2. **Missing API Key**: 500 Internal Server Error
3. **Transcoding Failure**: 500 with error message
4. **Gemini Upload Failure**: 500 with error details
5. **Empty Response**: 500 with error message
6. **File Cleanup**: Errors logged but don't fail request

### Frontend Errors
1. **API Errors**: Displayed in logs, fallback to demo data
2. **Parsing Errors**: Raw SRT displayed if parsing fails
3. **File Selection Errors**: Validation before upload
4. **Network Errors**: Caught and logged

---

## 📝 SRT Format Specification

The application generates standard SRT format:
```
1
00:00:01,500 --> 00:00:04,200
Subtitle text here

2
00:00:04,800 --> 00:00:08,500
Next subtitle entry
```

**Format Rules:**
- Sequence number (incrementing)
- Timestamp: `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- Text content (single line per entry)
- Blank line between entries

---

## 🚀 Performance Considerations

1. **Memory Storage**: Files stored in memory (Multer) for fast processing
2. **Temp File Cleanup**: Automatic cleanup after processing
3. **Gemini File Cleanup**: Uploaded files deleted after use
4. **Client-Side Formatting**: Word limit changes don't hit server
5. **Polling Strategy**: Efficient file activation polling (1.5s intervals)

---

## 🔄 State Management

### Frontend State Variables
```javascript
selectedFile          // Currently selected media file
originalSrtContent    // Raw SRT from Gemini
parsedSrtBlocks       // Parsed subtitle blocks array
currentWordLimit      // Current words per line setting
logEntries            // Array of log messages
originalFilename      // Base filename for downloads
```

### View States
- **upload-view**: Initial file selection and configuration
- **processing-view**: Transcription in progress
- **result-view**: Display results and allow download

---

## 📚 Dependencies

### Production Dependencies
- `@google/generative-ai` (^0.24.1) - Gemini AI SDK
- `cors` (^2.8.5) - CORS middleware
- `dotenv` (^17.2.3) - Environment variables
- `express` (^5.1.0) - Web framework
- `ffmpeg-static` (^5.2.0) - FFmpeg binary
- `fluent-ffmpeg` (^2.1.3) - FFmpeg wrapper
- `multer` (^2.0.2) - File upload handling

### Development
- No dev dependencies specified

---

## 🎨 UI/UX Features

1. **Modern Design**: Tailwind CSS with gradient accents
2. **Step Indicator**: Visual progress (Customize → Upload → Download)
3. **Drag & Drop Zone**: Highlighted on drag-over
4. **Processing Animation**: Spinning loader with pulse effect
5. **Live Logs**: Real-time status updates in terminal-style display
6. **Responsive Layout**: Mobile-friendly design
7. **Accessibility**: Semantic HTML and ARIA-friendly structure

---

## 🔍 Debugging Features

1. **Console Logging**: Raw Gemini response logged to server console
2. **Client Logs**: Processing steps logged in UI
3. **Error Messages**: Detailed error information
4. **Mock Data**: Fallback demo data for testing

---

This analysis provides a comprehensive understanding of the SRT-AI project's architecture, flow, and implementation details. The application demonstrates a well-structured approach to AI-powered media transcription with a focus on user experience and performance.

