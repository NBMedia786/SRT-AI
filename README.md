# SRT-AI - AI-Powered Subtitle Generator

🎬 Generate professional SRT subtitles from any audio or video file using Google's Gemini AI.

![SRT-AI Interface](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- **AI-Powered Transcription** - Uses Google Gemini for accurate speech-to-text
- **Multiple Format Support** - Works with MP4, MP3, WAV, MOV, and more
- **Customizable Output** - Adjust words per line (3-20 words)
- **Real-time Preview** - Edit and preview subtitles before downloading
- **History Tracking** - Access previous transcriptions locally
- **Production Ready** - Docker support, rate limiting, security features

## Quick Start

### Prerequisites

- [Node.js 18+](https://nodejs.org/)
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone or download the project**

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment**
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env and add your Gemini API key
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Start the server**
   ```bash
   npm start
   ```

5. **Open your browser**
   ```
   http://localhost:3000
   ```

## Configuration

Create a `.env` file in the project root:

```env
# Required: Your Google Gemini API Key
GEMINI_API_KEY=your_api_key_here

# Optional: Server port (default: 3000)
PORT=3000

# Optional: Gemini model (default: gemini-1.5-pro-latest)
GEMINI_MODEL=gemini-1.5-pro-latest

# Optional: Environment (development/production)
NODE_ENV=production
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Set your API key**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

2. **Start the container**
   ```bash
   docker-compose up -d
   ```

3. **Access the app**
   ```
   http://localhost:3000
   ```

### Manual Docker Build

```bash
# Build the image
docker build -t srt-ai .

# Run the container
docker run -d -p 3000:3000 \
  -e GEMINI_API_KEY=your_api_key \
  srt-ai
```

## API Endpoints

### Health Check
```http
GET /api/health
```
Returns server status and configuration.

### Transcribe
```http
POST /api/transcribe
Content-Type: multipart/form-data

file: <audio/video file>
wordLimit: <number 3-20>
```
Returns SRT content as JSON: `{ "srt": "..." }`

## Project Structure

```
SRT-AI/
├── server.js          # Express backend server
├── index.html         # Frontend SPA
├── package.json       # Dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose config
├── .env.example       # Environment template
└── .gitignore         # Git ignore rules
```

## Development

### Running in Development Mode
```bash
npm run dev
```
This uses Node.js watch mode for auto-restart on file changes.

### Available Scripts
- `npm start` - Start production server
- `npm run dev` - Start with auto-reload
- `npm run docker:build` - Build Docker image
- `npm run docker:run` - Start Docker container
- `npm run docker:stop` - Stop Docker container
- `npm run docker:logs` - View Docker logs

## Troubleshooting

### "GEMINI_API_KEY is not set"
Make sure you have created a `.env` file with your API key.

### "FFmpeg not found"
The application uses `ffmpeg-static` which includes FFmpeg binaries. If issues persist, install FFmpeg system-wide.

### "Too many requests"
The API has rate limiting (10 requests/minute). Wait a minute and try again.

### "File too large"
Maximum file size is 500MB. For larger files, consider splitting them.

## Security Features

- **Rate Limiting** - 10 requests per minute per IP
- **File Validation** - Only audio/video files accepted
- **Graceful Shutdown** - Clean server termination
- **Error Handling** - User-friendly error messages
- **Docker Security** - Non-root user in container

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Credits

- [Google Gemini AI](https://ai.google.dev/)
- [FFmpeg](https://ffmpeg.org/)
- [Express.js](https://expressjs.com/)
- [Tailwind CSS](https://tailwindcss.com/)

