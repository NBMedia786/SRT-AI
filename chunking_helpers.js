
// ═══════════════════════════════════════════════════════════
//   LONG AUDIO CHUNKING HELPERS
// ═══════════════════════════════════════════════════════════

/**
 * Split audio file into chunks of specified duration
 * @param {string} inputPath - Path to input file
 * @param {number} segmentTime - Segment time in seconds (default 600 = 10 mins)
 * @returns {Promise<string[]>} List of chunk file paths
 */
async function splitAudio(inputPath, segmentTime = 600) {
    const outputPattern = path.join(path.dirname(inputPath), `chunk_%03d${path.extname(inputPath)}`);

    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .outputOptions([
                '-f', 'segment',
                '-segment_time', segmentTime.toString(),
                '-c', 'copy',
                '-reset_timestamps', '1'
            ])
            .output(outputPattern)
            .on('end', async () => {
                // Find generated files
                const dir = path.dirname(inputPath);
                const files = await fsPromises.readdir(dir);
                const chunks = files
                    .filter(f => f.startsWith('chunk_') && f.endsWith(path.extname(inputPath)))
                    .map(f => path.join(dir, f))
                    .sort();
                resolve(chunks);
            })
            .on('error', (err) => reject(err))
            .run();
    });
}

/**
 * Adjust timestamps in SRT content by adding an offset
 */
function adjustChunkTimestamps(srtContent, timeOffsetMs, sequenceOffset) {
    if (!srtContent) return '';

    // Parse SRT blocks
    const blocks = srtContent.trim().split(/\n\s*\n/);

    return blocks.map(block => {
        const lines = block.split('\n');
        if (lines.length < 3) return block;

        // Adjust sequence number
        const seq = parseInt(lines[0]);
        if (!isNaN(seq)) {
            lines[0] = (seq + sequenceOffset).toString();
        }

        // Adjust timestamps
        // Format: 00:00:00,000 --> 00:00:00,000
        const timeLine = lines[1];
        const timeMatch = timeLine.match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);

        if (timeMatch) {
            const startMs = timeToMs(timeMatch[1]) + timeOffsetMs;
            const endMs = timeToMs(timeMatch[2]) + timeOffsetMs;
            lines[1] = `${msToTime(startMs)} --> ${msToTime(endMs)}`;
        }

        return lines.join('\n');
    }).join('\n\n');
}

/**
 * Count number of subtitles in an SRT string
 */
function countSubtitles(srtContent) {
    if (!srtContent) return 0;
    return (srtContent.match(/\n\s*\n/g) || []).length + 1;
}
