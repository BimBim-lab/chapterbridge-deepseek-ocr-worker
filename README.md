# ChapterBridge OCR Worker

Python worker daemon for processing manhwa page images through DeepSeek-OCR, integrated with Supabase and Cloudflare R2.

## Architecture

This worker is part of the ChapterBridge pipeline. It:
1. Polls Supabase `pipeline_jobs` for queued OCR tasks
2. Downloads raw images from Cloudflare R2
3. Runs OCR using DeepSeek-OCR (vision language model)
4. Uploads JSON results back to R2
5. Updates Supabase with asset records and job status

## Setup

### Environment Variables

Set these in Replit Secrets or a `.env` file:

**Required:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

R2_ENDPOINT=https://accountid.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET=chapterbridge-data
```

**DeepSeek-OCR Configuration:**
```
DEEPSEEK_MODEL_PATH=deepseek-ai/DeepSeek-OCR    # Model path or checkpoint
DEEPSEEK_MAX_NEW_TOKENS=800                      # Max output tokens (default: 800)
DEEPSEEK_TEMPERATURE=0                           # Decoding temperature (default: 0)
```

**Tiling Configuration (for tall/webtoon images):**
```
DEEPSEEK_TILE_HEIGHT=3000          # Default tile height in pixels
DEEPSEEK_TILE_OVERLAP=350          # Overlap between tiles
DEEPSEEK_ADAPTIVE_H1=3500          # Height threshold for tiling
DEEPSEEK_ADAPTIVE_H2=12000         # Height threshold for smaller tiles
DEEPSEEK_TILE_HEIGHT_MED=3000      # Tile height for medium images (3501-12000px)
DEEPSEEK_TILE_HEIGHT_LONG=2500     # Tile height for long images (>12000px)
```

**Debug Mode:**
```
OCR_DEBUG=1                        # Enable debug mode (saves tiles and responses)
OCR_DEBUG_DIR=./debug              # Debug output directory
```

**Worker Configuration:**
```
POLL_SECONDS=3                     # Job polling interval
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### GPU vs CPU Performance

DeepSeek-OCR is a vision-language model that benefits significantly from GPU acceleration:

- **GPU (CUDA)**: Recommended for production. Model runs in bfloat16 on CUDA.
- **CPU**: Works but significantly slower. Suitable for testing only.

For GPU support, ensure CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Running the Worker Daemon

```bash
python workers/ocr/main.py --poll-seconds 3
```

The worker will:
- Initialize DeepSeek-OCR once at startup (may take 30-60s for model download)
- Continuously poll for queued jobs
- Process one job at a time
- Handle errors gracefully without crashing

### Testing on a Single Image

Use the test script to verify DeepSeek-OCR is working:

```bash
# Basic test
python workers/ocr/test_one.py path/to/image.png

# With output file
python workers/ocr/test_one.py image.png --output result.json

# Enable debug mode (saves tiles)
python workers/ocr/test_one.py image.png --debug

# Disable tiling (for short images)
python workers/ocr/test_one.py image.png --no-tile
```

### Creating OCR Jobs

Use the enqueue script to create jobs for raw images:

```bash
# Process all raw images for a specific edition
python workers/ocr/enqueue.py --edition-id <uuid> --limit 500

# Process raw images matching a key prefix
python workers/ocr/enqueue.py --prefix raw/manhwa/work123/ed456 --limit 100

# Force re-processing even if output exists
python workers/ocr/enqueue.py --edition-id <uuid> --force
```

## Job Contract

OCR jobs use `job_type='clean'` with specific input format:

```json
{
  "task": "ocr_page",
  "raw_asset_id": "uuid-of-raw-image-asset",
  "force": false
}
```

## R2 Key Conventions

**Input (raw images):**
```
raw/manhwa/{work_id}/{edition_id}/chapter-0236/page-001.jpg
```

**Output (OCR JSON):**
```
derived/manhwa/{work_id}/{edition_id}/chapter-0236/ocr/page-001.json
```

## OCR Output Format

```json
{
  "version": "1.0",
  "timestamp": "2025-01-11T12:00:00Z",
  "engine": {
    "name": "deepseek-ocr",
    "model": "deepseek-ai/DeepSeek-OCR",
    "mode": "tile",
    "prompt_version": "v1"
  },
  "metadata": {
    "work_id": "...",
    "edition_id": "...",
    "segment_id": "...",
    "chapter": 236,
    "page": 1,
    "raw_r2_key": "raw/manhwa/..."
  },
  "stats": {
    "line_count": 15
  },
  "lines": [
    {
      "text": "Detected text",
      "confidence": null,
      "bbox": null
    }
  ]
}
```

**Note:** DeepSeek-OCR does not provide confidence scores or bounding boxes. These fields are `null` in the output.

## Tiling for Tall Images

DeepSeek-OCR uses adaptive tiling for tall/webtoon images:

| Image Height | Strategy | Tile Height |
|--------------|----------|-------------|
| ≤ 3500px | No tiling | - |
| 3501-12000px | Medium tiling | 3000px |
| > 12000px | Long tiling | 2500px |

Overlap between tiles is 350px by default. Duplicate lines in overlap regions are automatically deduplicated using text similarity matching.

## Project Structure

```
workers/ocr/
  main.py              # Daemon loop poller
  test_one.py          # Single image test script
  enqueue.py           # Job creator script
  supabase_client.py   # Database operations
  r2_client.py         # R2 storage client
  ocr_engine.py        # DeepSeek-OCR wrapper with tiling
  key_parser.py        # R2 key parsing utilities
  utils.py             # Logging, hashing utilities
```

## Troubleshooting

### Model Loading Issues
- Ensure you have enough disk space for the model (~2-4GB)
- Check internet connection for HuggingFace model download
- Verify `DEEPSEEK_MODEL_PATH` is correct

### Out of Memory
- Reduce `DEEPSEEK_TILE_HEIGHT` for smaller tiles
- Ensure no other GPU processes are running
- Consider using CPU mode for testing

### Poor OCR Results
- Enable debug mode to inspect tiles: `OCR_DEBUG=1`
- Check tile images in `debug/` directory
- Adjust tiling parameters for your image types
