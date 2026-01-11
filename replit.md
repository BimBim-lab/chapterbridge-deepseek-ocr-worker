# ChapterBridge OCR Worker

## Overview
Python OCR worker daemon for the ChapterBridge pipeline. Processes manhwa page images using DeepSeek-OCR (vision language model) and stores results in Cloudflare R2.

## Project Structure
```
workers/ocr/
  main.py              - Main daemon loop
  test_one.py          - Single image test script
  enqueue.py           - Job creation script
  supabase_client.py   - Supabase database client
  r2_client.py         - Cloudflare R2 client
  ocr_engine.py        - DeepSeek-OCR wrapper with tiling
  key_parser.py        - R2 key parsing utilities
  utils.py             - Logging and hashing helpers
```

## Required Secrets
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key
- `R2_ENDPOINT` - Cloudflare R2 endpoint
- `R2_ACCESS_KEY_ID` - R2 access key
- `R2_SECRET_ACCESS_KEY` - R2 secret key
- `R2_BUCKET` - R2 bucket name (default: chapterbridge-data)

## DeepSeek-OCR Configuration
- `DEEPSEEK_MODEL_PATH` - Model path (default: deepseek-ai/DeepSeek-OCR)
- `DEEPSEEK_MAX_NEW_TOKENS` - Max output tokens (default: 800)
- `DEEPSEEK_TEMPERATURE` - Decoding temperature (default: 0)

## Tiling Configuration (for tall images)
- `DEEPSEEK_TILE_HEIGHT` - Default tile height (default: 3000)
- `DEEPSEEK_TILE_OVERLAP` - Overlap between tiles (default: 350)
- `DEEPSEEK_ADAPTIVE_H1` - Height threshold for tiling (default: 3500)
- `DEEPSEEK_ADAPTIVE_H2` - Height threshold for smaller tiles (default: 12000)

## Optional Environment Variables
- `OCR_DEBUG` - Enable debug mode (0/1)
- `OCR_DEBUG_DIR` - Debug output directory (default: ./debug)
- `POLL_SECONDS` - Job polling interval (default: 3)

## Database
Uses external Supabase with tables: pipeline_jobs, assets, segment_assets, segments, editions

## Job Contract
OCR jobs use job_type='clean' with input.task='ocr_page'

## OCR Output Schema
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
    "work_id": "uuid",
    "edition_id": "uuid", 
    "segment_id": "uuid",
    "chapter": 123,
    "page": 1,
    "raw_r2_key": "raw/manhwa/..."
  },
  "stats": { "line_count": 15 },
  "lines": [{ "text": "...", "confidence": null, "bbox": null }]
}
```

## Recent Changes
- January 2026: Replaced PaddleOCR with DeepSeek-OCR (vision language model)
- January 2026: Updated tiling system for DeepSeek compatibility
- January 2026: Added test_one.py script for single image testing
- January 2026: Updated output schema with deepseek-ocr engine metadata
