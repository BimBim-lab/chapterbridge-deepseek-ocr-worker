#!/usr/bin/env python3
"""
Test DeepSeek-OCR on a single image file.

Usage:
    python workers/ocr/test_one.py <image_path> [--output <output.json>]
    
Examples:
    python workers/ocr/test_one.py sample.png
    python workers/ocr/test_one.py sample.png --output result.json
    python workers/ocr/test_one.py sample.png --debug
"""
import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-OCR on a single image")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves tiles)")
    parser.add_argument("--no-tile", action="store_true", help="Disable tiling (use direct OCR)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    if args.debug:
        os.environ["OCR_DEBUG"] = "1"
        os.environ["OCR_DEBUG_DIR"] = "debug"
    
    from workers.ocr.ocr_engine import (
        run_ocr, 
        run_ocr_adaptive, 
        build_ocr_output,
        get_deepseek_model
    )
    
    print(f"Loading DeepSeek-OCR model...")
    start_load = time.time()
    get_deepseek_model()
    print(f"Model loaded in {time.time() - start_load:.1f}s")
    
    print(f"\nProcessing: {args.image_path}")
    
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()
    
    print(f"Image size: {len(image_bytes)} bytes")
    
    start_ocr = time.time()
    
    if args.no_tile:
        print("Running OCR (no tiling)...")
        lines = run_ocr(image_bytes)
    else:
        print("Running adaptive OCR...")
        lines = run_ocr_adaptive(image_bytes)
    
    ocr_time = time.time() - start_ocr
    print(f"OCR completed in {ocr_time:.1f}s")
    print(f"Detected {len(lines)} text lines")
    
    output = build_ocr_output(
        lines=lines,
        raw_r2_key=args.image_path
    )
    
    output_path = args.output or args.image_path.rsplit('.', 1)[0] + "_ocr.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_path}")
    
    print("\n--- Extracted Text ---")
    for i, line in enumerate(lines[:20], 1):
        text = line.get("text", "")
        print(f"{i:3}. {text}")
    
    if len(lines) > 20:
        print(f"... and {len(lines) - 20} more lines")
    
    print("\n--- Summary ---")
    print(f"Engine: {output['engine']['name']} ({output['engine']['model']})")
    print(f"Lines: {output['stats']['line_count']}")
    print(f"Time: {ocr_time:.1f}s")


if __name__ == "__main__":
    main()
