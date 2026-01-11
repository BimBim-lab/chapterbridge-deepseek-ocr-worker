"""
DeepSeek-OCR engine wrapper with adaptive tiling for tall/webtoon images.
"""
import os
import logging
import time
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

DEBUG_MODE = os.environ.get("OCR_DEBUG", "0") == "1"
DEBUG_DIR = os.environ.get("OCR_DEBUG_DIR", "debug")

DEEPSEEK_PROMPT = """You are an OCR engine. Transcribe the EXACT visible English text from the image.
Rules:
- Output ONLY the transcription, do NOT translate or paraphrase.
- Keep original punctuation and capitalization.
- If unclear, write [UNK].
- Do NOT add any extra words not clearly visible.
- Preserve line breaks when text is visually separated.
Return JSON ONLY in schema: {"lines":[{"text":"..."}]}"""


def get_deepseek_model():
    """
    Get or create singleton DeepSeek-OCR model instance.
    Uses transformers for inference.
    """
    global _model, _tokenizer
    
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        model_path = os.environ.get("DEEPSEEK_MODEL_PATH", 
                                    os.environ.get("DEEPSEEK_CHECKPOINT", 
                                                   "deepseek-ai/DeepSeek-OCR"))
        
        logger.info(f"Loading DeepSeek-OCR model from: {model_path}")
        
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            _model = _model.eval().cuda().to(torch.bfloat16)
            logger.info("DeepSeek-OCR loaded on CUDA with bfloat16")
        else:
            _model = _model.eval()
            logger.info("DeepSeek-OCR loaded on CPU")
        
        logger.info("DeepSeek-OCR model initialized successfully")
    
    return _model, _tokenizer


def parse_ocr_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse DeepSeek-OCR response into structured lines.
    
    Args:
        response_text: Raw model output
        
    Returns:
        List of line dicts with text, confidence=None, bbox=None
    """
    lines = []
    
    json_match = re.search(r'\{[\s\S]*"lines"[\s\S]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "lines" in data and isinstance(data["lines"], list):
                for item in data["lines"]:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"].strip()
                        if text:
                            lines.append({
                                "text": text,
                                "confidence": None,
                                "bbox": None
                            })
                return lines
        except json.JSONDecodeError:
            pass
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('{') and not line.startswith('"lines"'):
            lines.append({
                "text": line,
                "confidence": None,
                "bbox": None
            })
    
    return lines


def run_ocr_on_tile(tile_image: Image.Image, tile_idx: int = 0) -> List[Dict[str, Any]]:
    """
    Run DeepSeek-OCR on a single tile image.
    
    Args:
        tile_image: PIL Image tile
        tile_idx: Index for debugging
        
    Returns:
        List of line dicts
    """
    import torch
    import tempfile
    import os as os_module
    
    model, tokenizer = get_deepseek_model()
    
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_NEW_TOKENS", "800"))
    temperature = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0"))
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tile_image.save(tmp_file.name, 'PNG')
        temp_path = tmp_file.name
    
    try:
        prompt = f"<image>\n{DEEPSEEK_PROMPT}"
        
        if hasattr(model, 'infer'):
            result = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=temp_path,
                base_size=1024,
                image_size=640,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )
            response_text = result if isinstance(result, str) else str(result)
        else:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                os.environ.get("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR"),
                trust_remote_code=True
            )
            inputs = processor(images=tile_image, text=prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if DEBUG_MODE:
            debug_path = os.path.join(DEBUG_DIR, f"tile_{tile_idx}_response.txt")
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(debug_path, 'w') as f:
                f.write(response_text)
            logger.debug(f"Saved raw response to {debug_path}")
        
        lines = parse_ocr_response(response_text)
        
        if DEBUG_MODE:
            logger.info(f"Tile {tile_idx}: extracted {len(lines)} lines")
        
        return lines
        
    finally:
        if os_module.path.exists(temp_path):
            os_module.unlink(temp_path)


def tile_image(image: Image.Image, tile_height: int, overlap: int) -> List[Tuple[Image.Image, int, int]]:
    """
    Slice a tall image into overlapping horizontal tiles.
    
    Args:
        image: PIL Image to tile
        tile_height: Height of each tile in pixels
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of (tile_image, y_start, y_end) tuples
    """
    width, height = image.size
    
    if height <= tile_height:
        return [(image, 0, height)]
    
    tiles = []
    y_start = 0
    
    while y_start < height:
        y_end = min(y_start + tile_height, height)
        tile = image.crop((0, y_start, width, y_end))
        tiles.append((tile, y_start, y_end))
        
        if y_end >= height:
            break
            
        y_start = y_end - overlap
    
    logger.debug(f"Image {width}x{height} tiled into {len(tiles)} tiles "
                f"(tile_height={tile_height}, overlap={overlap})")
    return tiles


def normalize_text(text: str) -> str:
    """Normalize text for deduplication comparison."""
    return ' '.join(text.strip().upper().split())


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()


def deduplicate_lines(lines: List[Dict[str, Any]], similarity_threshold: float = 0.92) -> List[Dict[str, Any]]:
    """
    Remove duplicate lines from overlapping tiles.
    
    Uses normalized text comparison and similarity ratio to detect duplicates
    from adjacent tiles.
    
    Args:
        lines: List of OCR line results with tile_index
        similarity_threshold: Threshold for considering lines duplicate
        
    Returns:
        Deduplicated list of lines
    """
    if len(lines) <= 1:
        return lines
    
    sorted_lines = sorted(lines, key=lambda x: (x.get('tile_index', 0), x.get('line_index', 0)))
    
    keep = []
    seen_normalized = set()
    
    for i, line in enumerate(sorted_lines):
        text = line.get('text', '')
        normalized = normalize_text(text)
        
        if normalized in seen_normalized:
            continue
        
        is_duplicate = False
        for kept_line in keep[-10:]:
            if abs(line.get('tile_index', 0) - kept_line.get('tile_index', 0)) <= 1:
                if text_similarity(text, kept_line.get('text', '')) > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            keep.append(line)
            seen_normalized.add(normalized)
    
    logger.debug(f"Deduplication: {len(lines)} -> {len(keep)} lines")
    return keep


def choose_tiling_params(height: int) -> Tuple[Optional[int], Optional[int], str]:
    """
    Choose tiling parameters based on image height.
    
    Args:
        height: Image height in pixels
        
    Returns:
        (tile_height, overlap, strategy_name) or (None, None, "NO_TILE")
    """
    h1 = int(os.environ.get("DEEPSEEK_ADAPTIVE_H1", "3500"))
    h2 = int(os.environ.get("DEEPSEEK_ADAPTIVE_H2", "12000"))
    tile_height_med = int(os.environ.get("DEEPSEEK_TILE_HEIGHT_MED", 
                                          os.environ.get("DEEPSEEK_TILE_HEIGHT", "3000")))
    tile_height_long = int(os.environ.get("DEEPSEEK_TILE_HEIGHT_LONG", "2500"))
    overlap = int(os.environ.get("DEEPSEEK_TILE_OVERLAP", "350"))
    
    if height <= h1:
        return None, None, "NO_TILE"
    elif height <= h2:
        return tile_height_med, overlap, "TILE_MED"
    else:
        return tile_height_long, overlap, "TILE_LONG"


def run_ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Run OCR on image bytes (no tiling).
    
    Returns list of line objects:
    [
        {
            "text": "detected text",
            "confidence": null,
            "bbox": null
        },
        ...
    ]
    """
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    lines = run_ocr_on_tile(image, tile_idx=0)
    
    for i, line in enumerate(lines):
        line['line_index'] = i
        line['tile_index'] = 0
    
    return lines


def run_ocr_with_tiling(
    image_bytes: bytes, 
    tile_height: Optional[int] = None, 
    overlap: Optional[int] = None,
    debug_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run OCR with manual tiling parameters.
    
    Args:
        image_bytes: Image data
        tile_height: Override tile height (default from env)
        overlap: Override overlap (default from env)
        debug_dir: Optional debug output directory
        
    Returns:
        List of deduplicated OCR results
    """
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if tile_height is None:
        tile_height = int(os.environ.get("DEEPSEEK_TILE_HEIGHT", "3000"))
    if overlap is None:
        overlap = int(os.environ.get("DEEPSEEK_TILE_OVERLAP", "350"))
    
    width, height = image.size
    
    if debug_dir or DEBUG_MODE:
        actual_debug_dir = debug_dir or DEBUG_DIR
        os.makedirs(actual_debug_dir, exist_ok=True)
    
    tiles = tile_image(image, tile_height, overlap)
    
    logger.info(f"Image {width}x{height} split into {len(tiles)} tiles")
    
    all_lines = []
    
    for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles):
        if DEBUG_MODE:
            tile_path = os.path.join(DEBUG_DIR, f"tile_{tile_idx}.png")
            tile_img.save(tile_path)
        
        tile_lines = run_ocr_on_tile(tile_img, tile_idx)
        
        for i, line in enumerate(tile_lines):
            line['tile_index'] = tile_idx
            line['line_index'] = i
            line['y_offset'] = y_start
        
        all_lines.extend(tile_lines)
        
        logger.debug(f"Tile {tile_idx + 1}/{len(tiles)}: {len(tile_lines)} lines")
    
    deduplicated = deduplicate_lines(all_lines)
    
    for line in deduplicated:
        line.pop('tile_index', None)
        line.pop('line_index', None)
        line.pop('y_offset', None)
    
    return deduplicated


def run_ocr_adaptive(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Adaptive OCR pipeline that chooses optimal tiling strategy based on image height.
    
    Strategy:
    - height <= 3500: No tiling, direct OCR
    - 3501-12000: Medium tiles (3000px height)
    - > 12000: Smaller tiles (2500px height)
    
    Returns:
        List of OCR results, deduplicated and ordered
    """
    start_time = time.time()
    
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    
    tile_height, overlap, strategy = choose_tiling_params(height)
    
    logger.info(f"[ADAPTIVE] Image: {width}x{height} | Strategy: {strategy}")
    
    if strategy == "NO_TILE":
        lines = run_ocr_on_tile(image, tile_idx=0)
        
        for i, line in enumerate(lines):
            line['line_index'] = i
        
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] NO_TILE: {len(lines)} lines in {time.time() - start_time:.1f}s")
        
        return lines
    
    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    
    assert tile_height is not None and overlap is not None
    tiles = tile_image(image, tile_height, overlap)
    
    logger.info(f"[ADAPTIVE] Created {len(tiles)} tiles (height={tile_height}, overlap={overlap})")
    
    all_lines = []
    
    for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles):
        tile_start = time.time()
        
        if DEBUG_MODE:
            tile_path = os.path.join(DEBUG_DIR, f"adaptive_tile_{tile_idx}.png")
            tile_img.save(tile_path)
        
        tile_lines = run_ocr_on_tile(tile_img, tile_idx)
        
        for i, line in enumerate(tile_lines):
            line['tile_index'] = tile_idx
            line['line_index'] = i
            line['y_offset'] = y_start
        
        all_lines.extend(tile_lines)
        
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] Tile {tile_idx + 1}/{len(tiles)}: {len(tile_lines)} lines "
                       f"in {time.time() - tile_start:.1f}s")
    
    deduplicated = deduplicate_lines(all_lines)
    
    for line in deduplicated:
        line.pop('tile_index', None)
        line.pop('line_index', None)
        line.pop('y_offset', None)
    
    if DEBUG_MODE:
        logger.info(f"[ADAPTIVE] Total: {len(deduplicated)} lines in {time.time() - start_time:.1f}s")
        
        final_json_path = os.path.join(DEBUG_DIR, "final_output.json")
        with open(final_json_path, 'w') as f:
            json.dump({"lines": deduplicated}, f, indent=2)
    
    return deduplicated


def build_ocr_output(
    lines: List[Dict[str, Any]],
    work_id: Optional[str] = None,
    edition_id: Optional[str] = None,
    segment_id: Optional[str] = None,
    chapter: Optional[int] = None,
    page: Optional[int] = None,
    raw_r2_key: Optional[str] = None,
    raw_asset_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build the standard OCR output JSON format.
    
    Returns format compatible with existing worker output.
    """
    import datetime
    
    model_path = os.environ.get("DEEPSEEK_MODEL_PATH", 
                                os.environ.get("DEEPSEEK_CHECKPOINT", 
                                               "deepseek-ai/DeepSeek-OCR"))
    
    return {
        "version": "1.0",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "engine": {
            "name": "deepseek-ocr",
            "model": model_path,
            "mode": "tile",
            "prompt_version": "v1"
        },
        "metadata": {
            "work_id": work_id,
            "edition_id": edition_id,
            "segment_id": segment_id,
            "chapter": chapter,
            "page": page,
            "raw_r2_key": raw_r2_key,
            "raw_asset_id": raw_asset_id
        },
        "lines": [
            {
                "text": line.get("text", ""),
                "confidence": line.get("confidence"),
                "bbox": line.get("bbox")
            }
            for line in lines
        ],
        "stats": {
            "line_count": len(lines)
        }
    }
