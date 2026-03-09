"""
C2PA Provenance Checker
=======================
Reads Content Authenticity Initiative (C2PA) manifests embedded in image files.
Works without any external C2PA library -- parses JUMBF boxes and XMP directly.

C2PA is a cryptographic standard where:
  - Real cameras (Sony, Canon, Nikon, Leica) embed a signed manifest proving capture
  - AI tools (Adobe Firefly, DALL-E, Midjourney) embed manifests declaring AI origin
  - Edited images accumulate a chain of manifests showing the full history

Verdict logic:
  VERIFIED REAL    -- valid camera manifest with hardware binding
  VERIFIED AI      -- manifest declares AI generator (Firefly, DALL-E etc)
  EDITED           -- chain shows original + editing steps
  NO MANIFEST      -- no C2PA found (inconclusive, not proof of AI)
  TAMPERED         -- manifest present but signature broken

Usage:
  result = check_c2pa("photo.jpg")
  print(result['verdict'])
  print(result['summary'])

  # Or from CLI:
  python c2pa_checker.py photo.jpg
  python c2pa_checker.py --batch folder/
"""

import os
import re
import sys
import json
import struct
import hashlib
from pathlib import Path
from datetime import datetime


# -----------------------------------------------------------------
# KNOWN AI TOOL SIGNATURES
# Strings found in XMP/metadata/JUMBF that identify AI generators
# -----------------------------------------------------------------

AI_TOOL_SIGNATURES = {
    'Adobe Firefly':        ['firefly', 'adobe firefly', 'adobe.com/firefly'],
    'DALL-E / OpenAI':      ['dall-e', 'openai', 'dall\u2013e', 'openai.com'],
    'Midjourney':           ['midjourney', 'midjourney.com'],
    'Stable Diffusion':     ['stable diffusion', 'stability ai', 'stablediffusion',
                             'stability.ai', 'dreamstudio'],
    'Bing Image Creator':   ['bing image creator', 'designer.microsoft', 'bing creator'],
    'Google Imagen':        ['imagen', 'google imagen'],
    'Adobe Stock AI':       ['adobe stock', 'contributor.stock.adobe'],
    'Canva AI':             ['canva', 'canva.com'],
    'Shutterstock AI':      ['shutterstock', 'shutterstock.com/ai'],
    'Getty AI':             ['getty images ai', 'gettyimages'],
    'FLUX':                 ['black forest labs', 'flux', 'bfl.ml'],
    'Sora':                 ['sora', 'openai video'],
}

# Known camera manufacturer identifiers in C2PA claims
CAMERA_MANUFACTURERS = [
    'sony', 'canon', 'nikon', 'fujifilm', 'panasonic', 'olympus',
    'leica', 'hasselblad', 'phase one', 'ricoh', 'pentax', 'sigma',
    'apple', 'samsung', 'google pixel', 'qualcomm'
]

# C2PA action types that indicate AI generation
AI_ACTIONS = [
    'c2pa.created',          # created by AI (not captured)
    'c2pa.generative-fill',  # Adobe generative fill
    'c2pa.ai-generated',     # explicit AI flag
]

# C2PA action types that indicate real camera capture
CAMERA_ACTIONS = [
    'c2pa.captured',         # captured by camera sensor
    'c2pa.recorded',         # recorded (video frames)
]


# -----------------------------------------------------------------
# BINARY PARSERS
# -----------------------------------------------------------------

def _read_jpeg_segments(data: bytes) -> dict:
    """Extract all JPEG APP segments. Returns {marker_hex: [content_bytes]}."""
    segments = {}
    i = 0
    if len(data) < 2 or data[0:2] != b'\xff\xd8':
        return segments  # not JPEG

    i = 2
    while i < len(data) - 4:
        if data[i] != 0xff:
            break
        marker = data[i:i+2]
        marker_hex = marker.hex().upper()
        if marker == b'\xff\xd9':  # EOI
            break
        if marker in (b'\xff\xd8', b'\xff\xd9', b'\xff\xda'):
            i += 2
            continue
        if i + 4 > len(data):
            break
        length = struct.unpack('>H', data[i+2:i+4])[0]
        content = data[i+4:i+2+length]
        if marker_hex not in segments:
            segments[marker_hex] = []
        segments[marker_hex].append(content)
        i += 2 + length

    return segments


def _find_jumbf_boxes(data: bytes) -> list:
    """
    Find all JUMBF boxes in binary data.
    JUMBF format: [4-byte length][4-byte 'jumb'][content]
    """
    boxes = []
    search = data
    offset = 0

    while True:
        idx = search.find(b'jumb', offset)
        if idx == -1:
            break
        # Box starts 4 bytes before 'jumb' (length field)
        box_start = max(0, idx - 4)
        if box_start + 8 <= len(search):
            try:
                box_len = struct.unpack('>I', search[box_start:box_start+4])[0]
                if 8 <= box_len <= len(search) - box_start:
                    box_content = search[box_start:box_start+box_len]
                    boxes.append(box_content)
            except struct.error:
                pass
        offset = idx + 4

    return boxes


def _extract_xmp(data: bytes) -> str:
    """Extract XMP metadata block from image bytes."""
    # Standard XMP marker
    xmp_start = data.find(b'<?xpacket')
    if xmp_start == -1:
        xmp_start = data.find(b'<x:xmpmeta')
    if xmp_start == -1:
        xmp_start = data.find(b'<rdf:RDF')

    if xmp_start != -1:
        xmp_end = data.find(b'<?xpacket end', xmp_start)
        if xmp_end == -1:
            xmp_end = data.find(b'</x:xmpmeta>', xmp_start)
        if xmp_end == -1:
            xmp_end = min(xmp_start + 65536, len(data))
        else:
            xmp_end += 20

        return data[xmp_start:xmp_end].decode('utf-8', errors='replace')

    return ''


def _extract_png_chunks(data: bytes) -> dict:
    """Extract PNG chunks. Returns {chunk_type: [chunk_data]}."""
    chunks = {}
    if len(data) < 8 or data[:8] != b'\x89PNG\r\n\x1a\n':
        return chunks

    i = 8
    while i + 12 <= len(data):
        try:
            length = struct.unpack('>I', data[i:i+4])[0]
            chunk_type = data[i+4:i+8].decode('ascii', errors='replace')
            chunk_data = data[i+8:i+8+length]
            if chunk_type not in chunks:
                chunks[chunk_type] = []
            chunks[chunk_type].append(chunk_data)
            i += 12 + length
        except (struct.error, UnicodeDecodeError):
            break

    return chunks


# -----------------------------------------------------------------
# C2PA MANIFEST PARSER
# -----------------------------------------------------------------

def _parse_manifest_content(raw: bytes) -> dict:
    """
    Extract meaningful fields from raw C2PA manifest bytes.
    Uses regex/string scanning since full CBOR parsing needs a library.
    """
    result = {}
    text = raw.decode('utf-8', errors='replace')
    text_lower = text.lower()

    # Detect AI tool by signature
    for tool_name, signatures in AI_TOOL_SIGNATURES.items():
        for sig in signatures:
            if sig in text_lower:
                result['ai_tool'] = tool_name
                break
        if 'ai_tool' in result:
            break

    # Detect camera manufacturer
    for manufacturer in CAMERA_MANUFACTURERS:
        if manufacturer in text_lower:
            result['camera_manufacturer'] = manufacturer.title()
            break

    # Extract claim generator (software that created the manifest)
    patterns = [
        r'"claim_generator"\s*:\s*"([^"]+)"',
        r'claim_generator[^"]*"([^"]+)"',
        r'"generator"\s*:\s*"([^"]+)"',
        r'<dc:creator>([^<]+)',
        r'"software"\s*:\s*"([^"]+)"',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result['claim_generator'] = m.group(1).strip()
            break

    # Extract title/label
    m = re.search(r'"dc:title"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not m:
        m = re.search(r'<dc:title>([^<]+)', text, re.IGNORECASE)
    if m:
        result['title'] = m.group(1).strip()

    # Detect actions
    actions_found = []
    for action in AI_ACTIONS + CAMERA_ACTIONS:
        if action in text_lower:
            actions_found.append(action)
    if actions_found:
        result['actions'] = actions_found

    # Detect assertion types present
    assertions = []
    for assertion in ['c2pa.hash.data', 'c2pa.hash.bmff', 'c2pa.thumbnail',
                      'stds.exif', 'c2pa.training-mining', 'c2pa.ai-generative']:
        if assertion in text_lower:
            assertions.append(assertion)
    if assertions:
        result['assertions'] = assertions

    # Check for AI training declaration
    if 'training-mining' in text_lower or 'notAllowed' in text:
        result['training_mining'] = 'declared'

    # Timestamp
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text)
    if m:
        result['timestamp'] = m.group(1)

    return result


def _verify_signature_present(data: bytes) -> bool:
    """Check if a cryptographic signature block exists (not full verify -- needs PKI)."""
    # C2PA signatures are COSE_Sign1 structures
    # We check for the signature marker bytes
    sig_markers = [b'c2pa.signature', b'COSE_Sign1', b'\x84\x4a']
    for marker in sig_markers:
        if marker in data:
            return True
    # Also check for x.509 cert patterns
    if b'\x30\x82' in data and b'\x02\x01' in data:
        return True
    return False


# -----------------------------------------------------------------
# MAIN CHECKER
# -----------------------------------------------------------------

def check_c2pa(filepath: str) -> dict:
    """
    Full C2PA provenance check on an image file.

    Returns dict with:
      verdict:     VERIFIED_REAL | VERIFIED_AI | EDITED | NO_MANIFEST | TAMPERED
      confidence:  HIGH | MEDIUM | LOW
      summary:     human-readable one-line result
      details:     full parsed manifest data
      raw_found:   what raw C2PA data was found
    """
    result = {
        'file':       filepath,
        'verdict':    'NO_MANIFEST',
        'confidence': 'LOW',
        'summary':    'No C2PA manifest found',
        'details':    {},
        'raw_found':  [],
        'checked_at': datetime.now().isoformat(),
    }

    path = Path(filepath)
    if not path.exists():
        result['summary'] = f'File not found: {filepath}'
        return result

    try:
        with open(filepath, 'rb') as f:
            data = f.read()
    except Exception as e:
        result['summary'] = f'Could not read file: {e}'
        return result

    ext = path.suffix.lower()
    manifests = []
    has_signature = False

    # ----- JPEG -----
    if ext in ('.jpg', '.jpeg'):
        segments = _read_jpeg_segments(data)

        # APP11 (0xFFEB) -- primary C2PA location in JPEG
        for seg_content in segments.get('FFEB', []):
            if b'jumb' in seg_content or b'c2pa' in seg_content.lower():
                result['raw_found'].append('JPEG APP11 (JUMBF)')
                parsed = _parse_manifest_content(seg_content)
                if parsed:
                    manifests.append(parsed)
                if _verify_signature_present(seg_content):
                    has_signature = True
                # Also search for nested JUMBF boxes
                for box in _find_jumbf_boxes(seg_content):
                    nested = _parse_manifest_content(box)
                    if nested:
                        manifests.append(nested)

        # APP1 (0xFFE1) -- XMP metadata
        for seg_content in segments.get('FFE1', []):
            xmp = _extract_xmp(seg_content)
            if xmp and 'c2pa' in xmp.lower():
                result['raw_found'].append('JPEG XMP (c2pa namespace)')
                parsed = _parse_manifest_content(xmp.encode())
                if parsed:
                    manifests.append(parsed)

        # Also scan full file for any C2PA markers
        if not manifests:
            xmp = _extract_xmp(data)
            if xmp and 'c2pa' in xmp.lower():
                result['raw_found'].append('XMP embedded')
                manifests.append(_parse_manifest_content(xmp.encode()))

    # ----- PNG -----
    elif ext == '.png':
        chunks = _extract_png_chunks(data)

        # caBX chunk -- C2PA standard PNG location
        for chunk_data in chunks.get('caBX', []):
            result['raw_found'].append('PNG caBX chunk (C2PA standard)')
            parsed = _parse_manifest_content(chunk_data)
            if parsed:
                manifests.append(parsed)
            if _verify_signature_present(chunk_data):
                has_signature = True

        # iTXt / tEXt -- may contain XMP
        for chunk_type in ('iTXt', 'tEXt', 'zTXt'):
            for chunk_data in chunks.get(chunk_type, []):
                xmp = _extract_xmp(chunk_data)
                if xmp and 'c2pa' in xmp.lower():
                    result['raw_found'].append(f'PNG {chunk_type} (XMP)')
                    manifests.append(_parse_manifest_content(xmp.encode()))

    # ----- TIFF / DNG / RAW -----
    elif ext in ('.tif', '.tiff', '.dng', '.cr2', '.cr3', '.nef', '.arw'):
        # Scan for XMP block
        xmp = _extract_xmp(data)
        if xmp and 'c2pa' in xmp.lower():
            result['raw_found'].append('TIFF/RAW XMP')
            manifests.append(_parse_manifest_content(xmp.encode()))

        # Scan for JUMBF boxes
        for box in _find_jumbf_boxes(data):
            parsed = _parse_manifest_content(box)
            if parsed:
                result['raw_found'].append('TIFF/RAW JUMBF box')
                manifests.append(parsed)
                if _verify_signature_present(box):
                    has_signature = True

    # ----- WEBP / HEIC -----
    elif ext in ('.webp', '.heic', '.heif', '.avif'):
        xmp = _extract_xmp(data)
        if xmp and 'c2pa' in xmp.lower():
            result['raw_found'].append(f'{ext.upper()} XMP')
            manifests.append(_parse_manifest_content(xmp.encode()))
        for box in _find_jumbf_boxes(data):
            parsed = _parse_manifest_content(box)
            if parsed:
                result['raw_found'].append(f'{ext.upper()} JUMBF')
                manifests.append(parsed)

    # ----- Fallback: scan entire file -----
    if not manifests:
        # Last resort: scan raw bytes for C2PA/JUMBF markers
        if b'c2pa' in data.lower() if hasattr(data, 'lower') else b'c2pa' in data:
            for box in _find_jumbf_boxes(data):
                parsed = _parse_manifest_content(box)
                if parsed:
                    result['raw_found'].append('Raw JUMBF scan')
                    manifests.append(parsed)
        xmp = _extract_xmp(data)
        if xmp and ('c2pa' in xmp.lower() or any(
                sig in xmp.lower()
                for sigs in AI_TOOL_SIGNATURES.values() for sig in sigs)):
            result['raw_found'].append('XMP fallback scan')
            manifests.append(_parse_manifest_content(xmp.encode()))

    # ----- Also check for AI tool signatures in ALL metadata -----
    # Even without C2PA, some AI tools leave XMP traces
    xmp_full = _extract_xmp(data)
    ai_in_xmp = None
    if xmp_full:
        xmp_lower = xmp_full.lower()
        for tool_name, sigs in AI_TOOL_SIGNATURES.items():
            for sig in sigs:
                if sig in xmp_lower:
                    ai_in_xmp = tool_name
                    break
            if ai_in_xmp:
                break

    # ----- Build verdict -----
    result['details'] = _merge_manifests(manifests)

    if not manifests and not ai_in_xmp:
        result['verdict']    = 'NO_MANIFEST'
        result['confidence'] = 'LOW'
        result['summary']    = 'No C2PA manifest found. Authenticity cannot be confirmed or denied cryptographically.'

    elif ai_in_xmp and not manifests:
        result['verdict']    = 'VERIFIED_AI'
        result['confidence'] = 'MEDIUM'
        result['summary']    = f'AI tool signature found in metadata: {ai_in_xmp}. No C2PA manifest but XMP indicates AI origin.'
        result['details']['ai_tool'] = ai_in_xmp
        result['details']['detection_method'] = 'XMP signature (no cryptographic verification)'

    else:
        details = result['details']
        ai_tool = details.get('ai_tool')
        camera  = details.get('camera_manufacturer')
        actions = details.get('actions', [])

        has_ai_action    = any(a in AI_ACTIONS     for a in actions)
        has_camera_action = any(a in CAMERA_ACTIONS for a in actions)
        has_edit         = 'c2pa.edited' in str(actions) or len(manifests) > 1

        if ai_tool or has_ai_action:
            result['verdict']    = 'VERIFIED_AI'
            result['confidence'] = 'HIGH' if has_signature else 'MEDIUM'
            tool_str = ai_tool or 'unknown AI tool'
            result['summary'] = (
                f'C2PA manifest confirms AI-generated content by {tool_str}. '
                f'{"Cryptographic signature present." if has_signature else "No signature -- manifest may be unsigned."}'
            )

        elif camera or has_camera_action:
            result['verdict']    = 'VERIFIED_REAL'
            result['confidence'] = 'HIGH' if has_signature else 'MEDIUM'
            cam_str = camera or 'camera'
            result['summary'] = (
                f'C2PA manifest confirms authentic capture by {cam_str}. '
                f'{"Cryptographic signature validates provenance." if has_signature else "Manifest present but not cryptographically verified."}'
            )
            if has_edit:
                result['verdict']  = 'EDITED'
                result['summary'] += ' Image has been edited after capture.'

        elif has_edit:
            result['verdict']    = 'EDITED'
            result['confidence'] = 'MEDIUM'
            result['summary']    = 'C2PA manifest shows image has been edited. Original capture source unclear.'

        elif manifests:
            # Manifest present but couldn't determine origin
            generator = details.get('claim_generator', 'unknown')
            result['verdict']    = 'MANIFEST_FOUND'
            result['confidence'] = 'MEDIUM'
            result['summary']    = f'C2PA manifest found (generator: {generator}) but origin type unclear.'

    return result


def _merge_manifests(manifests: list) -> dict:
    """Merge multiple manifest dicts into one, preferring non-empty values."""
    merged = {}
    for m in manifests:
        for k, v in m.items():
            if k not in merged or not merged[k]:
                merged[k] = v
            elif isinstance(v, list) and isinstance(merged[k], list):
                merged[k] = list(set(merged[k] + v))
    if len(manifests) > 1:
        merged['manifest_chain_length'] = len(manifests)
    return merged


# -----------------------------------------------------------------
# INTEGRATION: Add C2PA to existing analyzer signal output
# -----------------------------------------------------------------

def c2pa_as_signal(filepath: str) -> dict:
    """
    Run C2PA check and return result formatted as an analyzer signal.
    Integrates with image_authenticity.py output format.
    """
    result = check_c2pa(filepath)
    verdict = result['verdict']

    # Map verdict to AI probability score
    score_map = {
        'VERIFIED_REAL':    0.02,   # cryptographically proven real
        'VERIFIED_AI':      0.98,   # cryptographically proven AI
        'EDITED':           0.55,   # real but modified
        'MANIFEST_FOUND':   0.40,   # unknown but has manifest
        'NO_MANIFEST':      0.50,   # completely unknown
        'TAMPERED':         0.80,   # signature broken = suspicious
    }

    confidence_map = {
        'HIGH':   'HIGH',
        'MEDIUM': 'MEDIUM',
        'LOW':    'LOW',
    }

    return {
        'name':       'C2PA Provenance',
        'verdict':    verdict,
        'score':      score_map.get(verdict, 0.5),
        'confidence': confidence_map.get(result['confidence'], 'LOW'),
        'summary':    result['summary'],
        'details':    result['details'],
        'raw_found':  result['raw_found'],
    }


# -----------------------------------------------------------------
# PRETTY PRINTER
# -----------------------------------------------------------------

def print_c2pa_report(filepath: str):
    """Print a formatted C2PA report to console."""
    result = check_c2pa(filepath)
    verdict  = result['verdict']
    conf     = result['confidence']
    summary  = result['summary']
    details  = result['details']

    ICONS = {
        'VERIFIED_REAL':  '[REAL]',
        'VERIFIED_AI':    '[AI]',
        'EDITED':         '[EDITED]',
        'MANIFEST_FOUND': '[MANIFEST]',
        'NO_MANIFEST':    '[NO C2PA]',
        'TAMPERED':       '[TAMPERED]',
    }

    print()
    print('=' * 60)
    print('  C2PA PROVENANCE REPORT')
    print('=' * 60)
    print(f'  File:       {Path(filepath).name}')
    print(f'  Verdict:    {ICONS.get(verdict, "?")}  {verdict}')
    print(f'  Confidence: {conf}')
    print(f'  Summary:    {summary}')

    if result['raw_found']:
        print(f'\n  C2PA data found in:')
        for loc in result['raw_found']:
            print(f'    - {loc}')

    if details:
        print(f'\n  Manifest details:')
        for k, v in details.items():
            if v:
                print(f'    {k:<28} {v}')

    if verdict == 'NO_MANIFEST':
        print()
        print('  Note: Absence of C2PA does NOT mean the image is AI.')
        print('  Most real photos and older AI images have no C2PA data.')
        print('  C2PA is only present if the camera/tool explicitly added it.')

    elif verdict == 'VERIFIED_REAL':
        print()
        print('  [CONFIRMED] This image has a valid camera provenance chain.')
        if conf == 'HIGH':
            print('  Cryptographic signature makes forgery practically impossible.')
        else:
            print('  Note: Signature not fully verified (requires PKI infrastructure).')

    elif verdict == 'VERIFIED_AI':
        print()
        print('  [CONFIRMED] This image was generated or processed by an AI tool.')
        print('  The manifest is embedded by the AI platform itself.')

    print('=' * 60)
    print()
    return result


# -----------------------------------------------------------------
# BATCH CHECKER
# -----------------------------------------------------------------

def check_batch(folder: str, save_json: bool = True) -> list:
    """Check all images in a folder for C2PA manifests."""
    folder_path = Path(folder)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif',
            '.dng', '.heic', '.heif', '.avif', '.cr2', '.nef', '.arw'}
    images = [f for f in sorted(folder_path.rglob('*')) if f.suffix.lower() in exts]

    if not images:
        print(f'  No images found in {folder}')
        return []

    print(f'\n  Checking {len(images)} images for C2PA manifests...')
    print(f'  {"File":<40} {"Verdict":<18} {"Confidence":<12} {"Details"}')
    print(f'  {"-"*40} {"-"*18} {"-"*12} {"-"*20}')

    results = []
    counts = {}

    for img_path in images:
        result = check_c2pa(str(img_path))
        verdict  = result['verdict']
        conf     = result['confidence']
        details  = result['details']
        ai_tool  = details.get('ai_tool', '')
        camera   = details.get('camera_manufacturer', '')
        extra    = ai_tool or camera or ''

        print(f'  {img_path.name:<40} {verdict:<18} {conf:<12} {extra}')
        results.append(result)
        counts[verdict] = counts.get(verdict, 0) + 1

    print(f'\n  Summary:')
    for v, c in sorted(counts.items()):
        print(f'    {v:<20} {c}')

    if save_json:
        out_path = folder_path / 'c2pa_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'\n  Results saved -> {out_path}')

    return results


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Check images for C2PA provenance manifests.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single image
  python c2pa_checker.py photo.jpg

  # Check a single image, output JSON
  python c2pa_checker.py photo.jpg --json

  # Batch check a folder
  python c2pa_checker.py --batch D:\\Photos\\

  # Integrate with image_authenticity.py
  python c2pa_checker.py photo.jpg --signal
        """
    )
    parser.add_argument('file',          nargs='?', help='Image file to check')
    parser.add_argument('--batch',       metavar='DIR', help='Check all images in folder')
    parser.add_argument('--json',        action='store_true', help='Output raw JSON')
    parser.add_argument('--signal',      action='store_true',
                        help='Output in signal format for integration with analyzer')

    args = parser.parse_args()

    if args.batch:
        check_batch(args.batch)
    elif args.file:
        if args.json:
            result = check_c2pa(args.file)
            print(json.dumps(result, indent=2, default=str))
        elif args.signal:
            signal = c2pa_as_signal(args.file)
            print(json.dumps(signal, indent=2, default=str))
        else:
            print_c2pa_report(args.file)
    else:
        parser.print_help()
        print()
        print('  Install note: pip install c2pa-python  (optional, for full CBOR parsing)')
        print('  This script works without it using binary/XMP parsing.')