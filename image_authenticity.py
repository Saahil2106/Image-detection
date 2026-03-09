"""
Image Authenticity Analyzer
============================
Detects:
  1. GenAI vs Real Photograph (statistical + frequency + texture signals)
  2. Editing / Tampering (ELA, clone detection, noise inconsistency)
  3. Metadata forensics (EXIF, C2PA-style provenance hints)

Usage:
    python image_authenticity.py <image_path>
    python image_authenticity.py <image_path> --verbose
    python image_authenticity.py <image_path> --save-report

Requirements (auto-installed on first run):
    pip install pillow numpy scipy scikit-image scikit-learn opencv-python matplotlib
"""

# ─────────────────────────────────────────────
# AUTO-INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
import sys
import subprocess

REQUIRED = {
    "numpy":      "numpy",
    "PIL":        "pillow",
    "cv2":        "opencv-python",
    "scipy":      "scipy",
    "skimage":    "scikit-image",
    "sklearn":    "scikit-learn",
    "matplotlib": "matplotlib",
}

def _ensure_deps():
    missing = []
    for import_name, pip_name in REQUIRED.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"\n  Installing missing packages: {', '.join(missing)}")
        print("  (this only happens once)\n")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing
            )
            print("  Packages installed successfully.\n")
        except subprocess.CalledProcessError:
            print(f"\n  Auto-install failed. Please run manually:\n")
            print(f"      pip install {' '.join(missing)}\n")
            sys.exit(1)

_ensure_deps()

# ─────────────────────────────────────────────
# IMPORTS (after ensuring deps are present)
# ─────────────────────────────────────────────
import os
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ExifTags
import cv2
from scipy import stats
from scipy.fft import fft2, fftshift
from skimage import filters, feature, measure
from skimage.util import img_as_float
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class SignalResult:
    name: str
    value: float        # raw measured value
    score: float        # 0.0 → definitely photo, 1.0 → definitely AI/edited
    confidence: str     # HIGH / MEDIUM / LOW
    detail: str         # human-readable explanation
    category: str       # 'ai_detection' | 'edit_detection' | 'metadata'

@dataclass
class AnalysisReport:
    filepath: str
    image_size: tuple
    signals: list = field(default_factory=list)

    # Final verdicts
    ai_probability: float = 0.0       # 0–1
    edit_probability: float = 0.0     # 0–1
    authenticity_score: float = 0.0   # 0–1 (1 = very likely authentic photo)

    verdict: str = ""
    summary: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_image(path: str):
    """Load image as PIL + numpy (BGR for OpenCV, RGB for skimage)."""
    pil_img = Image.open(path).convert("RGB")
    np_rgb = np.array(pil_img, dtype=np.float32)
    np_bgr = cv2.cvtColor(np_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    return pil_img, np_rgb, np_bgr, gray


def score_to_label(score: float, threshold_low=0.35, threshold_high=0.65):
    if score < threshold_low:
        return "LOW"
    elif score < threshold_high:
        return "MEDIUM"
    else:
        return "HIGH"


# ─────────────────────────────────────────────
# MODULE 1 — METADATA ANALYSIS
# ─────────────────────────────────────────────

def analyze_metadata(pil_img: Image.Image, filepath: str) -> list[SignalResult]:
    results = []
    meta = {}

    # --- EXIF extraction ---
    try:
        exif_raw = pil_img._getexif()
        if exif_raw:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()
                    if not isinstance(v, bytes)}
        else:
            exif = {}
    except Exception:
        exif = {}

    meta['exif'] = exif

    # Signal: Does EXIF exist at all?
    has_exif = len(exif) > 0
    camera_fields = {'Make', 'Model', 'LensModel', 'LensMake'}
    has_camera = bool(camera_fields & set(exif.keys()))
    has_gps    = 'GPSInfo' in exif
    software   = str(exif.get('Software', '')).lower()

    # AI software keywords
    ai_software_hints = ['stable diffusion', 'midjourney', 'dall-e', 'firefly',
                         'generative', 'ai', 'comfyui', 'invoke', 'novelai']
    edit_software_hints = ['photoshop', 'lightroom', 'gimp', 'affinity',
                           'capture one', 'darktable', 'snapseed']

    software_ai_hit   = any(h in software for h in ai_software_hints)
    software_edit_hit = any(h in software for h in edit_software_hints)

    # Score: missing EXIF or camera data raises suspicion for AI
    if has_camera:
        exif_score = 0.1   # strong photo signal
        exif_detail = f"Camera: {exif.get('Make','')} {exif.get('Model','')}. Metadata suggests real device capture."
    elif has_exif and not has_camera:
        exif_score = 0.5
        exif_detail = "EXIF present but no camera hardware fields — could be exported/edited image."
    else:
        exif_score = 0.7
        exif_detail = "No EXIF data. Stripped (common in AI outputs or scrubbed edits)."

    if software_ai_hit:
        exif_score = 0.95
        exif_detail = f"Software tag reads '{exif.get('Software','')}' — indicates AI generation."
    elif software_edit_hit:
        exif_score = max(exif_score, 0.4)
        exif_detail += f" | Software: '{exif.get('Software','')}' — indicates post-processing."

    results.append(SignalResult(
        name="EXIF / Metadata",
        value=float(has_exif),
        score=exif_score,
        confidence="HIGH" if (has_camera or software_ai_hit) else "MEDIUM",
        detail=exif_detail,
        category="metadata"
    ))

    # Signal: DateTime consistency
    dt_orig    = exif.get('DateTimeOriginal', '')
    dt_digit   = exif.get('DateTimeDigitized', '')
    dt_mod     = exif.get('DateTime', '')

    if dt_orig and dt_mod and dt_orig != dt_mod:
        results.append(SignalResult(
            name="Timestamp Mismatch",
            value=1.0,
            score=0.7,
            confidence="MEDIUM",
            detail=f"Original: {dt_orig} | Modified: {dt_mod}. Suggests post-capture editing.",
            category="edit_detection"
        ))
    elif dt_orig:
        results.append(SignalResult(
            name="Timestamp Consistency",
            value=0.0,
            score=0.1,
            confidence="MEDIUM",
            detail=f"Capture timestamp {dt_orig} is consistent. No post-edit timestamp drift.",
            category="metadata"
        ))

    # Signal: File format & compression clues
    fmt = pil_img.format or Path(filepath).suffix.upper().lstrip('.')
    mode = pil_img.mode
    results.append(SignalResult(
        name="File Format",
        value=0.0,
        score=0.1 if fmt in ('JPEG', 'TIFF', 'RAW', 'DNG', 'CR2', 'NEF') else 0.4,
        confidence="LOW",
        detail=f"Format: {fmt}, Mode: {mode}. " +
               ("Camera formats favor authenticity." if fmt in ('JPEG','TIFF') else "PNG/WEBP common in AI outputs."),
        category="metadata"
    ))

    meta['software'] = software
    meta['has_camera'] = has_camera
    meta['has_gps'] = has_gps
    return results, meta


# ─────────────────────────────────────────────
# MODULE 2 — FREQUENCY DOMAIN ANALYSIS
# ─────────────────────────────────────────────

def analyze_frequency(gray: np.ndarray) -> list[SignalResult]:
    """
    Real photos have organic 1/f frequency falloff.
    AI images (especially diffusion) show characteristic spectral artifacts.
    """
    results = []

    # Work on center crop for stability — force exact sz×sz to avoid index mismatch
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    sz = min(h, w, 512)
    # Use even sz to guarantee symmetric slicing
    if sz % 2 != 0:
        sz -= 1
    crop = gray[cy - sz//2: cy + sz//2, cx - sz//2: cx + sz//2].astype(np.float32)
    # Enforce exact size in case of rounding on odd dimensions
    crop = crop[:sz, :sz]

    # 2D FFT
    f = fft2(crop)
    fshift = fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Radial power spectrum: real cameras follow power law ~1/f^alpha
    actual_sz = crop.shape[0]   # use actual after crop, not assumed sz
    cy2, cx2 = actual_sz // 2, actual_sz // 2
    y_idx, x_idx = np.indices((actual_sz, actual_sz))
    r = np.sqrt((x_idx - cx2)**2 + (y_idx - cy2)**2).astype(int)
    r = r.ravel()
    mag_flat = magnitude.ravel()

    max_r = actual_sz // 2
    radial_mean = np.array([mag_flat[r == ri].mean() if (r == ri).any() else 0
                            for ri in range(1, max_r)])

    # Fit log-log to get spectral slope
    log_r   = np.log(np.arange(1, len(radial_mean) + 1))
    log_mag = np.log1p(radial_mean)
    slope, intercept, r_val, p_val, _ = stats.linregress(log_r[5:], log_mag[5:])

    # Real photos: slope typically -1.5 to -3.0
    # AI images: slope tends to be shallower (-0.5 to -1.5) or irregular
    # Natural camera range: -0.5 to -3.5. AI images cluster near 0 (too flat) or extreme.
    # Only flag genuinely unusual slopes, not moderate deviations
    if -3.5 <= slope <= -0.5:
        slope_score = 0.15   # within natural camera range
    elif -4.5 <= slope <= 0.0:
        slope_score = 0.45   # borderline
    else:
        slope_score = 0.80   # extreme — strongly unnatural

    results.append(SignalResult(
        name="Spectral Power Slope",
        value=float(slope),
        score=slope_score,
        confidence="MEDIUM",
        detail=f"Log-log spectral slope = {slope:.3f}. "
               f"Natural cameras: -0.5 to -3.5. "
               f"{'Within natural range.' if slope_score < 0.3 else 'Outside natural range — unnatural frequency distribution.'}",
        category="ai_detection"
    ))

    # High-frequency energy ratio: AI often has more HF ringing
    low_r_mask  = r < max_r // 4
    high_r_mask = r >= max_r * 3 // 4
    lf_energy = mag_flat[low_r_mask].mean()
    hf_energy = mag_flat[high_r_mask].mean()
    hf_ratio  = hf_energy / (lf_energy + 1e-8)

    # Natural photos: HF energy ~0.2–0.5. AI: can be higher or oddly uniform
    # Many real photos also have high HF — this is a weak signal alone
    # Only flag extreme outliers (ratio > 0.85 or < 0.05)
    if hf_ratio > 0.85:
        hf_score = 0.60
    elif hf_ratio < 0.05:
        hf_score = 0.55
    else:
        hf_score = 0.15   # normal range, no signal

    results.append(SignalResult(
        name="High-Frequency Energy",
        value=float(hf_ratio),
        score=hf_score,
        confidence="LOW",
        detail=f"HF/LF ratio = {hf_ratio:.3f}. "
               f"{'Extreme HF ratio — outside normal range.' if hf_score > 0.4 else 'HF energy within normal photographic range.'}",
        category="ai_detection"
    ))

    return results, magnitude


# ─────────────────────────────────────────────
# MODULE 3 — NOISE ANALYSIS (PRNU / Camera Fingerprint)
# ─────────────────────────────────────────────

def analyze_noise(gray: np.ndarray) -> list[SignalResult]:
    """
    Real cameras leave consistent sensor noise patterns (PRNU).
    AI images have smoother or structurally different noise.
    """
    results = []
    img_f = gray.astype(np.float32) / 255.0

    # Denoise and extract residual noise
    denoised = cv2.GaussianBlur(img_f, (5, 5), 1.5)
    noise_residual = img_f - denoised

    # Noise statistics
    noise_std  = noise_residual.std()
    noise_mean = noise_residual.mean()
    noise_kurt = float(stats.kurtosis(noise_residual.ravel()))
    noise_skew = float(stats.skew(noise_residual.ravel()))

    # Real photos: noise std ~ 0.005–0.03, kurtosis mildly positive
    # AI images: often too clean (low std) or over-textured
    if noise_std < 0.003:
        noise_score = 0.8
        noise_detail = f"Very low noise (σ={noise_std:.4f}). Artificially clean — typical of AI renders."
    elif noise_std > 0.06:
        noise_score = 0.65
        noise_detail = f"Unusually high noise (σ={noise_std:.4f}). May indicate heavy compression or synthetic texture."
    else:
        noise_score = 0.2
        noise_detail = f"Noise level σ={noise_std:.4f} consistent with real camera sensor."

    results.append(SignalResult(
        name="Noise Level (PRNU)",
        value=float(noise_std),
        score=noise_score,
        confidence="MEDIUM",
        detail=noise_detail,
        category="ai_detection"
    ))

    # Kurtosis of noise: real cameras → near Gaussian (kurtosis ~0–1)
    # AI can produce super-Gaussian or sub-Gaussian noise
    kurt_score = min(1.0, abs(noise_kurt) / 8.0)
    results.append(SignalResult(
        name="Noise Distribution (Kurtosis)",
        value=float(noise_kurt),
        score=kurt_score,
        confidence="LOW",
        detail=f"Noise kurtosis={noise_kurt:.3f}. "
               f"{'Near-Gaussian noise: consistent with camera.' if kurt_score < 0.3 else 'Non-Gaussian noise distribution — atypical for sensors.'}",
        category="ai_detection"
    ))

    # Local noise variance map — look for inconsistency (edit signal)
    h, w = gray.shape
    block = 32
    var_map = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = noise_residual[y:y+block, x:x+block]
            var_map.append(patch.var())
    var_map = np.array(var_map)
    noise_inconsistency = var_map.std() / (var_map.mean() + 1e-8)

    edit_score = min(1.0, noise_inconsistency / 2.0)
    results.append(SignalResult(
        name="Noise Spatial Inconsistency",
        value=float(noise_inconsistency),
        score=edit_score,
        confidence="MEDIUM",
        detail=f"Noise variance CV={noise_inconsistency:.3f} across image blocks. "
               + ('High inconsistency → different image regions may have different origins (edit/splice).' if edit_score > 0.5 else 'Noise uniformly distributed across regions.'),
        category="edit_detection"
    ))

    return results, noise_residual


# ─────────────────────────────────────────────
# MODULE 4 — ERROR LEVEL ANALYSIS (ELA)
# ─────────────────────────────────────────────

def analyze_ela(pil_img: Image.Image, quality: int = 92) -> list[SignalResult]:
    """
    ELA: Re-save at known JPEG quality, compare to original.
    Edited regions that were re-saved fewer times show higher error levels.
    Authentic, once-compressed images have uniform error distribution.
    """
    results = []
    import io

    # Re-save as JPEG at controlled quality
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    orig_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    ela_map = np.abs(orig_arr - recomp_arr)
    ela_amplified = np.clip(ela_map * 10, 0, 255).astype(np.uint8)

    ela_gray = ela_map.mean(axis=2)
    ela_mean = ela_gray.mean()
    ela_std  = ela_gray.std()

    # Coefficient of Variation of ELA: uniform → authentic, high CV → edited
    ela_cv = ela_std / (ela_mean + 1e-8)

    # Block-level ELA variance
    h, w = ela_gray.shape
    block = 64
    block_means = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            block_means.append(ela_gray[y:y+block, x:x+block].mean())
    block_means = np.array(block_means)
    block_cv = block_means.std() / (block_means.mean() + 1e-8)

    if block_cv > 1.5:
        ela_score = 0.85
        ela_detail = f"ELA block CV={block_cv:.2f}: highly non-uniform error levels. Regions with low ELA amid high-ELA background strongly suggest copy-paste or inpainting."
    elif block_cv > 0.7:
        ela_score = 0.5
        ela_detail = f"ELA block CV={block_cv:.2f}: moderate spatial variation. Possible light editing or format conversion."
    else:
        ela_score = 0.15
        ela_detail = f"ELA block CV={block_cv:.2f}: uniform error levels. Consistent with single-generation JPEG encoding."

    results.append(SignalResult(
        name="Error Level Analysis (ELA)",
        value=float(block_cv),
        score=ela_score,
        confidence="HIGH" if block_cv > 1.5 else "MEDIUM",
        detail=ela_detail,
        category="edit_detection"
    ))

    return results, ela_amplified


# ─────────────────────────────────────────────
# MODULE 5 — TEXTURE & EDGE ANALYSIS
# ─────────────────────────────────────────────

def analyze_texture(gray: np.ndarray, np_rgb: np.ndarray) -> list[SignalResult]:
    """
    AI images often have:
    - Too-perfect edges (no chromatic aberration)
    - Over-smooth skin / surfaces (diffusion over-blending)
    - Suspiciously uniform texture energy
    """
    results = []
    img_f = gray.astype(np.float32) / 255.0

    # Laplacian variance (sharpness measure)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    # GLCM-like local texture measure using LBP
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

    # AI images tend to have mid-range but UNIFORM LBP entropy
    # Real photos show more varied local texture
    lbp_score = min(1.0, max(0.0, (3.0 - lbp_entropy) / 3.0))

    results.append(SignalResult(
        name="Texture Complexity (LBP Entropy)",
        value=float(lbp_entropy),
        score=lbp_score,
        confidence="MEDIUM",
        detail=f"LBP entropy={lbp_entropy:.3f}. "
               f"{'Low entropy → overly smooth/uniform texture, AI trait.' if lbp_score > 0.5 else 'Rich texture entropy — consistent with photographic complexity.'}",
        category="ai_detection"
    ))

    # Chromatic Aberration check: real lenses cause R/B channel fringing at edges
    r_ch = np_rgb[:,:,0].astype(np.float32)
    b_ch = np_rgb[:,:,2].astype(np.float32)
    edges_r = cv2.Canny(r_ch.astype(np.uint8), 50, 150)
    edges_b = cv2.Canny(b_ch.astype(np.uint8), 50, 150)

    r_edge_pts = set(zip(*np.where(edges_r > 0)))
    b_edge_pts = set(zip(*np.where(edges_b > 0)))
    total_edges = len(r_edge_pts) + len(b_edge_pts) + 1
    shared_edges = len(r_edge_pts & b_edge_pts)
    alignment_ratio = shared_edges / total_edges
    total_edge_count = len(r_edge_pts) + len(b_edge_pts)

    # Only reliable when image has enough edges to measure
    # Minimum ~500 edge pixels needed for meaningful CA measurement
    if total_edge_count < 500:
        ca_score = 0.0   # inconclusive — not enough edges
        ca_conf = "LOW"
        ca_detail = f"Too few edges ({total_edge_count}) to measure chromatic aberration reliably. Signal inconclusive."
    elif alignment_ratio > 0.55:
        ca_score = min(1.0, (alignment_ratio - 0.55) / 0.30)
        ca_conf = "MEDIUM"
        ca_detail = f"R/B edge alignment={alignment_ratio:.3f}. High alignment → no lens CA → AI-generated indicator."
    else:
        ca_score = max(0.0, (alignment_ratio - 0.1) / 0.45)
        ca_conf = "MEDIUM"
        ca_detail = f"R/B edge alignment={alignment_ratio:.3f}. Channel misalignment present — consistent with real optical lens."

    results.append(SignalResult(
        name="Chromatic Aberration",
        value=float(alignment_ratio),
        score=ca_score,
        confidence=ca_conf,
        detail=ca_detail,
        category="ai_detection"
    ))

    # Edge coherence across color channels
    results.append(SignalResult(
        name="Laplacian Sharpness",
        value=float(lap_var),
        score=0.3 if 100 < lap_var < 5000 else 0.6,
        confidence="LOW",
        detail=f"Laplacian variance={lap_var:.1f}. "
               f"{'Sharpness in natural range.' if 100 < lap_var < 5000 else 'Unusual sharpness value — over-sharpened or synthetic.'}",
        category="ai_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 6 — COPY-MOVE / CLONE DETECTION
# ─────────────────────────────────────────────

def analyze_clone_detection(gray: np.ndarray) -> list[SignalResult]:
    """
    Detect copy-move forgery: regions that are duplicated/cloned in the image.
    Uses feature matching on dense SIFT keypoints within-image.
    """
    results = []

    # Downsample for speed
    h, w = gray.shape
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(gray, (int(w*scale), int(h*scale)))

    # ORB is faster, less memory than SIFT
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(small, None)

    clone_score = 0.0
    clone_detail = "Unable to extract features for clone detection."
    clone_count = 0

    if des is not None and len(kp) > 20:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        suspicious = 0
        for m_list in matches:
            if len(m_list) < 3:
                continue
            best, second, third = m_list[0], m_list[1], m_list[2]
            if best.queryIdx == best.trainIdx:
                continue  # self-match

            # Check spatial distance between matched keypoints
            pt1 = np.array(kp[best.queryIdx].pt)
            pt2 = np.array(kp[best.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)

            if dist > 20 and best.distance < 50:
                suspicious += 1

        clone_count = suspicious
        ratio = suspicious / (len(kp) + 1)

        if ratio > 0.12:
            clone_score = min(1.0, ratio * 5)
            clone_detail = f"Found {suspicious} suspicious near-duplicate feature pairs (ratio={ratio:.3f}). Strong indicator of copy-move or clone-stamp editing."
        elif ratio > 0.04:
            clone_score = 0.4
            clone_detail = f"Found {suspicious} potentially repeated features (ratio={ratio:.3f}). Possible minor cloning or repetitive scene elements."
        else:
            clone_score = 0.1
            clone_detail = f"Found {suspicious} repeated feature pairs (ratio={ratio:.3f}). No significant clone regions detected."

    results.append(SignalResult(
        name="Clone / Copy-Move Detection",
        value=float(clone_count),
        score=clone_score,
        confidence="MEDIUM" if clone_score > 0.3 else "LOW",
        detail=clone_detail,
        category="edit_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 7 — COMPRESSION GHOST ANALYSIS
# ─────────────────────────────────────────────

def analyze_compression_history(pil_img: Image.Image) -> list[SignalResult]:
    """
    Multi-quality ELA (GHOST method): scans across re-compression qualities.
    If image shows minimum difference at quality != original → likely re-compressed (edited).
    """
    results = []
    import io

    orig = np.array(pil_img.convert("RGB"), dtype=np.float32)
    qualities = [65, 75, 85, 92, 97]
    diffs = []

    for q in qualities:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
        diff = np.abs(orig - recomp).mean()
        diffs.append(diff)

    min_idx = int(np.argmin(diffs))
    min_q   = qualities[min_idx]
    min_diff = diffs[min_idx]
    spread  = max(diffs) - min(diffs)

    if spread < 1.0:
        ghost_score = 0.55
        ghost_detail = f"Very flat GHOST curve (spread={spread:.2f}). Could indicate non-JPEG source (AI/PNG) or heavily processed."
    elif min_q < 85:
        ghost_score = 0.65
        ghost_detail = f"GHOST minimum at quality={min_q} (diff={min_diff:.2f}). Image appears originally compressed at low quality — possible re-save after editing."
    elif min_q >= 92:
        ghost_score = 0.15
        ghost_detail = f"GHOST minimum at quality={min_q} — consistent with single high-quality JPEG compression. No re-save signatures."
    else:
        ghost_score = 0.35
        ghost_detail = f"GHOST minimum at quality={min_q}. Moderate suggestion of original compression level."

    results.append(SignalResult(
        name="JPEG Ghost (Compression History)",
        value=float(min_q),
        score=ghost_score,
        confidence="MEDIUM",
        detail=ghost_detail,
        category="edit_detection"
    ))

    return results, diffs, qualities



# ─────────────────────────────────────────────
# MODULE 8 — GLCM TEXTURE (Gray-Level Co-occurrence)
# ─────────────────────────────────────────────

def analyze_glcm_texture(gray: np.ndarray) -> list[SignalResult]:
    """
    GLCM homogeneity: AI diffusion images over-smooth local texture,
    producing unnaturally high homogeneity and low contrast.
    """
    from skimage.feature import graycomatrix, graycoprops
    results = []
    small = cv2.resize(gray, (128, 128))
    quantized = (small // 8).astype(np.uint8)  # reduce to 32 levels
    glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi/4],
                        levels=32, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    contrast    = graycoprops(glcm, 'contrast').mean()
    energy      = graycoprops(glcm, 'energy').mean()

    # Real photos: homogeneity ~0.6-0.85. AI: often >0.88 (over-smooth)
    hom_score = min(1.0, max(0.0, (homogeneity - 0.80) / 0.18))

    results.append(SignalResult(
        name="GLCM Texture Homogeneity",
        value=float(homogeneity),
        score=hom_score,
        confidence="HIGH" if abs(homogeneity - 0.83) > 0.05 else "MEDIUM",
        detail=f"Homogeneity={homogeneity:.4f}, Contrast={contrast:.4f}, Energy={energy:.4f}. "
               f"{'Unnaturally smooth texture (high homogeneity) — strong AI indicator.' if hom_score > 0.5 else 'Texture complexity consistent with real photography.'}",
        category="ai_detection"
    ))

    # Local variance uniformity — AI images have suspiciously uniform local detail
    g = gray.astype(np.float32)
    h, w = g.shape
    block = 16
    local_vars = [g[y:y+block, x:x+block].var()
                  for y in range(0, h - block, block)
                  for x in range(0, w - block, block)]
    lv = np.array(local_vars)
    lv_cv = lv.std() / (lv.mean() + 1e-8)
    # Real photos: CV typically 0.15-1.5. AI: often < 0.10 (unnaturally uniform)
    # Only flag genuinely low CV, not moderate values
    if lv_cv < 0.08:
        lv_score = 0.90
    elif lv_cv < 0.12:
        lv_score = 0.60
    elif lv_cv < 0.18:
        lv_score = 0.30
    else:
        lv_score = 0.05   # natural variation present

    results.append(SignalResult(
        name="Local Variance Uniformity",
        value=float(lv_cv),
        score=lv_score,
        confidence="HIGH" if lv_cv < 0.08 else "MEDIUM",
        detail=f"Local variance CV={lv_cv:.4f}. "
               f"{'Suspiciously uniform local detail (CV<0.10) — strong AI indicator.' if lv_score > 0.5 else f'Natural local detail variation (CV={lv_cv:.3f}) — consistent with real scene.'}",
        category="ai_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 9 — COLOR STATISTICS
# ─────────────────────────────────────────────

def analyze_color_statistics(np_rgb: np.ndarray) -> list[SignalResult]:
    """
    Real photos have specific saturation distributions and inter-channel
    correlations driven by lighting physics. AI violates these.
    """
    results = []

    # Saturation analysis
    hsv = cv2.cvtColor(np_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1].astype(float).ravel()
    sat_mean = sat.mean()
    sat_std  = sat.std()
    sat_kurt = float(stats.kurtosis(sat))

    # Real photos: saturation kurtosis > 2 (peaky distribution, many desaturated areas)
    # AI: often negative kurtosis (flat, globally saturated)
    kurt_score = min(1.0, max(0.0, (2.0 - sat_kurt) / 6.0))

    results.append(SignalResult(
        name="Saturation Distribution",
        value=float(sat_kurt),
        score=kurt_score,
        confidence="MEDIUM",
        detail=f"Saturation kurtosis={sat_kurt:.3f}, mean={sat_mean:.1f}, std={sat_std:.1f}. "
               f"{'Low/negative kurtosis: abnormally uniform saturation across image — AI trait.' if kurt_score > 0.5 else 'Peaky saturation distribution consistent with natural lighting.'}",
        category="ai_detection"
    ))

    # RGB channel inter-correlation
    # Real photos: channels correlated by scene illumination (r~0.85-0.97)
    # AI: channel relationships can be artificially decorrelated or over-correlated
    r = np_rgb[:,:,0].ravel()[::20].astype(float)
    g = np_rgb[:,:,1].ravel()[::20].astype(float)
    b = np_rgb[:,:,2].ravel()[::20].astype(float)
    rg_corr = float(np.corrcoef(r, g)[0,1])
    gb_corr = float(np.corrcoef(g, b)[0,1])
    avg_corr = (abs(rg_corr) + abs(gb_corr)) / 2

    # Real photos lit by natural/artificial light: avg corr 0.70-0.98
    # Too low (<0.30): pure synthetic gradient (no shared illumination)
    # Too high (>0.995): perfectly cloned channels (degenerate AI output)
    # Mid range is NORMAL — don't penalize it
    if avg_corr < 0.30:
        corr_score = 0.80   # no inter-channel illumination relationship
    elif avg_corr > 0.995:
        corr_score = 0.70   # suspiciously identical channels
    elif 0.70 <= avg_corr <= 0.98:
        corr_score = 0.05   # healthy natural range
    else:
        corr_score = 0.30   # borderline

    results.append(SignalResult(
        name="RGB Channel Correlation",
        value=float(avg_corr),
        score=corr_score,
        confidence="MEDIUM" if corr_score > 0.5 else "LOW",
        detail=f"Avg channel correlation={avg_corr:.4f} (RG={rg_corr:.3f}, GB={gb_corr:.3f}). "
               f"Natural photos: 0.70-0.98. "
               f"{'Abnormal: no illumination relationship between channels — synthetic.' if avg_corr < 0.30 else ('Channels nearly identical — degenerate output.' if avg_corr > 0.995 else 'Channel correlation within natural illumination range.')}",
        category="ai_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 10 — DCT BLOCK ARTIFACT ANALYSIS
# ─────────────────────────────────────────────

def analyze_dct_artifacts(gray: np.ndarray) -> list[SignalResult]:
    """
    Authentic JPEG photos have characteristic 8x8 DCT block boundaries.
    AI-generated images (especially PNGs or diffusion outputs) lack these.
    Also detects AI spectral grid artifacts from upsampling.
    """
    results = []
    g = gray.astype(np.float32)
    h, w = g.shape

    # Measure boundary vs interior differences at 8-pixel grid
    v_bound = np.abs(g[8::8, :] - g[7:-1:8, :]).mean() if h > 16 else 1.0
    v_inter = np.abs(np.diff(g, axis=0)).mean()
    h_bound = np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean() if w > 16 else 1.0
    h_inter = np.abs(np.diff(g, axis=1)).mean()
    block_ratio = ((v_bound / (v_inter + 1e-8)) + (h_bound / (h_inter + 1e-8))) / 2

    # Real JPEG: ratio > 1.05 (boundary stronger than interior)
    # AI/PNG: ratio ~1.0 (no 8x8 grid structure)
    dct_score = min(1.0, max(0.0, (1.08 - block_ratio) / 0.12))

    results.append(SignalResult(
        name="DCT Block Grid Strength",
        value=float(block_ratio),
        score=dct_score,
        confidence="MEDIUM",
        detail=f"8×8 block boundary ratio={block_ratio:.4f}. "
               f"{'Weak JPEG block structure — PNG/AI output typical signature.' if dct_score > 0.5 else f'Strong JPEG block artifacts (ratio={block_ratio:.3f}) — consistent with camera-captured JPEG.'}",
        category="ai_detection"
    ))

    # FFT periodic artifacts — AI upsampling (bilinear, nearest-neighbor) creates grid
    f = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(f))
    mag[h//2-8:h//2+8, w//2-8:w//2+8] = 0  # mask DC
    threshold = mag.mean() + 5 * mag.std()
    peak_density = float((mag > threshold).sum()) / (h * w)

    # AI upsampling artifacts: higher off-center periodicity
    fft_score = min(1.0, peak_density / 0.005)

    results.append(SignalResult(
        name="FFT Periodicity Artifacts",
        value=float(peak_density),
        score=fft_score,
        confidence="LOW",
        detail=f"Off-center FFT peak density={peak_density:.6f}. "
               f"{'Elevated periodic artifacts in frequency domain — may indicate AI upsampling grid.' if fft_score > 0.4 else 'No significant periodic artifacts in spectrum.'}",
        category="ai_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 11 — FACIAL / REGION COHERENCE CHECK
# ─────────────────────────────────────────────

def analyze_region_coherence(np_rgb: np.ndarray, gray: np.ndarray) -> list[SignalResult]:
    """
    AI portraits often have:
    - Teeth/eyes with unnatural perfection (too-smooth, uniform)
    - Background sharpness inconsistency (everything in focus)
    - Skin region noise different from background noise
    Uses adaptive region analysis without requiring a face detector.
    """
    results = []
    h, w = gray.shape
    g = gray.astype(np.float32) / 255.0

    # Depth-of-field check: real photos have focus falloff
    # Divide image into 9 zones, check sharpness map
    zones_y = np.array_split(g, 3, axis=0)
    zone_sharpness = []
    for zy in zones_y:
        zones_x = np.array_split(zy, 3, axis=1)
        for zone in zones_x:
            lap = cv2.Laplacian((zone * 255).astype(np.uint8), cv2.CV_64F)
            zone_sharpness.append(lap.var())

    zs = np.array(zone_sharpness)
    sharpness_cv = zs.std() / (zs.mean() + 1e-8)

    # DoF only meaningful when image has enough contrast/sharpness range.
    # Wide-angle, macro, or flat scenes legitimately have uniform sharpness.
    # Only flag truly extreme cases (CV < 0.05) with low confidence.
    if sharpness_cv < 0.05:
        dof_score = 0.55
        dof_conf = "LOW"
        dof_detail = f"Zone sharpness CV={sharpness_cv:.3f} — very uniform across all zones. Possible AI (no optical depth), but also valid for wide-angle/flat scenes."
    elif sharpness_cv < 0.15:
        dof_score = 0.25
        dof_conf = "LOW"
        dof_detail = f"Zone sharpness CV={sharpness_cv:.3f} — mildly uniform. Inconclusive."
    else:
        dof_score = 0.05
        dof_conf = "LOW"
        dof_detail = f"Zone sharpness CV={sharpness_cv:.3f} — natural sharpness gradient present."

    results.append(SignalResult(
        name="Depth-of-Field Uniformity",
        value=float(sharpness_cv),
        score=dof_score,
        confidence=dof_conf,
        detail=dof_detail,
        category="ai_detection"
    ))

    return results


# ─────────────────────────────────────────────
# MODULE 12 — GRADIENT DIRECTION ENTROPY
# ─────────────────────────────────────────────

def analyze_gradient_direction(gray: np.ndarray) -> list[SignalResult]:
    """
    Real photos have gradients pointing in many directions (natural scenes).
    AI images often have dominant gradient directions from their synthesis process.
    """
    results = []
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # Only use strong gradients (top 30%) — weak ones are noise
    threshold = np.percentile(mag, 70)
    mask = mag > threshold
    if mask.sum() < 100:
        return results

    angles = np.arctan2(gy[mask], gx[mask])
    hist, _ = np.histogram(angles.ravel(), bins=36,
                           range=(-np.pi, np.pi), density=True)
    hist = hist + 1e-10
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(36)
    norm_entropy = entropy / max_entropy   # 1.0 = all directions equally represented

    # Low entropy = AI (few dominant directions); high = natural photo
    grad_score = min(1.0, max(0.0, (0.75 - norm_entropy) / 0.45))

    results.append(SignalResult(
        name="Gradient Direction Entropy",
        value=float(norm_entropy),
        score=grad_score,
        confidence="MEDIUM",
        detail=f"Gradient direction entropy={norm_entropy:.3f} (max=1.0). "
               f"{'Low entropy — few dominant directions, artificial pattern.' if grad_score > 0.5 else 'High entropy — gradients in all directions, consistent with natural scene.'}",
        category="ai_detection"
    ))
    return results


# ─────────────────────────────────────────────
# MODULE 13 — COLOR CHANNEL UNIFORMITY
# ─────────────────────────────────────────────

def analyze_channel_uniformity(np_rgb: np.ndarray) -> list[SignalResult]:
    """
    Real photos lit by natural/artificial light have channels with different
    mean intensities (color casts). AI images are often unnaturally balanced.
    """
    results = []
    means = np.array([np_rgb[:,:,c].mean() for c in range(3)])
    stds  = np.array([np_rgb[:,:,c].std()  for c in range(3)])

    mean_cv = float(np.std(means) / (np.mean(means) + 1e-8))
    std_cv  = float(np.std(stds)  / (np.mean(stds)  + 1e-8))

    # Real photos: channels differ meaningfully (mean_cv > 0.12)
    # AI: often balanced (mean_cv < 0.08)
    if mean_cv < 0.05:
        uniformity_score = 0.85
    elif mean_cv < 0.10:
        uniformity_score = 0.55
    elif mean_cv < 0.18:
        uniformity_score = 0.20
    else:
        uniformity_score = 0.05   # strongly different channels = natural

    results.append(SignalResult(
        name="Channel Mean Uniformity",
        value=float(mean_cv),
        score=uniformity_score,
        confidence="MEDIUM",
        detail=f"Channel mean CV={mean_cv:.4f}, std CV={std_cv:.4f}. "
               f"{'Channels are suspiciously balanced — AI images lack natural color casts.' if uniformity_score > 0.5 else 'Natural inter-channel variation from real lighting conditions.'}",
        category="ai_detection"
    ))
    return results


# ─────────────────────────────────────────────
# MODULE 14 — JPEG QUANTIZATION FINGERPRINT
# ─────────────────────────────────────────────

def analyze_jpeg_quantization(pil_img: Image.Image) -> list[SignalResult]:
    """
    Real cameras embed device-specific JPEG quantization tables.
    AI images saved as JPEG use generic/default tables.
    PNG AI outputs have no quantization tables at all.
    """
    results = []
    try:
        qtables = getattr(pil_img, 'quantization', None)
        if not qtables:
            # No quantization table = PNG or stripped JPEG — common in AI outputs
            results.append(SignalResult(
                name="JPEG Quantization Table",
                value=0.0,
                score=0.65,
                confidence="MEDIUM",
                detail="No JPEG quantization table found. PNG format or stripped metadata — common in AI-generated outputs.",
                category="ai_detection"
            ))
            return results

        t0 = np.array(qtables[0] if isinstance(qtables[0], (list, np.ndarray))
                      else list(qtables[0].values()), dtype=float)

        # Camera-specific tables have non-uniform, irregular values
        # Generic/default tables have smooth, regular patterns
        variance    = float(t0.var())
        sorted_diffs = np.diff(np.sort(t0))
        regularity  = float(1.0 - min(1.0, sorted_diffs.std() / (sorted_diffs.mean() + 1e-8) / 3.0))

        # Low variance + high regularity = generic table = AI/edited
        if variance < 200 and regularity > 0.7:
            qt_score = 0.70
            qt_detail = f"Generic/default quantization table (var={variance:.0f}, regularity={regularity:.2f}) — typical of AI outputs or image editors, not camera capture."
        elif variance > 800:
            qt_score = 0.10
            qt_detail = f"Device-specific quantization table (var={variance:.0f}) — consistent with direct camera capture."
        else:
            qt_score = 0.35
            qt_detail = f"Quantization table variance={variance:.0f}. Moderate — may be camera or editor."

        results.append(SignalResult(
            name="JPEG Quantization Table",
            value=float(variance),
            score=qt_score,
            confidence="MEDIUM",
            detail=qt_detail,
            category="ai_detection"
        ))
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────
# MODULE 15 — HOG FEATURE STATISTICS
# ─────────────────────────────────────────────

def analyze_hog_statistics(gray: np.ndarray) -> list[SignalResult]:
    """
    HOG (Histogram of Oriented Gradients) captures mid-level structure.
    AI images show characteristic HOG skew/kurtosis patterns distinct from photos.
    """
    from skimage.feature import hog as skimage_hog
    results = []
    try:
        resized = cv2.resize(gray, (128, 128))
        fd = skimage_hog(resized, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False)
        hog_mean = float(fd.mean())
        hog_std  = float(fd.std())
        hog_kurt = float(stats.kurtosis(fd))
        hog_skew = float(stats.skew(fd))

        # Real photos: HOG skew typically negative (−0.8 to −0.2)
        # AI images: HOG skew often positive or near zero (different structure distribution)
        skew_score = min(1.0, max(0.0, (hog_skew + 0.1) / 1.2))

        results.append(SignalResult(
            name="HOG Distribution Skew",
            value=float(hog_skew),
            score=skew_score,
            confidence="MEDIUM",
            detail=f"HOG skew={hog_skew:.3f}, kurt={hog_kurt:.3f}. "
                   f"{'Positive/zero skew — gradient structure atypical for natural scenes.' if skew_score > 0.5 else 'Negative skew — gradient distribution consistent with real photography.'}",
            category="ai_detection"
        ))
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────
# MODULE 16 — FFT AZIMUTHAL BAND ANALYSIS
# ─────────────────────────────────────────────

def analyze_fft_bands(gray: np.ndarray) -> list[SignalResult]:
    """
    AI generators (diffusion, GAN) create characteristic energy distributions
    across spatial frequency bands due to their upsampling architectures.
    Bilinear/nearest upsampling causes periodic energy spikes at mid/high bands.
    """
    results = []
    h, w = gray.shape
    f = np.abs(fftshift(fft2(gray.astype(float))))
    f[h//2-3:h//2+3, w//2-3:w//2+3] = 0   # mask DC component
    f = f / (f.max() + 1e-8)

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    max_r = min(h, w) // 2

    def band_mean(lo, hi):
        mask = (r > lo * max_r) & (r < hi * max_r)
        return float(f[mask].mean()) if mask.any() else 0.0

    low   = band_mean(0.05, 0.15)
    mid   = band_mean(0.25, 0.35)
    high  = band_mean(0.45, 0.55)

    mid_ratio  = mid  / (low + 1e-8)
    high_ratio = high / (low + 1e-8)

    # Real photos: energy falls off smoothly (mid/low ~0.5-0.8)
    # AI upsampling: mid and high bands have less relative energy (ratio < 0.3)
    # OR spike above 1.2 depending on model
    band_score = min(1.0, max(0.0, (0.45 - mid_ratio) / 0.40))

    results.append(SignalResult(
        name="FFT Band Energy Ratio",
        value=float(mid_ratio),
        score=band_score,
        confidence="MEDIUM",
        detail=f"Spectral band ratios — mid/low={mid_ratio:.3f}, high/low={high_ratio:.3f}. "
               f"Natural falloff range: 0.45-0.85. "
               f"{'Abnormal spectral energy distribution — AI upsampling architecture fingerprint.' if band_score > 0.5 else 'Normal spectral falloff consistent with optical imaging.'}",
        category="ai_detection"
    ))
    return results


# ─────────────────────────────────────────────
# MODULE 17 — MULTI-SCALE NOISE PROFILE
# ─────────────────────────────────────────────

def analyze_multiscale_noise(gray: np.ndarray) -> list[SignalResult]:
    """
    Real camera noise follows a consistent power-law decay across scales.
    AI images have unnaturally flat or irregular noise across scales.
    """
    results = []
    current = gray.astype(np.float32)
    scale_noise = []

    for _ in range(4):
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        noise = current - blurred
        scale_noise.append(float(noise.std()))
        new_h = max(current.shape[0] // 2, 16)
        new_w = max(current.shape[1] // 2, 16)
        current = cv2.resize(current, (new_w, new_h))
        if current.shape[0] <= 16 or current.shape[1] <= 16:
            break

    if len(scale_noise) < 2:
        return results

    # Scale ratio: real cameras have fine noise >> coarse noise (ratio > 3.0)
    # AI images: noise flat or irregular across scales (ratio < 2.0)
    scale_ratio = scale_noise[0] / (scale_noise[-1] + 1e-8)

    ms_score = min(1.0, max(0.0, (2.5 - scale_ratio) / 2.2))

    # Also check that noise decreases monotonically (camera property)
    diffs = np.diff(scale_noise)
    non_monotonic = (diffs > 0).sum()   # should be 0 for real camera
    monotonic_penalty = min(0.3, non_monotonic * 0.1)

    ms_score = min(1.0, ms_score + monotonic_penalty)

    results.append(SignalResult(
        name="Multi-Scale Noise Profile",
        value=float(scale_ratio),
        score=ms_score,
        confidence="HIGH" if abs(scale_ratio - 2.5) > 1.5 else "MEDIUM",
        detail=f"Noise σ across scales: {[f'{s:.3f}' for s in scale_noise]}. "
               f"Scale ratio={scale_ratio:.2f} (real cameras >3.0). "
               f"{'Flat/weak noise decay — AI-generated or synthetic image.' if ms_score > 0.5 else 'Consistent noise falloff across scales — real camera sensor pattern.'}",
        category="ai_detection"
    ))
    return results

# ─────────────────────────────────────────────
# AGGREGATOR
# ─────────────────────────────────────────────

# Per-signal reliability weights — calibrated by empirical signal quality
# Higher = more reliable discriminator for that signal
SIGNAL_WEIGHTS = {
    # AI Detection signals (ordered by reliability)
    "GLCM Texture Homogeneity":      2.5,   # very reliable: AI over-smooths texture
    "Local Variance Uniformity":     2.0,   # AI has unnaturally uniform detail
    "Noise Level (PRNU)":            2.0,   # AI is too clean or too noisy
    "Chromatic Aberration":          1.5,   # only reliable when enough edges present
    "Saturation Distribution":       1.5,   # AI has unnatural saturation spread
    "RGB Channel Correlation":       1.5,   # real lighting physics constrains this
    "Depth-of-Field Uniformity":     0.6,   # weak: many legit scenes are flat
    "Texture Complexity (LBP Entropy)": 1.2,
    "DCT Block Grid Strength":       1.2,   # JPEG-vs-PNG structural evidence
    "Spectral Power Slope":          0.8,  # weak on non-textured images
    "Noise Distribution (Kurtosis)": 0.8,
    "FFT Periodicity Artifacts":     0.7,
    "High-Frequency Energy":         0.5,   # unreliable alone
    "Laplacian Sharpness":           0.4,
    # Edit Detection signals
    "Error Level Analysis (ELA)":    2.5,   # gold standard for edits
    "Noise Spatial Inconsistency":   2.0,   # regional edit signature
    "JPEG Ghost (Compression History)": 1.5,
    "Clone / Copy-Move Detection":   2.0,
    "Timestamp Mismatch":            1.8,
    # Metadata
    "EXIF / Metadata":               1.5,
    "File Format":                   0.5,
    "Timestamp Consistency":         0.3,
    # New signals v2
    "Gradient Direction Entropy":    1.8,
    "Channel Mean Uniformity":       1.5,
    "JPEG Quantization Table":       1.6,
    # Advanced signals v3
    "HOG Distribution Skew":         2.0,   # mid-level structure highly discriminative
    "FFT Band Energy Ratio":         2.0,   # AI upsampling architecture fingerprint
    "Multi-Scale Noise Profile":     2.5,   # camera noise physics vs synthetic
}

def _load_learned_weights() -> dict:
    """
    Auto-load weights trained by image_trainer.py if they exist.
    Falls back to hardcoded SIGNAL_WEIGHTS if not found.
    """
    weights_path = Path(__file__).parent / "model" / "learned_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                data = json.load(f)
            learned = data.get("weights", {})
            if learned:
                return learned
        except Exception:
            pass
    return SIGNAL_WEIGHTS


def aggregate_scores(signals: list[SignalResult]) -> tuple[float, float, float]:
    """
    Weighted average using per-signal reliability weights + confidence multiplier.
    Applies soft voting: no single signal dominates; convergence of evidence matters.
    Auto-uses learned weights from image_trainer.py when available.
    """
    # Use learned weights if trainer has run, else fall back to hardcoded
    active_weights = _load_learned_weights()
    conf_mult = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.45}

    ai_signals   = [s for s in signals if s.category == "ai_detection"]
    edit_signals = [s for s in signals if s.category == "edit_detection"]
    meta_signals = [s for s in signals if s.category == "metadata"]

    def weighted_avg(sigs):
        if not sigs:
            return 0.0
        total_w, total_v = 0.0, 0.0
        for s in sigs:
            base_w  = active_weights.get(s.name, 1.0)
            conf_w  = conf_mult[s.confidence]
            w       = base_w * conf_w
            total_w += w
            total_v += s.score * w
        return total_v / total_w if total_w > 0 else 0.0

    ai_prob   = weighted_avg(ai_signals)
    edit_prob = weighted_avg(edit_signals)

    # Metadata can nudge AI probability
    meta_nudge = 0.0
    for s in meta_signals:
        if s.name == "EXIF / Metadata":
            meta_nudge = (s.score - 0.5) * 0.25   # missing EXIF → +0.125 AI

    ai_prob = max(0.0, min(1.0, ai_prob + meta_nudge))

    # Convergence bonus: if many independent signals agree, boost confidence
    ai_high = sum(1 for s in ai_signals if s.score > 0.65)
    if ai_high >= 4:
        ai_prob = min(1.0, ai_prob + 0.08)   # strong multi-signal agreement

    edit_high = sum(1 for s in edit_signals if s.score > 0.55)
    if edit_high >= 2:
        edit_prob = min(1.0, edit_prob + 0.08)

    authenticity = 1.0 - max(ai_prob, edit_prob * 0.75)
    authenticity = max(0.0, min(1.0, authenticity))

    return round(ai_prob, 3), round(edit_prob, 3), round(authenticity, 3)


def build_verdict(ai_prob, edit_prob, auth_score, meta):
    lines = []
    if ai_prob > 0.65:
        lines.append(f"⚠️  LIKELY AI-GENERATED (AI probability: {ai_prob*100:.0f}%)")
    elif ai_prob > 0.40:
        lines.append(f"🔍  POSSIBLY AI-GENERATED (AI probability: {ai_prob*100:.0f}%)")
    else:
        lines.append(f"✅  LIKELY AUTHENTIC PHOTOGRAPH (AI probability: {ai_prob*100:.0f}%)")

    if edit_prob > 0.60:
        lines.append(f"✂️  SIGNIFICANT EDITING DETECTED (edit probability: {edit_prob*100:.0f}%)")
    elif edit_prob > 0.35:
        lines.append(f"🖊️  MINOR EDITING POSSIBLE (edit probability: {edit_prob*100:.0f}%)")
    else:
        lines.append(f"✅  NO SIGNIFICANT EDITS DETECTED (edit probability: {edit_prob*100:.0f}%)")

    if meta.get('has_camera'):
        lines.append(f"📷  Camera metadata present: {meta.get('exif',{}).get('Make','')} {meta.get('exif',{}).get('Model','')}")
    else:
        lines.append("❓  No camera hardware metadata (stripped or synthetic)")

    lines.append(f"\n   Authenticity Score: {auth_score*100:.0f}/100")
    return lines


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize(pil_img, gray, ela_map, noise_residual, freq_magnitude,
              signals, ai_prob, edit_prob, auth_score, ghost_data, output_path):

    ghost_diffs, ghost_qualities = ghost_data

    fig = plt.figure(figsize=(20, 14), facecolor='#0a0a0f')
    fig.patch.set_facecolor('#0a0a0f')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.04, right=0.96,
                           top=0.92, bottom=0.06)

    ax_color = {'bg': '#0a0a0f', 'surface': '#111118',
                'text': '#e8e8f0', 'muted': '#7a7a95',
                'green': '#00e5a0', 'orange': '#ff6b35',
                'purple': '#7c5cfc', 'yellow': '#ffcc00',
                'red': '#ff4757'}

    def style_ax(ax, title=""):
        ax.set_facecolor(ax_color['surface'])
        ax.tick_params(colors=ax_color['muted'], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2a2a3d')
        if title:
            ax.set_title(title, color=ax_color['text'], fontsize=9,
                         fontweight='bold', pad=8)

    # ── Row 0: Images ──────────────────────────────────────────────

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(pil_img)
    style_ax(ax0, "Original Image")
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ela_disp = np.clip(ela_map * 10, 0, 255).astype(np.uint8) if ela_map.max() <= 25.5 else ela_map
    ax1.imshow(ela_disp, cmap='hot')
    style_ax(ax1, "ELA Map (Bright = Edited)")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    noise_disp = (noise_residual - noise_residual.min()) / (noise_residual.max() - noise_residual.min() + 1e-8)
    ax2.imshow(noise_disp, cmap='RdBu_r')
    style_ax(ax2, "Noise Residual")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(freq_magnitude, cmap='inferno')
    style_ax(ax3, "Frequency Spectrum (FFT)")
    ax3.axis('off')

    # ── Row 1: Signal bars ─────────────────────────────────────────

    ax_signals = fig.add_subplot(gs[1, :3])
    style_ax(ax_signals, "Detection Signals — Score (0=Photo, 1=AI/Edited)")

    names  = [s.name for s in signals]
    scores = [s.score for s in signals]
    cats   = [s.category for s in signals]
    y_pos  = range(len(names))

    colors = []
    for s in signals:
        if s.category == 'ai_detection':
            colors.append(ax_color['purple'] if s.score > 0.5 else ax_color['green'])
        elif s.category == 'edit_detection':
            colors.append(ax_color['orange'] if s.score > 0.5 else ax_color['green'])
        else:
            colors.append(ax_color['yellow'] if s.score > 0.5 else ax_color['green'])

    bars = ax_signals.barh(list(y_pos), scores, color=colors, height=0.6, alpha=0.85)
    ax_signals.set_yticks(list(y_pos))
    ax_signals.set_yticklabels(names, color=ax_color['text'], fontsize=8)
    ax_signals.set_xlim(0, 1)
    ax_signals.axvline(0.5, color=ax_color['muted'], linestyle='--', alpha=0.4, lw=1)
    ax_signals.set_xlabel("Score (0 = authentic, 1 = suspicious)", color=ax_color['muted'], fontsize=8)

    # Annotate scores
    for bar, score in zip(bars, scores):
        ax_signals.text(min(score + 0.02, 0.97), bar.get_y() + bar.get_height()/2,
                        f'{score:.2f}', va='center', color=ax_color['text'], fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=ax_color['purple'], label='AI Detection'),
                  Patch(facecolor=ax_color['orange'], label='Edit Detection'),
                  Patch(facecolor=ax_color['yellow'], label='Metadata')]
    ax_signals.legend(handles=legend_els, loc='lower right',
                      facecolor=ax_color['surface'], edgecolor='#2a2a3d',
                      labelcolor=ax_color['text'], fontsize=7)

    # ── Row 1 Col 3: Verdict gauge ─────────────────────────────────

    ax_gauge = fig.add_subplot(gs[1, 3])
    style_ax(ax_gauge, "Authenticity Score")

    theta = np.linspace(0, np.pi, 200)
    # Background arc
    ax_gauge.plot(np.cos(theta), np.sin(theta), color='#2a2a3d', lw=12, solid_capstyle='round')
    # Score arc
    score_theta = np.linspace(0, np.pi * auth_score, 200)
    score_color = ax_color['green'] if auth_score > 0.6 else \
                  ax_color['yellow'] if auth_score > 0.35 else ax_color['red']
    ax_gauge.plot(np.cos(score_theta), np.sin(score_theta), color=score_color, lw=12,
                  solid_capstyle='round')
    ax_gauge.text(0, 0.2, f"{auth_score*100:.0f}", ha='center', va='center',
                  color=ax_color['text'], fontsize=32, fontweight='bold')
    ax_gauge.text(0, -0.15, '/100', ha='center', color=ax_color['muted'], fontsize=11)
    ax_gauge.text(0, -0.4, 'Authenticity', ha='center', color=ax_color['muted'], fontsize=9)
    ax_gauge.set_xlim(-1.3, 1.3); ax_gauge.set_ylim(-0.6, 1.3)
    ax_gauge.axis('off')

    # ── Row 2: Ghost curve + probability bars ──────────────────────

    ax_ghost = fig.add_subplot(gs[2, 0:2])
    style_ax(ax_ghost, "JPEG Ghost — Compression History")
    ax_ghost.plot(ghost_qualities, ghost_diffs, 'o-',
                  color=ax_color['purple'], lw=2, ms=6, markerfacecolor=ax_color['purple'])
    ax_ghost.fill_between(ghost_qualities, ghost_diffs, alpha=0.15, color=ax_color['purple'])
    min_idx = int(np.argmin(ghost_diffs))
    ax_ghost.axvline(ghost_qualities[min_idx], color=ax_color['yellow'],
                     linestyle='--', alpha=0.7, lw=1.5, label=f'Min @ Q={ghost_qualities[min_idx]}')
    ax_ghost.set_xlabel("Re-compression Quality", color=ax_color['muted'], fontsize=8)
    ax_ghost.set_ylabel("Mean Diff from Original", color=ax_color['muted'], fontsize=8)
    ax_ghost.legend(facecolor=ax_color['surface'], edgecolor='#2a2a3d',
                    labelcolor=ax_color['text'], fontsize=7)

    # ── Row 2 Col 2-3: Final probability display ───────────────────

    ax_prob = fig.add_subplot(gs[2, 2:])
    style_ax(ax_prob, "Final Probabilities")
    ax_prob.axis('off')

    labels = ['AI-Generated', 'Edited/Tampered', 'Authentic Photo']
    probs  = [ai_prob, edit_prob, auth_score]
    colors_p = [ax_color['purple'], ax_color['orange'], ax_color['green']]
    for i, (lbl, prob, clr) in enumerate(zip(labels, probs, colors_p)):
        y = 0.75 - i * 0.3
        ax_prob.text(0.0, y + 0.08, lbl, transform=ax_prob.transAxes,
                     color=ax_color['muted'], fontsize=9)
        # Background bar
        ax_prob.barh([y], [1.0], left=[0], height=0.12, color='#2a2a3d',
                     transform=ax_prob.transAxes)
        ax_prob.barh([y], [prob], left=[0], height=0.12, color=clr, alpha=0.85,
                     transform=ax_prob.transAxes)
        ax_prob.text(prob + 0.01 if prob < 0.9 else prob - 0.05, y + 0.01,
                     f"{prob*100:.0f}%", transform=ax_prob.transAxes,
                     color=ax_color['text'], fontsize=11, fontweight='bold',
                     va='center')

    # ── Title ───────────────────────────────────────────────────────

    fname = Path(output_path).stem.replace('_analysis', '')
    fig.text(0.5, 0.97, f"Image Authenticity Report — {fname}",
             ha='center', color=ax_color['text'], fontsize=13, fontweight='bold')
    fig.text(0.5, 0.95, "Signals: Frequency · Noise · ELA · Clone Detection · Metadata",
             ha='center', color=ax_color['muted'], fontsize=9)

    plt.savefig(output_path, facecolor='#0a0a0f', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Visual report saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN ANALYSIS RUNNER
# ─────────────────────────────────────────────

def analyze(filepath: str, verbose: bool = False, save_report: bool = True) -> AnalysisReport:
    print(f"\n{'='*60}")
    print(f"  Image Authenticity Analyzer")
    print(f"  File: {filepath}")
    print(f"{'='*60}")

    # Create outputs directory next to the script (works on Windows & Linux)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(filepath):
        print(f"  ❌  File not found: {filepath}")
        sys.exit(1)

    pil_img, np_rgb, np_bgr, gray = load_image(filepath)
    h, w = gray.shape
    print(f"  Size: {w}×{h} px | Format: {pil_img.format}")

    report = AnalysisReport(filepath=filepath, image_size=(w, h))

    # ── Run all modules ───────────────────────────────────────────

    print("\n  [1/7] Metadata Analysis...")
    meta_signals, meta_info = analyze_metadata(pil_img, filepath)
    report.signals.extend(meta_signals)
    report.metadata = meta_info

    print("  [2/7] Frequency Domain Analysis...")
    freq_signals, freq_magnitude = analyze_frequency(gray)
    report.signals.extend(freq_signals)

    print("  [3/7] Noise / PRNU Analysis...")
    noise_signals, noise_residual = analyze_noise(gray)
    report.signals.extend(noise_signals)

    print("  [4/7] Error Level Analysis (ELA)...")
    ela_signals, ela_map = analyze_ela(pil_img)
    report.signals.extend(ela_signals)

    print("  [5/7] Texture & Chromatic Aberration...")
    texture_signals = analyze_texture(gray, np_rgb)
    report.signals.extend(texture_signals)

    print("  [6/7] Clone / Copy-Move Detection...")
    clone_signals = analyze_clone_detection(gray)
    report.signals.extend(clone_signals)

    print("  [7/7] JPEG Compression History (GHOST)...")
    ghost_signals, ghost_diffs, ghost_qualities = analyze_compression_history(pil_img)
    report.signals.extend(ghost_signals)

    print("  [8/11] GLCM Texture Analysis...")
    glcm_signals = analyze_glcm_texture(gray)
    report.signals.extend(glcm_signals)

    print("  [9/11] Color Statistics...")
    color_signals = analyze_color_statistics(np_rgb)
    report.signals.extend(color_signals)

    print("  [10/11] DCT Block Artifacts...")
    dct_signals = analyze_dct_artifacts(gray)
    report.signals.extend(dct_signals)

    print("  [11/11] Region Coherence & Depth-of-Field...")
    region_signals = analyze_region_coherence(np_rgb, gray)
    report.signals.extend(region_signals)

    print("  [12/14] Gradient Direction Entropy...")
    grad_signals = analyze_gradient_direction(gray)
    report.signals.extend(grad_signals)

    print("  [13/14] Channel Uniformity...")
    chan_signals = analyze_channel_uniformity(np_rgb)
    report.signals.extend(chan_signals)

    print("  [14/14] JPEG Quantization Fingerprint...")
    qt_signals = analyze_jpeg_quantization(pil_img)
    report.signals.extend(qt_signals)

    print("  [15/17] HOG Distribution Analysis...")
    hog_signals = analyze_hog_statistics(gray)
    report.signals.extend(hog_signals)

    print("  [16/17] FFT Azimuthal Band Analysis...")
    band_signals = analyze_fft_bands(gray)
    report.signals.extend(band_signals)

    print("  [17/17] Multi-Scale Noise Profile...")
    ms_signals = analyze_multiscale_noise(gray)
    report.signals.extend(ms_signals)

    # ── Aggregate ─────────────────────────────────────────────────

    report.ai_probability, report.edit_probability, report.authenticity_score = \
        aggregate_scores(report.signals)

    report.verdict = build_verdict(
        report.ai_probability, report.edit_probability,
        report.authenticity_score, meta_info
    )

    # ── Print Report ──────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print("  VERDICT")
    print(f"{'─'*60}")
    for line in report.verdict:
        print(f"  {line}")

    print(f"\n{'─'*60}")
    print("  SIGNAL BREAKDOWN")
    print(f"{'─'*60}")

    categories = {'ai_detection': '🤖 AI Detection',
                  'edit_detection': '✂️  Edit Detection',
                  'metadata': '📋 Metadata'}

    for cat_key, cat_label in categories.items():
        cat_signals = [s for s in report.signals if s.category == cat_key]
        if not cat_signals:
            continue
        print(f"\n  {cat_label}")
        for s in cat_signals:
            bar_len = int(s.score * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            flag = '⚠' if s.score > 0.6 else ('~' if s.score > 0.35 else '✓')
            w = SIGNAL_WEIGHTS.get(s.name, 1.0)
            wt = f"w={w:.1f}"
            print(f"   {flag} {s.name:<36} [{bar}] {s.score:.2f}  {wt:<6} [{s.confidence}]")
            if verbose:
                print(f"       → {s.detail}")

    if not verbose:
        print("\n  (run with --verbose for full signal explanations)")

    # ── Visualization ─────────────────────────────────────────────

    if save_report:
        base = os.path.splitext(filepath)[0]
        vis_path = os.path.join(output_dir, f"{Path(filepath).stem}_analysis.png")
        visualize(pil_img, gray, ela_map, noise_residual, freq_magnitude,
                  report.signals, report.ai_probability, report.edit_probability,
                  report.authenticity_score, (ghost_diffs, ghost_qualities), vis_path)

    # ── JSON dump ─────────────────────────────────────────────────

    if save_report:
        json_path = os.path.join(output_dir, f"{Path(filepath).stem}_report.json")
        json_report = {
            "filepath": filepath,
            "size": list(report.image_size),
            "ai_probability": report.ai_probability,
            "edit_probability": report.edit_probability,
            "authenticity_score": report.authenticity_score,
            "verdict": report.verdict,
            "signals": [
                {"name": s.name, "score": s.score, "confidence": s.confidence,
                 "category": s.category, "detail": s.detail}
                for s in report.signals
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
        print(f"  📄 JSON report saved → {json_path}")

    print(f"\n{'='*60}\n")
    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze image authenticity: GenAI detection, edit forensics, provenance.")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to image file (optional — file picker opens if omitted)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full signal explanations")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Skip saving the visual report and JSON")
    args = parser.parse_args()
    # save_report is ON by default; only skip if --no-save is passed
    args.save_report = not args.no_save

    image_path = args.image

    # No path given → open a native file-picker dialog
    if not image_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()          # hide the empty Tk window
            root.attributes("-topmost", True)
            image_path = filedialog.askopenfilename(
                title="Select an image to analyze",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.webp *.tiff *.tif *.bmp *.gif"),
                    ("All files", "*.*"),
                ]
            )
            root.destroy()
        except Exception:
            image_path = None

        if not image_path:
            print("\n  No image selected. Usage:")
            print("      python image_authenticity.py path/to/image.jpg")
            print("      python image_authenticity.py path/to/image.jpg --verbose\n")
            sys.exit(0)

    analyze(image_path, verbose=args.verbose, save_report=args.save_report)