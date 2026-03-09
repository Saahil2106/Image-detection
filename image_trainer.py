"""
Image Authenticity Trainer
===========================
Trains a machine learning model on labeled images using signals extracted
by image_authenticity.py. The trained model is saved and automatically
picked up by the analyzer to improve its verdicts.

Architecture:
  image_authenticity.py  --extract signals--  image_trainer.py
                                                      train & evaluate
                                                     
                                             model/  (saved weights + calibration)
                                                      load at runtime
  image_authenticity.py  --apply learned weights--

Workflow:
  1. Collect images into labeled folders:
       dataset/
         real/         authentic photos
         ai/           AI-generated images
         edited/       tampered/edited photos  (optional)

  2. Train:
       python image_trainer.py --train --dataset dataset/

  3. Evaluate a single image (uses trained model):
       python image_trainer.py --predict photo.jpg

  4. Batch evaluate a folder:
       python image_trainer.py --predict-dir dataset/test/

  5. Run together with analyzer (full pipeline):
       python image_trainer.py --analyze photo.jpg
"""

# ---------------------------------------------
# AUTO-INSTALL
# ---------------------------------------------
import sys
import subprocess

REQUIRED = {
    "numpy": "numpy", "PIL": "pillow", "cv2": "opencv-python",
    "scipy": "scipy", "skimage": "scikit-image",
    "sklearn": "scikit-learn", "matplotlib": "matplotlib",
}

def _ensure_deps():
    import importlib.util as ilu
    missing = [pip for imp, pip in REQUIRED.items()
               if not ilu.find_spec(imp)]
    if missing:
        print(f"\n  [PKG]  Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
        print("  [OK]  Done.\n")

_ensure_deps()

# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
import os
import math
import json
import warnings
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ---------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
MODEL_DIR    = SCRIPT_DIR / "model"
OUTPUT_DIR   = SCRIPT_DIR / "outputs"
DATASET_FILE = MODEL_DIR / "dataset.json"
MODEL_FILE     = MODEL_DIR / "classifier.joblib"       # RF on handcrafted features
CNN_MODEL_FILE = MODEL_DIR / "cnn_classifier.joblib"   # LR on CNN features
SCALER_FILE    = MODEL_DIR / "scaler.joblib"
META_FILE      = MODEL_DIR / "meta.json"
CNN_META_FILE  = MODEL_DIR / "cnn_meta.json"
WEIGHTS_FILE   = MODEL_DIR / "learned_weights.json"  # fed back to analyzer

# Active model slot -- can be overridden by --model flag at runtime
# Slots let you keep multiple trained models and switch between them:
#   default   -> model/classifier.joblib  (current behavior)
#   genimage  -> model/genimage/classifier.joblib
#   cifake    -> model/cifake/classifier.joblib
#   combined  -> model/combined/classifier.joblib
_ACTIVE_SLOT = "default"

def set_model_slot(slot: str):
    """Switch all model file paths to a named slot."""
    global _ACTIVE_SLOT, DATASET_FILE, MODEL_FILE, CNN_MODEL_FILE
    global SCALER_FILE, META_FILE, CNN_META_FILE, WEIGHTS_FILE

    _ACTIVE_SLOT = slot
    if slot == "default":
        slot_dir = MODEL_DIR
    else:
        slot_dir = MODEL_DIR / slot
        slot_dir.mkdir(parents=True, exist_ok=True)

    DATASET_FILE   = slot_dir / "dataset.json"
    MODEL_FILE     = slot_dir / "classifier.joblib"
    CNN_MODEL_FILE = slot_dir / "cnn_classifier.joblib"
    SCALER_FILE    = slot_dir / "scaler.joblib"
    META_FILE      = slot_dir / "meta.json"
    CNN_META_FILE  = slot_dir / "cnn_meta.json"
    WEIGHTS_FILE   = slot_dir / "learned_weights.json"
    print(f"  [MODEL]  Active slot: '{slot}' -> {slot_dir}")

def list_model_slots() -> list:
    """Return all available trained model slots."""
    slots = []
    # Check default slot
    if (MODEL_DIR / "classifier.joblib").exists():
        slots.append(("default", MODEL_DIR))
    # Check named subslots
    for d in sorted(MODEL_DIR.iterdir()):
        if d.is_dir() and (d / "classifier.joblib").exists():
            slots.append((d.name, d))
    return slots

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Labels
LABEL_REAL   = 0
LABEL_AI     = 1
LABEL_EDITED = 2
LABEL_NAMES  = {0: "real", 1: "ai", 2: "edited"}
LABEL_FROM_STR = {"real": 0, "authentic": 0, "photo": 0,
                  "realart": 0, "real_art": 0, "realphoto": 0,
                  "nature": 0, "natural": 0, "imagenet": 0,    # GenImage real folder names
                  "ai": 1, "generated": 1, "synthetic": 1, "fake": 1,
                  "aiartdata": 1, "ai_art": 1, "aiart": 1, "aidata": 1,
                  "edited": 2, "tampered": 2, "manipulated": 2}

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp'}

# All signal names in fixed order (feature vector definition)
SIGNAL_NAMES = [
    "Spectral Power Slope",
    "High-Frequency Energy",
    "Noise Level (PRNU)",
    "Noise Distribution (Kurtosis)",
    "Noise Spatial Inconsistency",
    "Texture Complexity (LBP Entropy)",
    "Chromatic Aberration",
    "Laplacian Sharpness",
    "GLCM Texture Homogeneity",
    "Local Variance Uniformity",
    "Saturation Distribution",
    "RGB Channel Correlation",
    "DCT Block Grid Strength",
    "FFT Periodicity Artifacts",
    "Depth-of-Field Uniformity",
    "Error Level Analysis (ELA)",
    "Clone / Copy-Move Detection",
    "JPEG Ghost (Compression History)",
    "EXIF / Metadata",
    "File Format",
    # New signals (v2)
    "Gradient Direction Entropy",
    "Channel Mean Uniformity",
    "JPEG Quantization Table",
    # Advanced signals (v3)
    "HOG Distribution Skew",
    "FFT Band Energy Ratio",
    "Multi-Scale Noise Profile",
]


# ---------------------------------------------
# RICH FEATURE EXTRACTOR (CNN-alternative)
# Uses multi-scale HOG + Gabor filters + spatial LBP + color moments
# Produces ~262 features  much more discriminative than 34 signal scores
# ---------------------------------------------

def extract_rich_cnn_features(gray: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Extract ~262 rich spatial features without requiring PyTorch/TF.
    Combines multi-scale HOG, Gabor filter bank, spatial LBP pyramid,
    and color moment grid  approximating lower CNN layer activations.
    """
    from skimage.feature import hog as skimage_hog, local_binary_pattern
    features = []

    target     = cv2.resize(gray.astype(np.uint8), (128, 128))
    target_rgb = cv2.resize(rgb.astype(np.uint8),  (128, 128)).astype(np.float32)

    # 1. Multi-scale HOG  captures local edge structure at 3 resolutions
    for scale in [128, 64, 32]:
        g = cv2.resize(target, (scale, scale))
        cells = max(2, scale // 16)
        ppc   = scale // cells
        try:
            fd = skimage_hog(g, orientations=9,
                             pixels_per_cell=(ppc, ppc),
                             cells_per_block=(2, 2), visualize=False)
            fd_r = fd.reshape(-1, 9) if len(fd) >= 9 else fd.reshape(1, -1)
            features.append(fd_r.mean(axis=0))   # 9 values
            features.append(fd_r.std(axis=0))    # 9 values
        except Exception:
            features.append(np.zeros(18))

    # 2. Gabor filter bank  frequency + orientation tuned (real texture analysis)
    for freq in [0.1, 0.25, 0.4]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            try:
                kern     = cv2.getGaborKernel((21, 21), 3.0, theta, 1.0/freq, 0.5, 0)
                response = cv2.filter2D(target.astype(np.float32), -1, kern)
                features.append(np.array([response.mean(), response.std()]))
            except Exception:
                features.append(np.zeros(2))

    # 3. Spatial LBP pyramid  texture in 4 quadrants
    try:
        lbp = local_binary_pattern(target, P=8, R=1, method='uniform')
        h2, w2 = target.shape[0]//2, target.shape[1]//2
        for region in [lbp[:h2, :w2], lbp[:h2, w2:], lbp[h2:, :w2], lbp[h2:, w2:]]:
            hist, _ = np.histogram(region.ravel(), bins=10, range=(0, 10), density=True)
            features.append(hist)
    except Exception:
        features.append(np.zeros(40))

    # 4. Color moment grid  4x4 spatial grid, 3 channels, 3 moments
    try:
        ch_  = target_rgb.shape[0] // 4
        cw_  = target_rgb.shape[1] // 4
        moments = []
        for i in range(4):
            for j in range(4):
                cell = target_rgb[i*ch_:(i+1)*ch_, j*cw_:(j+1)*cw_]
                for c in range(3):
                    px = cell[:, :, c].ravel() / 255.0
                    mean_ = px.mean()
                    moments.extend([
                        mean_,
                        px.std(),
                        float(np.cbrt(np.mean((px - mean_)**3)))
                    ])
        features.append(np.array(moments))
    except Exception:
        features.append(np.zeros(144))

    vec = np.concatenate([np.array(f).ravel() for f in features])
    return vec.astype(np.float32)



# ---------------------------------------------
# CNN FEATURE EXTRACTOR (EfficientNet-B0)
# Produces 1280-dim semantic features from
# pretrained ImageNet model via PyTorch.
# Falls back to zeros if torch not installed.
# ---------------------------------------------

_cnn_model  = None
_cnn_device = None

def _load_cnn_model():
    """Load EfficientNet-B0 once and cache it. Returns (model, device) or (None, None)."""
    global _cnn_model, _cnn_device
    if _cnn_model is not None:
        return _cnn_model, _cnn_device
    try:
        import torch
        import torchvision.models as models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [CNN] Loading EfficientNet-B0 on {device}...")

        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model.eval()
        model = model.to(device)

        _cnn_model  = model
        _cnn_device = device
        print(f"  [CNN] Ready. Feature dim: 1280")
        return model, device
    except ImportError:
        return None, None
    except Exception as e:
        print(f"  [CNN] Load failed: {e}")
        return None, None


def extract_cnn_features(filepath: str) -> np.ndarray:
    """
    Run EfficientNet-B0 on one image.
    Returns 1280-dim float32 vector, or zeros if torch unavailable.
    """
    model, device = _load_cnn_model()
    if model is None:
        return np.zeros(1280, dtype=np.float32)

    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        img = PILImage.open(filepath).convert("RGB")
        x   = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(x)

        vec = features.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    except Exception as e:
        return np.zeros(1280, dtype=np.float32)


CNN_FEATURE_DIM = 1280

# ---------------------------------------------
# BRIDGE: import signal extractors from analyzer
# ---------------------------------------------

def _import_analyzer():
    """Import image_authenticity.py as a module from the same directory."""
    import importlib.util
    analyzer_path = SCRIPT_DIR / "image_authenticity.py"
    if not analyzer_path.exists():
        print(f"\n  [ERR]  image_authenticity.py not found at {analyzer_path}")
        print("      Both files must be in the same folder.\n")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("image_authenticity", analyzer_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_run(fn, *args, default=None):
    """Run a signal module safely  returns default on any error."""
    try:
        return fn(*args)
    except Exception:
        return default


def extract_feature_vector(filepath: str, analyzer_mod) -> np.ndarray | None:
    """
    Run all signal extractors from the analyzer on one image.
    Each module is isolated  one failing module does not abort the image.
    Returns a fixed-length numpy feature vector aligned to SIGNAL_NAMES.
    """
    try:
        pil_img, np_rgb, np_bgr, gray = analyzer_mod.load_image(filepath)
    except Exception as e:
        print(f"    [WARN]  Could not load {Path(filepath).name}: {e}")
        return None

    all_signals = []

    # Each module isolated  failure yields empty list, not a crash
    def run(fn, *args):
        try:
            result = fn(*args)
            # Modules return (signals, extras...) or just signals
            return result[0] if isinstance(result, tuple) else result
        except Exception as e:
            return []

    all_signals += run(analyzer_mod.analyze_metadata,           pil_img, filepath)
    all_signals += run(analyzer_mod.analyze_frequency,          gray)
    all_signals += run(analyzer_mod.analyze_noise,              gray)
    all_signals += run(analyzer_mod.analyze_ela,                pil_img)
    all_signals += run(analyzer_mod.analyze_texture,            gray, np_rgb)
    all_signals += run(analyzer_mod.analyze_clone_detection,    gray)
    all_signals += run(analyzer_mod.analyze_compression_history, pil_img)
    all_signals += run(analyzer_mod.analyze_glcm_texture,       gray)
    all_signals += run(analyzer_mod.analyze_color_statistics,   np_rgb)
    all_signals += run(analyzer_mod.analyze_dct_artifacts,      gray)
    all_signals += run(analyzer_mod.analyze_region_coherence,    np_rgb, gray)
    all_signals += run(analyzer_mod.analyze_gradient_direction,  gray)
    all_signals += run(analyzer_mod.analyze_channel_uniformity,  np_rgb)
    all_signals += run(analyzer_mod.analyze_jpeg_quantization,   pil_img)
    all_signals += run(analyzer_mod.analyze_hog_statistics,      gray)
    all_signals += run(analyzer_mod.analyze_fft_bands,           gray)
    all_signals += run(analyzer_mod.analyze_multiscale_noise,    gray)

    if not all_signals:
        print(f"    [WARN]  All modules failed for {Path(filepath).name}  skipping")
        return None

    # Build fixed-length vector  missing signals default to 0.0
    score_map = {s.name: s.score for s in all_signals}
    base_vec = np.array([score_map.get(name, 0.0) for name in SIGNAL_NAMES],
                        dtype=np.float32)

    # Engineered interaction features
    noise    = score_map.get("Noise Level (PRNU)", 0.5)
    texture  = score_map.get("GLCM Texture Homogeneity", 0.5)
    grad     = score_map.get("Gradient Direction Entropy", 0.5)
    ela      = score_map.get("Error Level Analysis (ELA)", 0.5)
    sat      = score_map.get("Saturation Distribution", 0.5)
    chrom    = score_map.get("Chromatic Aberration", 0.5)
    lv       = score_map.get("Local Variance Uniformity", 0.5)
    ms_noise = score_map.get("Multi-Scale Noise Profile", 0.5)
    hog_skew = score_map.get("HOG Distribution Skew", 0.5)
    fft_band = score_map.get("FFT Band Energy Ratio", 0.5)

    engineered = np.array([
        noise * texture,
        grad * (1.0 - chrom),
        ela * lv,
        (noise + texture + lv) / 3.0,
        abs(sat - 0.5) * 2.0,
        ms_noise * (1.0 - noise),
        hog_skew * fft_band,
        (ms_noise + hog_skew + fft_band) / 3.0,
    ], dtype=np.float32)

    # Rich handcrafted features (262 dims)
    try:
        pil_img2, np_rgb2, _, gray2 = analyzer_mod.load_image(filepath)
        rich_vec = extract_rich_cnn_features(gray2, np_rgb2)
    except Exception:
        rich_vec = np.zeros(262, dtype=np.float32)

    # EfficientNet CNN features (1280 dims) -- semantic realism
    cnn_vec = extract_cnn_features(filepath)

    vec = np.concatenate([base_vec, engineered, rich_vec, cnn_vec])
    return vec


# ---------------------------------------------
# DATASET MANAGEMENT
# ---------------------------------------------

def load_dataset() -> dict:
    if DATASET_FILE.exists():
        with open(DATASET_FILE) as f:
            return json.load(f)
    return {"samples": [], "created": datetime.now().isoformat()}


def save_dataset(ds: dict):
    with open(DATASET_FILE, 'w') as f:
        json.dump(ds, f, indent=2)


def build_dataset_from_folder(dataset_dir: str, analyzer_mod, max_per_class: int = 0, split_filter: str = None) -> dict:
    """
    Scan a folder structure like:
      dataset_dir/
        real/   or  authentic/  or  photo/
        ai/     or  generated/  or  fake/
        edited/ or  tampered/         (optional)
    """
    ds   = load_dataset()

    # -- DEBUG: print cache file location and state ----------------------------
    print(f"\n  [CACHE]  Cache file: {DATASET_FILE}")
    print(f"  [CACHE]  Cache exists: {DATASET_FILE.exists()}")
    if ds['samples']:
        cached_features = len(ds['samples'][0].get('features', []))
        print(f"  [CACHE]  Cached samples: {len(ds['samples'])}, features per sample: {cached_features}")
    else:
        print(f"  [CACHE]  Cache is empty (0 samples)")

    # -- Auto cache-bust: if cached feature count doesn't match current --------
    expected_features = len(SIGNAL_NAMES) + 8 + 262 + CNN_FEATURE_DIM  # signals + engineered + rich + CNN
    print(f"  [CACHE]  Expected features: {expected_features}")
    if ds['samples']:
        cached_features = len(ds['samples'][0].get('features', []))
        if cached_features != expected_features:
            print(f"\n  [WARN]  Cache mismatch: cached={cached_features} features, "
                  f"current={expected_features} features.")
            print(f"  [DELETE]   Auto-clearing old cache and re-extracting all features...\n")
            ds = {"samples": [], "created": datetime.now().isoformat()}
            save_dataset(ds)
            print(f"  [OK]  Cache cleared. dataset.json now has 0 samples.")
        else:
            print(f"  [OK]  Cache feature count matches  using cached features.")

    root = Path(dataset_dir)
    existing_paths = {s['path'] for s in ds['samples']}

    added = 0
    skipped = 0
    failed = 0

    # Build list of (folder, label_str) -- supports flat AND nested layouts
    # Flat:   dataset/real/  dataset/fake/
    # Nested: dataset/train/REAL/  dataset/train/FAKE/  (CIFAKE format)
    label_folders = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        lname = item.name.lower()
        if lname in LABEL_FROM_STR:
            label_folders.append((item, lname))
        else:
            # Could be a split folder like train/ or test/
            if split_filter and lname != split_filter.lower():
                print(f"  Skipping split '{item.name}' (--split={split_filter})")
                continue
            for sub in sorted(item.iterdir()):
                if sub.is_dir() and sub.name.lower() in LABEL_FROM_STR:
                    label_folders.append((sub, sub.name.lower()))

    if not label_folders:
        print(f"  [ERR]  No labeled folders found.")
        print(f"  Expected: real/, fake/, ai/ -- or nested: train/REAL/, train/FAKE/")
        return ds

    # Show detected structure
    print("\n  Structure detected:")
    seen = {}
    for folder, lstr in label_folders:
        imgs = [f for f in folder.rglob('*')
                if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]
        key = LABEL_NAMES[LABEL_FROM_STR[lstr]]
        seen[key] = seen.get(key, 0) + len(imgs)
    for cls, cnt in sorted(seen.items()):
        cap = f" (will use {max_per_class})" if max_per_class and cnt > max_per_class else ""
        print(f"    {cls:<12} {cnt} images{cap}")

    class_added = {}

    for folder, label_str in label_folders:
        label = LABEL_FROM_STR[label_str]
        images = [f for f in sorted(folder.rglob('*'))
                  if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]

        # Per-class cap
        label_key = LABEL_NAMES[label]
        already_have = class_added.get(label_key, 0) + sum(
            1 for s in ds['samples'] if s['label_str'] == label_key)
        if max_per_class and already_have >= max_per_class:
            print(f"  [{label_key.upper()}] Cap of {max_per_class} reached -- skipping {folder.name}/")
            continue

        new_images = [f for f in images if str(f.resolve()) not in existing_paths]
        if max_per_class:
            remaining = max_per_class - already_have
            new_images = new_images[:max(0, remaining)]

        # Show subfolder breakdown so user knows what was found
        subfolders = sorted(set(f.parent for f in images if f.parent != folder))
        if subfolders:
            print(f"\n  [{LABEL_NAMES[label].upper()}] {len(images)} images found recursively in '{folder.name}/'")
            for sf in subfolders[:5]:
                sf_count = sum(1 for f in images if f.parent == sf)
                print(f"      [DIR] {sf.name}/  ({sf_count} images)")
            if len(subfolders) > 5:
                print(f"      ... and {len(subfolders)-5} more subfolders")
        else:
            print(f"\n  [{LABEL_NAMES[label].upper()}] {len(images)} images in '{folder.name}/'")
        print(f"     {len(new_images)} new, {len(images)-len(new_images)} already cached")

        skipped += len(images) - len(new_images)

        for idx, img_path in enumerate(new_images, 1):
            abs_path = str(img_path.resolve())

            # Compact progress line  overwrites itself
            pct  = idx / len(new_images) * 100 if new_images else 100
            bar  = ('#' * int(pct / 5)).ljust(20)
            print(f"\r    [{bar}] {idx}/{len(new_images)}  {img_path.name[:35]:<35}",
                  end='', flush=True)

            vec = extract_feature_vector(abs_path, analyzer_mod)
            if vec is None:
                failed += 1
                continue

            ds['samples'].append({
                "path":      abs_path,
                "label":     label,
                "label_str": LABEL_NAMES[label],
                "features":  vec.tolist(),
                "added":     datetime.now().isoformat(),
            })
            existing_paths.add(abs_path)
            added += 1

        # Save after each folder so progress isn't lost if interrupted
        save_dataset(ds)
        class_added[label_key] = class_added.get(label_key, 0) + added
        print(f"\r    [OK]  {len(new_images)} processed, {failed} failed{' '*40}")

    ds['updated'] = datetime.now().isoformat()
    save_dataset(ds)
    print(f"\n  Dataset: +{added} added, {skipped} skipped, {failed} failed")
    print(f"  Total samples: {len(ds['samples'])}")
    return ds


def add_single_image(filepath: str, label_str: str, analyzer_mod) -> bool:
    """Add one manually labeled image to the dataset."""
    label_str = label_str.lower()
    if label_str not in LABEL_FROM_STR:
        print(f"  [ERR]  Unknown label '{label_str}'. Use: real, ai, edited")
        return False

    label    = LABEL_FROM_STR[label_str]
    abs_path = str(Path(filepath).resolve())
    ds       = load_dataset()

    existing = {s['path'] for s in ds['samples']}
    if abs_path in existing:
        print(f"    Already in dataset: {Path(filepath).name}")
        return False

    print(f"  Extracting features from {Path(filepath).name}...", end='', flush=True)
    vec = extract_feature_vector(abs_path, analyzer_mod)
    if vec is None:
        return False

    ds['samples'].append({
        "path":      abs_path,
        "label":     label,
        "label_str": LABEL_NAMES[label],
        "features":  vec.tolist(),
        "added":     datetime.now().isoformat(),
    })
    save_dataset(ds)
    print(f"   Added as '{LABEL_NAMES[label]}'")
    return True


def dataset_stats(ds: dict):
    from collections import Counter
    counts = Counter(s['label_str'] for s in ds['samples'])
    print(f"\n  Dataset: {len(ds['samples'])} total samples")
    for lbl, cnt in sorted(counts.items()):
        bar = '#' * cnt + '-' * max(0, 20 - cnt)
        print(f"    {lbl:<10} [{bar}] {cnt}")


# ---------------------------------------------
# MODEL TRAINING
# ---------------------------------------------

def prepare_arrays(ds: dict, binary: bool = True):
    """
    Convert dataset to X, y arrays.
    binary=True: real(0) vs AI+edited(1)   best for small datasets
    binary=False: 3-class real/ai/edited
    """
    samples = [s for s in ds['samples'] if s.get('features')]
    if not samples:
        return None, None, []

    X = np.array([s['features'] for s in samples], dtype=np.float32)
    y_raw = np.array([s['label'] for s in samples])

    if binary:
        y = (y_raw > 0).astype(int)   # 0=real, 1=ai/edited
    else:
        y = y_raw

    paths = [s['path'] for s in samples]
    return X, y, paths



def split_feature_vector(vec: np.ndarray):
    """Split a combined feature vector into handcrafted and CNN parts."""
    n_handcrafted = len(SIGNAL_NAMES) + 8 + 262   # 296
    return vec[:n_handcrafted], vec[n_handcrafted:]  # handcrafted, cnn


def train(ds: dict, binary: bool = True):
    """Train an ensemble classifier and save to model/."""

    print(f"\n{'='*60}")
    print("  Training Image Authenticity Classifier")
    print(f"{'='*60}")

    X_full, y, paths = prepare_arrays(ds, binary=binary)
    if X_full is None or len(X_full) < 6:
        print(f"\n  [ERR]  Need at least 6 labeled images to train.")
        print("      Add images with --add-image or --dataset flags.")
        return None

    # RF trains on handcrafted features only (296 dims)
    # CNN model trains separately on CNN features (1280 dims)
    X = np.array([split_feature_vector(row)[0] for row in X_full])

    n_samples  = len(X)
    n_features = X_full.shape[1]
    classes    = np.unique(y)
    print(f"\n  Samples:  {n_samples}")
    print(f"  Features: {n_features} total ({X.shape[1]} handcrafted + {n_features - X.shape[1]} CNN)")
    print(f"  Classes:  {['real', 'ai/edited'] if binary else ['real','ai','edited']}")

    if n_samples < 20:
        print(f"\n  [WARN]  Small dataset ({n_samples} samples). Results may be unreliable.")
        print(f"     Add more images for better accuracy (aim for 50+ per class).")

    # -- Build ensemble --------------------------------------------
    # Three complementary classifiers voted together
    rf  = RandomForestClassifier(n_estimators=400, max_depth=12,
                                  min_samples_leaf=2, class_weight='balanced',
                                  max_features='sqrt',
                                  random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.03,
                                      max_depth=5, subsample=0.8,
                                      random_state=42)
    lr  = LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced',
                              solver='saga', random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft',
        weights=[2, 1, 3]   # LR dominates with CNN high-dim features
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    ensemble)
    ])

    # -- Cross-validation ------------------------------------------
    n_splits = min(5, n_samples // max(len(classes), 2))
    n_splits = max(n_splits, 2)

    print(f"\n  Running {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc',
                                 error_score='raise')
    print(f"  CV ROC-AUC: {cv_scores.mean():.3f}  {cv_scores.std():.3f}")
    print(f"  Per-fold:   {[f'{s:.3f}' for s in cv_scores]}")

    # -- Final train / test split ----------------------------------
    if n_samples >= 20:
        X_train, X_test, y_train, y_test, p_train, p_test = \
            train_test_split(X, y, paths, test_size=0.2,
                             stratify=y, random_state=42)
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
        p_test = paths

    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(classes) == 2 else None

    print(f"\n  Test set classification report:")
    print(classification_report(y_test, y_pred,
          target_names=['real', 'ai/edited'] if binary else ['real','ai','edited'],
          zero_division=0))

    if y_proba is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  Test ROC-AUC: {auc:.3f}")

    # -- Feature importance (from RF inside ensemble) --------------
    scaler  = pipeline.named_steps['scaler']
    clf     = pipeline.named_steps['clf']
    rf_clf  = clf.estimators_[0]  # the RandomForest

    importances = rf_clf.feature_importances_
    feat_ranked = sorted(zip(SIGNAL_NAMES, importances),
                         key=lambda x: x[1], reverse=True)

    print(f"\n  Top signal importances (learned by Random Forest):")
    for name, imp in feat_ranked[:8]:
        bar = '#' * int(imp * 60)
        print(f"    {name:<38} {bar} {imp:.4f}")

    # -- Save model ------------------------------------------------
    joblib.dump(pipeline, MODEL_FILE)
    joblib.dump(scaler,   SCALER_FILE)

    meta = {
        "trained":      datetime.now().isoformat(),
        "n_samples":    n_samples,
        "n_features":   n_features,
        "binary":       binary,
        "cv_auc_mean":  float(cv_scores.mean()),
        "cv_auc_std":   float(cv_scores.std()),
        "signal_names": SIGNAL_NAMES,
        "feature_importances": {n: float(i) for n, i in feat_ranked},
        "n_features_expected": int(X.shape[1]),
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Train separate CNN classifier (LR on CNN features only) ──────────────
    _train_cnn_model(X_full, y, binary, n_splits, cv)

    # -- Export learned weights back to analyzer -------------------
    learned_weights = {}
    for name, imp in feat_ranked:
        # Map 01 importance to 0.33.0 weight range
        # Minimum weight 0.3 so no signal is fully ignored
        new_w = round(0.3 + imp * 25.0, 2)   # scale: top signal ~2.5-3.0
        new_w = min(3.0, max(0.3, new_w))
        learned_weights[name] = new_w

    with open(WEIGHTS_FILE, 'w') as f:
        json.dump({
            "weights":  learned_weights,
            "trained":  datetime.now().isoformat(),
            "n_samples": n_samples,
            "cv_auc":   float(cv_scores.mean()),
        }, f, indent=2)

    print(f"\n  [OK]  Model saved  -> {MODEL_FILE}")
    print(f"  [OK]  Weights saved -> {WEIGHTS_FILE}")
    print(f"      (image_authenticity.py will auto-load these on next run)\n")

    # -- Generate training report ----------------------------------
    _plot_training_report(feat_ranked, cv_scores, y_test, y_pred, y_proba,
                          binary, n_samples, cv_scores.mean())

    return pipeline


def _plot_training_report(feat_ranked, cv_scores, y_test, y_pred,
                           y_proba, binary, n_samples, cv_auc):
    """Save a visual training report PNG."""

    fig = plt.figure(figsize=(18, 12), facecolor='#0a0a0f')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.90, bottom=0.07)

    clr = {'bg': '#0a0a0f', 'surf': '#111118', 'text': '#e8e8f0',
           'muted': '#7a7a95', 'green': '#00e5a0', 'orange': '#ff6b35',
           'purple': '#7c5cfc', 'yellow': '#ffcc00', 'red': '#ff4757',
           'border': '#2a2a3d'}

    def style(ax, title=""):
        ax.set_facecolor(clr['surf'])
        ax.tick_params(colors=clr['muted'], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(clr['border'])
        if title: ax.set_title(title, color=clr['text'], fontsize=9,
                                fontweight='bold', pad=8)

    # -- Feature importances ---------------------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    style(ax0, "Signal Importances (Learned by RF)")
    names  = [n for n, _ in feat_ranked]
    imps   = [i for _, i in feat_ranked]
    colors = [clr['purple'] if i > np.median(imps) else clr['muted'] for i in imps]
    bars   = ax0.barh(range(len(names)), imps, color=colors, height=0.7, alpha=0.85)
    ax0.set_yticks(range(len(names)))
    ax0.set_yticklabels(names, color=clr['text'], fontsize=7)
    ax0.set_xlabel("Importance", color=clr['muted'], fontsize=8)
    ax0.invert_yaxis()
    for bar, imp in zip(bars, imps):
        ax0.text(imp + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{imp:.3f}', va='center', color=clr['muted'], fontsize=6)

    # -- CV scores -------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    style(ax1, "Cross-Validation AUC")
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bar_colors = [clr['green'] if s > 0.7 else clr['yellow'] if s > 0.55 else clr['red']
                  for s in cv_scores]
    ax1.bar(folds, cv_scores, color=bar_colors, alpha=0.85, width=0.6)
    ax1.axhline(cv_scores.mean(), color=clr['purple'], linestyle='--',
                lw=1.5, label=f'Mean={cv_scores.mean():.3f}')
    ax1.axhline(0.5, color=clr['muted'], linestyle=':', lw=1, alpha=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("AUC", color=clr['muted'], fontsize=8)
    ax1.legend(facecolor=clr['surf'], edgecolor=clr['border'],
               labelcolor=clr['text'], fontsize=7)

    # -- Confusion matrix ------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    style(ax2, "Confusion Matrix (Test Set)")
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Real', 'AI/Edited'] if binary else ['Real', 'AI', 'Edited']
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, color=clr['text'], fontsize=8)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, color=clr['text'], fontsize=8)
    ax2.set_xlabel("Predicted", color=clr['muted'], fontsize=8)
    ax2.set_ylabel("Actual",    color=clr['muted'], fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color=clr['text'], fontsize=12, fontweight='bold')

    # -- ROC curve -------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "ROC Curve")
    if y_proba is not None and len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax3.plot(fpr, tpr, color=clr['green'], lw=2, label=f'AUC = {auc:.3f}')
        ax3.fill_between(fpr, tpr, alpha=0.1, color=clr['green'])
        ax3.plot([0,1],[0,1], color=clr['muted'], linestyle='--', lw=1)
        ax3.set_xlabel("False Positive Rate", color=clr['muted'], fontsize=8)
        ax3.set_ylabel("True Positive Rate", color=clr['muted'], fontsize=8)
        ax3.legend(facecolor=clr['surf'], edgecolor=clr['border'],
                   labelcolor=clr['text'], fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Need both classes\nin test set",
                 ha='center', va='center', color=clr['muted'], fontsize=10)
        ax3.axis('off')

    # -- Stats summary ---------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    style(ax4, "Training Summary")
    ax4.axis('off')
    stats_text = [
        ("Training samples", str(n_samples)),
        ("Features",         str(len(SIGNAL_NAMES))),
        ("CV AUC",           f"{cv_auc:.3f}"),
        ("Mode",             "Binary (real vs AI)" if binary else "3-class"),
        ("Model",            "RF + GBM + LR Ensemble"),
        ("Saved to",         "model/classifier.joblib"),
        ("Weights exported", "model/learned_weights.json"),
    ]
    for i, (k, v) in enumerate(stats_text):
        y_pos = 0.88 - i * 0.12
        ax4.text(0.0, y_pos, k + ":", transform=ax4.transAxes,
                 color=clr['muted'], fontsize=9)
        ax4.text(0.55, y_pos, v, transform=ax4.transAxes,
                 color=clr['text'], fontsize=9, fontweight='bold')

    fig.text(0.5, 0.95, "Image Authenticity Classifier  Training Report",
             ha='center', color=clr['text'], fontsize=13, fontweight='bold')
    fig.text(0.5, 0.93, datetime.now().strftime("%Y-%m-%d %H:%M"),
             ha='center', color=clr['muted'], fontsize=9)

    out = OUTPUT_DIR / "training_report.png"
    plt.savefig(out, facecolor='#0a0a0f', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [CHART]  Training report -> {out}")


# ---------------------------------------------
# PREDICTION (using trained model)
# ---------------------------------------------


def _train_cnn_model(X_full, y, binary, n_splits, cv):
    """
    Train a Logistic Regression classifier on CNN features only.
    LR excels at high-dimensional dense feature spaces like CNN embeddings.
    """
    # Split out CNN features (last 1280 dims)
    _, X_cnn = split_feature_vector(X_full[0])
    cnn_dim = len(X_cnn)

    if cnn_dim == 0 or X_full[:, -cnn_dim:].max() == 0:
        print("  [CNN model] No CNN features found -- skipping CNN classifier")
        return

    X_cnn_all = np.array([split_feature_vector(row)[1] for row in X_full])

    # Check CNN features are non-zero (torch installed and working)
    nonzero = (X_cnn_all.sum(axis=1) != 0).sum()
    if nonzero < len(y) * 0.5:
        print(f"  [CNN model] Only {nonzero}/{len(y)} samples have CNN features.")
        print(f"  [CNN model] Make sure PyTorch is installed: py -3.12 -m pip install torch torchvision")
        return

    print(f"  Training CNN classifier (LR on {cnn_dim}-dim EfficientNet features)...")
    print(f"  Non-zero CNN samples: {nonzero}/{len(y)}")

    cnn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=3000,
                                   class_weight='balanced',
                                   solver='saga', random_state=42))
    ])

    cv_scores = cross_val_score(cnn_pipeline, X_cnn_all, y,
                                 cv=cv, scoring='roc_auc', error_score='raise')
    print(f"  CNN CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    cnn_pipeline.fit(X_cnn_all, y)
    joblib.dump(cnn_pipeline, CNN_MODEL_FILE)

    cnn_meta = {
        "trained":    datetime.now().isoformat(),
        "cv_auc":     float(cv_scores.mean()),
        "cnn_dim":    cnn_dim,
        "n_samples":  len(y),
    }
    with open(CNN_META_FILE, 'w') as f:
        json.dump(cnn_meta, f, indent=2)

    print(f"  [OK] CNN model saved -> {CNN_MODEL_FILE}")
    print(f"  [OK] CNN CV AUC: {cv_scores.mean():.3f}")


def _calibrate_prob(prob: float, T: float = 2.0,
                     extreme_threshold: float = 0.90) -> float:
    """
    Temperature scaling applied only to extreme probabilities (>90% or <10%).
    Preserves the middle range where the model is trustworthy.
    Prevents LR from saturating at 0% or 100% on out-of-distribution images.
    T=2.0: 99%->91%, 95%->81%, 5%->19%, 1%->9%
    """
    import math
    prob = max(1e-6, min(1 - 1e-6, float(prob)))
    if prob > extreme_threshold or prob < (1 - extreme_threshold):
        log_odds = math.log(prob / (1 - prob))
        return 1 / (1 + math.exp(-log_odds / T))
    return prob


def _blend_predictions(hc_proba: float, cnn_proba: float,
                        hc_auc: float, cnn_auc: float) -> tuple:
    """
    Calibrated blend of handcrafted RF and CNN model probabilities.

    Three safeguards:
      1. Temperature scaling on CNN extremes (prevents 0%/100% saturation)
      2. AUC-weighted blend (better model gets more say)
      3. Agreement gate (if models disagree >40%, anchor to RF, flag uncertainty)

    Returns (blended_prob, cnn_calibrated, disagreement, is_uncertain)
    """
    cnn_cal      = _calibrate_prob(cnn_proba, T=2.0)
    disagreement = abs(hc_proba - cnn_cal)

    if disagreement > 0.40:
        # Models strongly disagree -- CNN is likely miscalibrated (wrong training data)
        # Anchor to RF (more conservative), reduce CNN weight
        rf_weight  = 0.75
        cnn_weight = 0.25
        uncertain  = True
    else:
        total      = hc_auc + cnn_auc
        rf_weight  = hc_auc  / total
        cnn_weight = cnn_auc / total
        uncertain  = False

    blended = rf_weight * hc_proba + cnn_weight * cnn_cal
    return blended, cnn_cal, disagreement, uncertain


def predict_single(filepath: str, analyzer_mod) -> dict | None:
    """Run full pipeline: extract signals -> apply ML model -> return result."""

    if not MODEL_FILE.exists():
        print("  [WARN]  No trained model found. Run --train first.")
        return None

    pipeline = joblib.load(MODEL_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)

    print(f"\n  Extracting features from {Path(filepath).name}...", end='', flush=True)
    vec = extract_feature_vector(filepath, analyzer_mod)
    if vec is None:
        return None
    print(" done")

    # Split handcrafted (296) vs CNN (1280) features
    hc_vec, cnn_vec_pred = split_feature_vector(vec)

    # Handcrafted model prediction
    X_hc      = hc_vec.reshape(1, -1)
    pred      = pipeline.predict(X_hc)[0]
    probas_hc = pipeline.predict_proba(X_hc)[0]

    binary    = meta.get('binary', True)
    ai_prob   = float(probas_hc[1])
    real_prob = float(probas_hc[0])
    hc_auc    = meta.get('cv_auc_mean', 0.5)

    # CNN model prediction -- calibrated blend if available
    if CNN_MODEL_FILE.exists() and cnn_vec_pred.sum() != 0:
        try:
            cnn_pipe    = joblib.load(CNN_MODEL_FILE)
            cnn_meta_d  = json.load(open(CNN_META_FILE))
            cnn_probas  = cnn_pipe.predict_proba(cnn_vec_pred.reshape(1, -1))[0]
            cnn_ai_raw  = float(cnn_probas[1])
            cnn_auc     = cnn_meta_d.get('cv_auc', 0.5)

            blended, cnn_cal, disagree, uncertain = _blend_predictions(
                ai_prob, cnn_ai_raw, hc_auc, cnn_auc)

            # Extra guard: if forensic signals lean AI but CNN leans real,
            # CNN is likely miscalibrated (CIFAKE trained on 32px, not photorealistic AI)
            # In this case, trust RF more heavily
            forensic_says_ai = ai_prob > 0.45
            cnn_says_real    = cnn_cal  < 0.40
            cnn_likely_wrong = forensic_says_ai and cnn_says_real

            if cnn_likely_wrong:
                # Override blend: 85% RF, 15% CNN
                blended   = 0.85 * ai_prob + 0.15 * cnn_cal
                uncertain = True

            print(f"  Handcrafted: {ai_prob*100:.0f}% AI  (AUC={hc_auc:.3f})")
            if abs(cnn_ai_raw - cnn_cal) > 0.02:
                print(f"  CNN model:   {cnn_ai_raw*100:.0f}% AI -> calibrated {cnn_cal*100:.0f}%  (AUC={cnn_auc:.3f})")
            else:
                print(f"  CNN model:   {cnn_cal*100:.0f}% AI  (AUC={cnn_auc:.3f})")
            if cnn_likely_wrong:
                print(f"  [WARN] CNN contradicts forensics -- CNN likely miscalibrated (CIFAKE vs photorealistic AI)")
                print(f"         Anchoring to forensic signals (RF=85%, CNN=15%)")
            elif uncertain:
                print(f"  [WARN] Models disagree by {disagree*100:.0f}% -- anchoring to forensic signals (RF weight=75%)")
            print(f"  Blended:     {blended*100:.0f}% AI")

            ai_prob   = blended
            real_prob = 1.0 - blended
            pred      = 1 if blended > 0.45 else 0
        except Exception:
            pass

    label_str  = "ai/edited" if pred == 1 else "real"
    auth_score = int((1.0 - ai_prob) * 100)

    result = {
        "file":          str(filepath),
        "prediction":    label_str,
        "ai_probability":  round(ai_prob * 100, 1),
        "real_probability": round(real_prob * 100, 1),
        "authenticity_score": auth_score,
        "model_cv_auc":  meta.get('cv_auc_mean', 0),
        "signal_scores": dict(zip(SIGNAL_NAMES, vec.tolist())),
    }

    # Top contributing signals
    feat_imp = meta.get('feature_importances', {})
    contrib  = sorted(
        [(name, vec[SIGNAL_NAMES.index(name)] * imp)
         for name, imp in feat_imp.items() if name in SIGNAL_NAMES],
        key=lambda x: x[1], reverse=True
    )

    print(f"\n{'-'*55}")
    print(f"  ML MODEL PREDICTION")
    print(f"{'-'*55}")
    if pred == 1:
        emoji = "[WARN] " if ai_prob > 0.55 else ""
        print(f"  {emoji}  PREDICTION: {label_str.upper()}  ({ai_prob*100:.0f}% confidence)")
    else:
        print(f"  [OK]  PREDICTION: REAL / AUTHENTIC  ({real_prob*100:.0f}% confidence)")
    print(f"  Authenticity Score: {auth_score}/100")
    print(f"  Model CV-AUC: {meta.get('cv_auc_mean', 0):.3f}  (trained on {meta['n_samples']} samples)")

    print(f"\n  Top signals driving this prediction:")
    for name, score in contrib[:5]:
        bar = '#' * int(abs(score) * 80)
        direction = "-> AI" if vec[SIGNAL_NAMES.index(name)] > 0.5 else "-> Real"
        print(f"    {name:<38} {bar[:20]} {direction}")

    return result


def predict_batch(folder: str, analyzer_mod) -> list[dict]:
    """Predict all images in a folder and print a summary table."""
    folder_path = Path(folder)
    images = [f for f in sorted(folder_path.rglob('*'))
              if f.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print(f"  No images found in {folder}")
        return []

    print(f"\n  Batch predicting {len(images)} images from {folder}")
    print(f"{'-'*65}")
    print(f"  {'File':<30} {'Prediction':<14} {'AI%':>5}  {'Auth':>5}")
    print(f"{'-'*65}")

    results = []
    for img_path in images:
        vec = extract_feature_vector(str(img_path), analyzer_mod)
        if vec is None:
            continue

        pipeline = joblib.load(MODEL_FILE)
        pred   = pipeline.predict(vec.reshape(1,-1))[0]
        probas = pipeline.predict_proba(vec.reshape(1,-1))[0]
        ai_p   = float(probas[1])
        flag   = "[WARN] AI" if ai_p > 0.55 else (" Maybe" if ai_p > 0.35 else "[OK] Real")

        print(f"  {img_path.name:<30} {flag:<14} {ai_p*100:>4.0f}%  {(1-ai_p)*100:>4.0f}")
        results.append({"file": str(img_path), "ai_prob": ai_p, "pred": pred})

    print(f"{'-'*65}")
    ai_count   = sum(1 for r in results if r['pred'] == 1)
    real_count = len(results) - ai_count
    print(f"  Summary: {real_count} real, {ai_count} AI/edited out of {len(results)} images\n")
    return results


# ---------------------------------------------
# FULL PIPELINE (analyzer + trainer together)
# ---------------------------------------------

# C2PA verdict categories
_C2PA_CONFIRMED_REAL = {'VERIFIED_REAL'}
_C2PA_CONFIRMED_AI   = {'VERIFIED_AI'}
_C2PA_NEEDS_ML       = {'NO_MANIFEST', 'MANIFEST_FOUND', 'EDITED', None}


def _load_c2pa_checker(analyzer_mod):
    """Load c2pa_checker from same directory as analyzer."""
    try:
        import importlib.util
        checker_path = Path(analyzer_mod.__file__).parent / 'c2pa_checker.py'
        if not checker_path.exists():
            return None
        spec = importlib.util.spec_from_file_location('c2pa_checker', checker_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def run_full_pipeline(filepath: str, analyzer_mod, verbose: bool = False):
    """
    3-stage pipeline with C2PA short-circuit:

    STAGE 1: C2PA provenance check (instant, cryptographic)
      -> VERIFIED_REAL : stop. Image is cryptographically proven real.
      -> VERIFIED_AI   : stop. Image is cryptographically proven AI.
      -> NO_MANIFEST   : continue to Stage 2 (inconclusive)

    STAGE 2: Rule-based forensic signals (17 modules)
      -> Runs always when C2PA is inconclusive

    STAGE 3: ML model (EfficientNet CNN + Random Forest)
      -> Only runs if Stage 2 score is ambiguous (35-75% AI)
      -> Skipped if Stage 2 gives a clear verdict, saving time
    """
    print(f"\n{'='*60}")
    print(f"  Full Pipeline Analysis")
    print(f"  File: {Path(filepath).name}")
    print(f"{'='*60}")

    # ── STAGE 1: C2PA ────────────────────────────────────────────
    print("\n  [STAGE 1/3] C2PA Provenance Check...")
    c2pa_mod     = _load_c2pa_checker(analyzer_mod)
    c2pa_result  = None
    c2pa_verdict = None
    c2pa_conf    = 'LOW'
    c2pa_summary = ''
    c2pa_details = {}

    if c2pa_mod:
        c2pa_result  = c2pa_mod.check_c2pa(filepath)
        c2pa_verdict = c2pa_result.get('verdict')
        c2pa_conf    = c2pa_result.get('confidence', 'LOW')
        c2pa_summary = c2pa_result.get('summary', '')
        c2pa_details = c2pa_result.get('details', {})

        if c2pa_verdict == 'NO_MANIFEST':
            # Silent pass-through -- absence of C2PA means nothing, don't alarm user
            print("  No C2PA manifest found -- proceeding to forensic analysis.")

        elif c2pa_verdict == 'VERIFIED_REAL':
            print(f"  [REAL]  VERIFIED REAL  ({c2pa_conf} confidence)")
            print(f"  {c2pa_summary}")
            if c2pa_details.get('camera_manufacturer'):
                print(f"  Camera: {c2pa_details['camera_manufacturer'].title()}")

        elif c2pa_verdict == 'VERIFIED_AI':
            print(f"  [AI]  VERIFIED AI-GENERATED  ({c2pa_conf} confidence)")
            print(f"  {c2pa_summary}")
            if c2pa_details.get('ai_tool'):
                print(f"  Tool:   {c2pa_details['ai_tool']}")

        elif c2pa_verdict == 'EDITED':
            print(f"  [EDIT]  MANIFEST FOUND -- image has been edited  ({c2pa_conf} confidence)")
            print(f"  {c2pa_summary}")

        elif c2pa_verdict == 'TAMPERED':
            print(f"  [WARN]  MANIFEST TAMPERED -- signature broken  ({c2pa_conf} confidence)")

        elif c2pa_verdict == 'MANIFEST_FOUND':
            print(f"  [?]  MANIFEST FOUND but origin unclear  ({c2pa_conf} confidence)")
            if c2pa_details.get('claim_generator'):
                print(f"  Generator: {c2pa_details['claim_generator']}")
    else:
        # c2pa_checker.py not found -- silent, don't interrupt workflow
        print("  c2pa_checker.py not in folder -- C2PA stage skipped.")

    # ── SHORT-CIRCUIT: Cryptographic verdict ─────────────────────
    if c2pa_verdict in _C2PA_CONFIRMED_REAL and c2pa_result.get('confidence') == 'HIGH':
        print(f"\n{'='*60}")
        print(f"  FINAL VERDICT  [C2PA VERIFIED]")
        print(f"{'='*60}")
        print(f"  [CONFIRMED REAL]  Cryptographic provenance verified.")
        cam = c2pa_details.get('camera_manufacturer', 'camera')
        print(f"  Captured by: {cam.title()}")
        print(f"  Stages 2 and 3 skipped -- C2PA signature is definitive.")
        print(f"  Authenticity: 98/100")
        print(f"{'='*60}\n")
        return

    if c2pa_verdict in _C2PA_CONFIRMED_AI:
        print(f"\n{'='*60}")
        print(f"  FINAL VERDICT  [C2PA VERIFIED]")
        print(f"{'='*60}")
        print(f"  [CONFIRMED AI]  AI generation confirmed by manifest.")
        tool = c2pa_details.get('ai_tool', 'AI tool')
        print(f"  Generated by: {tool}")
        print(f"  Stages 2 and 3 skipped -- C2PA manifest is definitive.")
        print(f"  Authenticity: 2/100")
        print(f"{'='*60}\n")
        return

    # ── STAGE 2: Rule-based forensic signals ─────────────────────
    print("\n  [STAGE 2/3] Forensic signal analysis...")
    report = analyzer_mod.analyze(filepath, verbose=verbose, save_report=True)
    rb_ai  = report.ai_probability

    # ── STAGE 3: ML model (only if verdict is ambiguous) ─────────
    AMBIGUOUS_LOW  = 0.30
    AMBIGUOUS_HIGH = 0.70

    ml_skipped_reason = None
    ml_result         = None

    if not MODEL_FILE.exists():
        ml_skipped_reason = "No trained model (run --train first)"
    elif rb_ai < AMBIGUOUS_LOW:
        ml_skipped_reason = f"Stage 2 clear: {rb_ai*100:.0f}% AI -- image looks real, ML not needed"
    elif rb_ai > AMBIGUOUS_HIGH:
        ml_skipped_reason = f"Stage 2 clear: {rb_ai*100:.0f}% AI -- image looks AI, ML not needed"

    if ml_skipped_reason:
        print(f"\n  [STAGE 3/3] ML model -- SKIPPED")
        print(f"  Reason: {ml_skipped_reason}")
        blended = rb_ai
    else:
        print(f"\n  [STAGE 3/3] ML model (Stage 2 ambiguous: {rb_ai*100:.0f}% AI)...")
        ml_result = predict_single(filepath, analyzer_mod)

        if ml_result:
            meta      = json.load(open(META_FILE))
            cv_auc    = meta.get('cv_auc_mean', 0.5)
            ml_weight = min(0.7, max(0.3, cv_auc))
            rb_weight = 1.0 - ml_weight
            ml_ai     = ml_result['ai_probability'] / 100.0
            blended   = rb_weight * rb_ai + ml_weight * ml_ai
        else:
            blended = rb_ai

    # ── FINAL VERDICT ─────────────────────────────────────────────
    auth = int((1.0 - blended) * 100)
    print(f"\n{'='*60}")
    print(f"  FINAL VERDICT")
    print(f"{'='*60}")

    if blended > 0.55:
        print(f"  [AI]   LIKELY AI-GENERATED")
    elif blended > 0.35:
        print(f"  [?]    POSSIBLY AI-GENERATED (uncertain)")
    else:
        print(f"  [REAL] LIKELY AUTHENTIC PHOTOGRAPH")

    if c2pa_verdict and c2pa_verdict != 'NO_MANIFEST':
        print(f"  C2PA:        {c2pa_verdict}  ({c2pa_conf})")
    print(f"  Forensic:    {rb_ai*100:.0f}% AI")
    if ml_result:
        print(f"  ML model:    {ml_result['ai_probability']:.0f}% AI")
    if ml_skipped_reason:
        print(f"  ML model:    skipped")
    print(f"  Final score: {blended*100:.0f}% AI")
    print(f"  Authenticity: {auth}/100")
    print(f"{'='*60}\n")


# ---------------------------------------------
# DATASET STATUS
# ---------------------------------------------

def show_status():
    print(f"\n{'-'*55}")
    print(f"  Image Trainer  Status")
    print(f"{'-'*55}")
    print(f"  Script dir:  {SCRIPT_DIR}")
    print(f"  Model dir:   {MODEL_DIR}")
    print(f"  Outputs dir: {OUTPUT_DIR}")

    ds = load_dataset()
    dataset_stats(ds)

    if MODEL_FILE.exists():
        meta = json.load(open(META_FILE)) if META_FILE.exists() else {}
        print(f"\n  Trained model: [OK]  (CV AUC={meta.get('cv_auc_mean', '?'):.3f}, "
              f"trained {meta.get('trained','?')[:10]})")
        print(f"  Samples used:  {meta.get('n_samples','?')}")
    else:
        print(f"\n  Trained model: [ERR]  Not yet trained")

    if WEIGHTS_FILE.exists():
        wdata = json.load(open(WEIGHTS_FILE))
        print(f"  Learned weights: [OK]  (will be used by analyzer automatically)")
        top3 = sorted(wdata['weights'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top weighted signals: {', '.join(f'{n} ({w})' for n,w in top3)}")
    else:
        print(f"  Learned weights: [ERR]  Not yet generated (train first)")
    print()




# ---------------------------------------------
# CONFUSION INSPECTOR
# Shows the most-confused images so you can clean the dataset
# ---------------------------------------------

def inspect_confusion(ds: dict, top_n: int = 40):
    """
    Re-run the trained model on all dataset samples and find the most
    confidently wrong predictions  these reveal labeling errors or
    ambiguous images that are hurting model accuracy.
    """
    if not MODEL_FILE.exists():
        print("  [ERR]  No trained model. Run --train first.")
        return

    pipeline = joblib.load(MODEL_FILE)
    meta     = json.load(open(META_FILE))
    binary   = meta.get('binary', True)
    n_exp    = meta.get('n_features_expected', None)

    X, y, paths = prepare_arrays(ds, binary=binary)
    if X is None:
        print("  [ERR]  No samples in dataset.")
        return

    # Validate feature count
    if n_exp and X.shape[1] != n_exp:
        print(f"  [ERR]  Feature mismatch: dataset has {X.shape[1]} features, "
              f"model expects {n_exp}.")
        print("      Delete model/dataset.json and retrain.")
        return

    probas = pipeline.predict_proba(X)[:, 1]
    preds  = (probas > 0.5).astype(int)
    wrong  = preds != y

    # Confidence of wrong predictions  most confident errors are most interesting
    wrong_idx   = np.where(wrong)[0]
    wrong_conf  = np.abs(probas[wrong_idx] - 0.5)   # distance from decision boundary
    sorted_idx  = wrong_idx[np.argsort(-wrong_conf)][:top_n]

    correct_total = (~wrong).sum()
    print(f"\n{'='*60}")
    print(f"  Confusion Inspector")
    print(f"{'='*60}")
    print(f"  Total samples:     {len(y)}")
    print(f"  Correct:           {correct_total} ({correct_total/len(y)*100:.0f}%)")
    print(f"  Wrong:             {wrong.sum()} ({wrong.sum()/len(y)*100:.0f}%)")
    print(f"\n  Showing top {min(top_n, len(sorted_idx))} most confidently wrong predictions:")
    print(f"  (These are the images most worth reviewing / removing)\n")

    label_name = {0: "real", 1: "ai/edited"}
    type_counts = {"real_as_ai": [], "ai_as_real": []}

    print(f"  {'File':<45} {'True':>8} {'Predicted':>10} {'Confidence':>11}")
    print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*11}")

    for idx in sorted_idx:
        path      = paths[idx]
        true_lbl  = label_name[int(y[idx])]
        pred_lbl  = label_name[int(preds[idx])]
        conf      = float(probas[idx])
        fname     = Path(path).name[:44]
        conf_str  = f"{conf:.3f} ({'AI' if conf > 0.5 else 'Real'})"

        print(f"  {fname:<45} {true_lbl:>8} {pred_lbl:>10} {conf_str:>11}")

        key = "real_as_ai" if y[idx] == 0 else "ai_as_real"
        type_counts[key].append(path)

    # Summary by error type
    print(f"\n  Error breakdown:")
    print(f"    Real images predicted as AI:    {len(type_counts['real_as_ai'])}   likely noisy/stock/screenshot in RealArt")
    print(f"    AI images predicted as Real:    {len(type_counts['ai_as_real'])}   likely photo-realistic AI or composites")

    # Save confused image paths to a file for easy review
    confused_path = OUTPUT_DIR / "confused_images.txt"
    with open(confused_path, 'w') as f:
        f.write("# Most confused images  review and consider removing from dataset\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write("# REAL images predicted as AI (check for: stock photos, screenshots, heavy edits)\n")
        for p in type_counts['real_as_ai']:
            f.write(f"REAL_AS_AI\t{p}\n")
        f.write("\n# AI images predicted as Real (check for: photo-composites, photo-realistic AI)\n")
        for p in type_counts['ai_as_real']:
            f.write(f"AI_AS_REAL\t{p}\n")

    print(f"\n  [FILE]  Full list saved -> {confused_path}")
    print(f"      Open this file, review the images, remove the bad ones,")
    print(f"      delete model/dataset.json, and retrain.\n")

    # Also generate a visual grid of the most confused images
    _plot_confused_grid(type_counts, OUTPUT_DIR / "confused_grid.png")


def _plot_confused_grid(type_counts, out_path, max_per_type=12):
    """Save a visual grid of confused images for quick review."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    all_confused = (
        [("REAL predicted as AI", p) for p in type_counts['real_as_ai'][:max_per_type]] +
        [("AI predicted as Real", p) for p in type_counts['ai_as_real'][:max_per_type]]
    )

    if not all_confused:
        return

    cols = 6
    rows = math.ceil(len(all_confused) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.8))
    fig.patch.set_facecolor('#0a0a0f')
    axes = axes.ravel() if rows > 1 else [axes] if cols == 1 else axes.ravel()

    for i, (label, path) in enumerate(all_confused):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((200, 200))
            ax.imshow(np.array(img))
            color = '#ff4757' if 'REAL' in label else '#7c5cfc'
            ax.set_title(f"{label}\n{Path(path).name[:20]}", 
                        fontsize=6, color=color, pad=2)
        except Exception:
            ax.text(0.5, 0.5, 'Load failed', ha='center', va='center',
                   color='white', fontsize=7)
        ax.axis('off')
        ax.set_facecolor('#111118')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3d')

    # Hide unused axes
    for j in range(len(all_confused), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confused Images  Review & Clean Dataset",
                color='#e8e8f0', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, facecolor='#0a0a0f', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [IMG]   Visual grid saved -> {out_path}")

# ---------------------------------------------
# CLI
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate image authenticity ML model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset from folder, train, and evaluate
  python image_trainer.py --dataset dataset/ --train

  # Add a single labeled image to the dataset
  python image_trainer.py --add-image photo.jpg --label real
  python image_trainer.py --add-image midjourney.png --label ai

  # Predict one image using trained model
  python image_trainer.py --predict image.jpg

  # Predict all images in a folder
  python image_trainer.py --predict-dir test_images/

  # Full pipeline (analyzer + ML model together)
  python image_trainer.py --analyze image.jpg --verbose

  # Show dataset and model status
  python image_trainer.py --status
        """
    )

    parser.add_argument("--dataset",      metavar="DIR",
                        help="Folder with real/ ai/ edited/ subfolders to build dataset")
    parser.add_argument("--dataset2",     metavar="DIR",
                        help="Second dataset to merge (e.g. CIFAKE alongside GenImage)")
    parser.add_argument("--max-per-class2", type=int, default=0,
                        help="Max images per class for --dataset2 (default: same as --max-per-class)")
    parser.add_argument("--multi-dataset", action="store_true",
                        help="Crawl all subfolders of --dataset for ai/nature pairs (GenImage structure)")
    parser.add_argument("--train",        action="store_true",
                        help="Train the ML model on the current dataset")
    parser.add_argument("--add-image",    metavar="FILE",
                        help="Add a single image to the dataset")
    parser.add_argument("--label",        metavar="LABEL",
                        help="Label for --add-image: real | ai | edited")
    parser.add_argument("--predict",      metavar="FILE",
                        help="Predict a single image using trained model")
    parser.add_argument("--predict-dir",  metavar="DIR",
                        help="Batch predict all images in a folder")
    parser.add_argument("--analyze",      metavar="FILE",
                        help="Run full pipeline: analyzer + ML model")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose signal output")
    parser.add_argument("--status",       action="store_true",
                        help="Show dataset and model status")
    parser.add_argument("--multiclass",   action="store_true",
                        help="Train 3-class model (real/ai/edited) instead of binary")
    parser.add_argument("--inspect",      action="store_true",
                        help="Show most confused predictions to help clean the dataset")
    parser.add_argument("--top-n",        type=int, default=40,
                        help="Number of confused images to show (default: 40)")
    parser.add_argument("--clear-cache",  action="store_true",
                        help="Delete cached feature vectors and force full re-extraction")

    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Max images per class (0=all). Use 2000 for CIFAKE.")
    parser.add_argument("--split", type=str, default=None,
                        help="Only scan a specific subfolder e.g. 'train' or 'test'")
    parser.add_argument("--model", type=str, default="default", metavar="SLOT",
                        help="Model slot to use/save: default | genimage | cifake | combined | any name")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available trained model slots")

    args = parser.parse_args()

    # Load analyzer module once
    analyzer = _import_analyzer()

    # Apply model slot FIRST so all subsequent operations use correct paths
    if args.model != "default":
        set_model_slot(args.model)

    if args.list_models:
        slots = list_model_slots()
        if not slots:
            print("  No trained models found.")
        else:
            print(f"\n  Available model slots ({len(slots)}):")
            print(f"  {'Slot':<16} {'Path':<50} {'Status'}")
            print("  " + "-" * 80)
            for slot_name, slot_dir in slots:
                meta_f = slot_dir / "meta.json"
                cnn_f  = slot_dir / "cnn_classifier.joblib"
                if meta_f.exists():
                    import json as _j
                    m = _j.load(open(meta_f))
                    auc   = m.get("cv_auc_mean", 0)
                    n     = m.get("n_samples", 0)
                    label = f"RF AUC={auc:.3f}  n={n}"
                else:
                    label = "RF only (no meta)"
                cnn_label = " + CNN" if cnn_f.exists() else ""
                active = " <- ACTIVE" if slot_name == _ACTIVE_SLOT else ""
                print(f"  {slot_name:<16} {str(slot_dir):<50} {label}{cnn_label}{active}")
        print()
        if not any([args.dataset, args.train, args.predict, args.predict_dir, args.analyze]):
            sys.exit(0)

    if args.clear_cache:
        if DATASET_FILE.exists():
            DATASET_FILE.unlink()
            print(f"  [DELETE]   Cleared feature cache: {DATASET_FILE}")
        else:
            print(f"     No cache found at {DATASET_FILE}")
        if not any([args.dataset, args.train]):
            sys.exit(0)

    if args.status or not any([args.dataset, args.train, args.add_image,
                                args.predict, args.predict_dir, args.analyze,
                                args.list_models]):
        show_status()

    if args.dataset:
        print(f"\n  [DATASET 1]  {args.dataset}")

        if args.multi_dataset:
            # Crawl all subfolders (e.g. imagenet_ai_0419_biggan/, imagenet_midjourney/)
            # Each contains train/ai/ and train/nature/ -- collect from all of them
            root = Path(args.dataset)
            subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
            print(f"  Found {len(subdirs)} subfolders -- crawling each for ai/nature pairs...")

            ds = {"samples": [], "feature_names": [], "feature_dim": 0}
            # Track per-class counts to enforce cap across all subfolders
            class_counts = {"real": 0, "ai": 0}
            cap = args.max_per_class  # per class cap TOTAL across all subfolders
            per_sub_cap = max(1, cap // len(subdirs)) if cap else 0

            for sub in subdirs:
                split_dir = sub / (args.split or "train")
                if not split_dir.exists():
                    split_dir = sub  # fallback: no split subfolder
                print(f"  Scanning {sub.name}/ ({split_dir.name}/)...")
                sub_ds = build_dataset_from_folder(str(split_dir), analyzer,
                                                   max_per_class=per_sub_cap,
                                                   split_filter=None)
                # Merge, respecting total cap
                for sample in sub_ds["samples"]:
                    lstr = sample["label_str"]
                    if cap and class_counts.get(lstr, 0) >= cap:
                        continue
                    ds["samples"].append(sample)
                    class_counts[lstr] = class_counts.get(lstr, 0) + 1
                if sub_ds.get("feature_names"):
                    ds["feature_names"] = sub_ds["feature_names"]
                    ds["feature_dim"]   = sub_ds["feature_dim"]

            real_n = class_counts.get("real", 0)
            ai_n   = class_counts.get("ai", 0)
            print(f"\n  Multi-dataset crawl complete: {real_n} real + {ai_n} AI = {real_n+ai_n} total")

            # Save merged dataset
            import json as _json
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(DATASET_FILE, "w") as _f:
                _json.dump(ds, _f)
            print(f"  Saved to {DATASET_FILE}")

        else:
            ds = build_dataset_from_folder(args.dataset, analyzer,
                                           max_per_class=args.max_per_class,
                                           split_filter=args.split)

        if args.dataset2:
            cap2 = args.max_per_class2 or args.max_per_class
            print(f"\n  [DATASET 2]  {args.dataset2}  (max_per_class={cap2 or 'all'})")
            ds2 = build_dataset_from_folder(args.dataset2, analyzer,
                                            max_per_class=cap2,
                                            split_filter=args.split)

            # Merge: combine samples, deduplicate by file path
            existing_paths = {s["path"] for s in ds["samples"]}
            added = 0
            for sample in ds2["samples"]:
                if sample["path"] not in existing_paths:
                    ds["samples"].append(sample)
                    existing_paths.add(sample["path"])
                    added += 1

            # Recount
            real_count = sum(1 for s in ds["samples"] if s["label_str"] == "real")
            ai_count   = sum(1 for s in ds["samples"] if s["label_str"] == "ai")
            print(f"\n  Merged dataset: {real_count} real + {ai_count} AI = {len(ds['samples'])} total")
            print(f"  ({added} new samples added from dataset2)")

            # Save merged dataset
            import json as _json
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(DATASET_FILE, "w") as _f:
                _json.dump(ds, _f)
            print(f"  Merged dataset saved to {DATASET_FILE}")

        if args.train:
            train(ds, binary=not args.multiclass)

    elif args.train:
        ds = load_dataset()
        train(ds, binary=not args.multiclass)

    if args.add_image:
        if not args.label:
            print("  [ERR]  --add-image requires --label (real | ai | edited)")
            sys.exit(1)
        add_single_image(args.add_image, args.label, analyzer)

    if args.predict:
        predict_single(args.predict, analyzer)

    if args.predict_dir:
        predict_batch(args.predict_dir, analyzer)

    if args.analyze:
        run_full_pipeline(args.analyze, analyzer, verbose=args.verbose)

    if args.inspect:
        ds = load_dataset()
        inspect_confusion(ds, top_n=args.top_n)