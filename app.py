"""
Image Authenticity Web Interface
=================================
Flask web server that wraps the full 3-stage pipeline:
  Stage 1: C2PA provenance check
  Stage 2: 17 forensic signal modules
  Stage 3: ML model (RF + CNN calibrated blend)

Usage:
  py -3.12 app.py
  Then open: http://localhost:5000

Requirements:
  py -3.12 -m pip install flask
  (torch, scikit-learn etc. already installed from training)
"""

import os
import sys
import json
import time
import base64
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
UPLOAD_DIR  = SCRIPT_DIR / "uploads"
OUTPUT_DIR  = SCRIPT_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp'}

app = Flask(__name__, static_folder=str(SCRIPT_DIR))


# ── Module loader ──────────────────────────────────────────────────────────────
_trainer  = None
_analyzer = None

def _load_modules():
    global _trainer, _analyzer
    if _trainer and _analyzer:
        return _trainer, _analyzer

    trainer_path  = SCRIPT_DIR / "image_trainer.py"
    analyzer_path = SCRIPT_DIR / "image_authenticity.py"

    if not trainer_path.exists():
        raise RuntimeError(f"image_trainer.py not found in {SCRIPT_DIR}")
    if not analyzer_path.exists():
        raise RuntimeError(f"image_authenticity.py not found in {SCRIPT_DIR}")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _trainer  = load("image_trainer",       trainer_path)
    _analyzer = load("image_authenticity",  analyzer_path)
    return _trainer, _analyzer


# ── Analysis runner ────────────────────────────────────────────────────────────
def run_analysis(filepath: str, model_slot: str = 'default') -> dict:
    """Run full 3-stage pipeline and return structured result dict."""
    trainer, analyzer = _load_modules()
    result = {
        "file":       Path(filepath).name,
        "timestamp":  datetime.now().isoformat(),
        "c2pa":       None,
        "forensic":   None,
        "ml":         None,
        "final":      None,
        "error":      None,
    }

    t0 = time.perf_counter()

    # Switch model slot if requested
    if model_slot != "default":
        trainer.set_model_slot(model_slot)
    result["model_slot"] = model_slot

    try:
        import joblib

        # ── Stage 1: C2PA ──────────────────────────────────────────
        c2pa_path = SCRIPT_DIR / "c2pa_checker.py"
        if c2pa_path.exists():
            spec = importlib.util.spec_from_file_location("c2pa_checker", c2pa_path)
            c2pa_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(c2pa_mod)
            c2pa_raw = c2pa_mod.check_c2pa(filepath)
            result["c2pa"] = {
                "verdict":    c2pa_raw.get("verdict"),
                "confidence": c2pa_raw.get("confidence"),
                "summary":    c2pa_raw.get("summary"),
                "ai_tool":    c2pa_raw.get("details", {}).get("ai_tool"),
                "camera":     c2pa_raw.get("details", {}).get("camera_manufacturer"),
                "found_in":   c2pa_raw.get("raw_found", []),
            }

            # Short-circuit on definitive C2PA
            v = c2pa_raw.get("verdict")
            if v == "VERIFIED_AI":
                result["final"] = {"verdict": "AI", "score": 2,
                                    "ai_prob": 98, "source": "c2pa"}
                result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                return result
            if v == "VERIFIED_REAL" and c2pa_raw.get("confidence") == "HIGH":
                result["final"] = {"verdict": "REAL", "score": 98,
                                    "ai_prob": 2, "source": "c2pa"}
                result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                return result

        # ── Stage 2: Forensic signals ──────────────────────────────
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report = analyzer.analyze(filepath, verbose=False, save_report=True)

        signals = []
        for s in report.signals:
            signals.append({
                "name":       s.name,
                "score":      round(float(s.score), 3),
                "category":   s.category,
                "confidence": s.confidence,
                "detail":     s.detail,
            })

        result["forensic"] = {
            "ai_prob":    round(float(report.ai_probability) * 100, 1),
            "edit_prob":  round(float(report.edit_probability) * 100, 1),
            "auth_score": round(float(report.authenticity_score) * 100),
            "signals":    signals,
            "verdict":    report.verdict,
        }

        rb_ai = float(report.ai_probability)

        # ── Stage 3: ML model (only if ambiguous) ──────────────────
        # Find model dir -- check script location, parent, and D:\Coding
        def _find_model_dir():
            candidates = [
                SCRIPT_DIR / "model",
                SCRIPT_DIR.parent / "model",
                Path("D:/Coding/model"),
            ]
            for c in candidates:
                if (c / "classifier.joblib").exists():
                    return c
            return SCRIPT_DIR / "model"

        found_model_dir = _find_model_dir()
        model_file = found_model_dir / "classifier.joblib"
        ml_data    = None

        if model_file.exists() and 0.30 <= rb_ai <= 0.70:
            vec = trainer.extract_feature_vector(filepath, analyzer)
            if vec is not None:
                hc_vec, cnn_vec = trainer.split_feature_vector(vec)
                pipeline  = joblib.load(model_file)
                meta      = json.load(open(found_model_dir / "meta.json"))
                hc_auc    = meta.get("cv_auc_mean", 0.5)
                rf_probas = pipeline.predict_proba(hc_vec.reshape(1, -1))[0]
                rf_prob   = float(rf_probas[1])

                cnn_prob_raw = None
                cnn_prob_cal = None
                disagree     = None
                uncertain    = False
                blended      = rf_prob

                cnn_file = found_model_dir / "cnn_classifier.joblib"
                if cnn_file.exists() and cnn_vec.sum() != 0:
                    cnn_pipe   = joblib.load(cnn_file)
                    cnn_meta   = json.load(open(found_model_dir / "cnn_meta.json"))
                    cnn_probas = cnn_pipe.predict_proba(cnn_vec.reshape(1, -1))[0]
                    cnn_ai_raw = float(cnn_probas[1])
                    cnn_auc    = cnn_meta.get("cv_auc", 0.5)
                    blended, cnn_cal, disagree, uncertain = trainer._blend_predictions(
                        rf_prob, cnn_ai_raw, hc_auc, cnn_auc)

                    # If forensic says AI but CNN says real, CNN is likely miscalibrated
                    if rf_prob > 0.45 and cnn_cal < 0.40:
                        blended   = 0.85 * rf_prob + 0.15 * cnn_cal
                        uncertain = True

                    cnn_prob_raw = round(cnn_ai_raw * 100, 1)
                    cnn_prob_cal = round(cnn_cal    * 100, 1)

                ml_data = {
                    "rf_prob":      round(rf_prob  * 100, 1),
                    "cnn_prob_raw": cnn_prob_raw,
                    "cnn_prob_cal": cnn_prob_cal,
                    "blended":      round(blended  * 100, 1),
                    "disagreement": round(disagree * 100, 1) if disagree else None,
                    "uncertain":    uncertain,
                    "rf_auc":       round(hc_auc, 3),
                    "stage_skipped": False,
                }
        elif model_file.exists():
            ml_data = {"stage_skipped": True,
                       "skip_reason": f"Stage 2 clear ({rb_ai*100:.0f}% AI)"}

        result["ml"] = ml_data

        # ── Final blended verdict ──────────────────────────────────
        if ml_data and not ml_data.get("stage_skipped") and ml_data.get("blended") is not None:
            final_ai = ml_data["blended"]
        else:
            final_ai = rb_ai * 100

        auth_score = int(100 - final_ai)
        if final_ai > 55:
            verdict = "AI"
        elif final_ai > 35:
            verdict = "UNCERTAIN"
        else:
            verdict = "REAL"

        result["final"] = {
            "verdict":   verdict,
            "ai_prob":   round(final_ai, 1),
            "score":     auth_score,
            "source":    "ml" if (ml_data and not ml_data.get("stage_skipped")) else "forensic",
        }

        # Attach chart if saved
        chart_path = OUTPUT_DIR / f"{Path(filepath).stem}_analysis.png"
        if chart_path.exists():
            with open(chart_path, "rb") as f:
                result["chart_b64"] = base64.b64encode(f.read()).decode()

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_PAGE

@app.route("/models")
def list_models():
    """Return available model slots for the UI dropdown."""
    trainer, _ = _load_modules()
    slots = trainer.list_model_slots()
    result = []
    for slot_name, slot_dir in slots:
        meta_f = slot_dir / "meta.json"
        cnn_f  = slot_dir / "cnn_classifier.joblib"
        info = {"slot": slot_name, "path": str(slot_dir), "has_cnn": cnn_f.exists()}
        if meta_f.exists():
            import json as _j
            m = _j.load(open(meta_f))
            info["rf_auc"]   = m.get("cv_auc_mean", 0)
            info["n_samples"]= m.get("n_samples", 0)
            info["trained"]  = m.get("trained_at", "")
        result.append(info)
    return jsonify(result)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f    = request.files["file"]
    ext  = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    slot      = request.form.get("model_slot", "default")
    safe_name = secure_filename(f.filename)
    save_path = UPLOAD_DIR / safe_name
    f.save(str(save_path))

    try:
        result = run_analysis(str(save_path), model_slot=slot)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        try:
            save_path.unlink()
        except Exception:
            pass


# ── HTML/CSS/JS (single-file UI) ───────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lens — Image Authenticity</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080c10;
    --surface:  #0e1420;
    --border:   #1e2a3a;
    --text:     #c8d8e8;
    --muted:    #4a6070;
    --accent:   #00d4ff;
    --real:     #00ff88;
    --ai:       #ff3a5c;
    --warn:     #ffaa00;
    --font-hd:  'Syne', sans-serif;
    --font-mono:'DM Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrapper {
    position: relative;
    z-index: 1;
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px 80px;
  }

  /* Header */
  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 56px;
  }

  .logo {
    font-family: var(--font-hd);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), #7b5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .tagline {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* Drop zone */
  .dropzone {
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 64px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    background: var(--surface);
    position: relative;
    overflow: hidden;
  }

  .dropzone::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
  }

  .dropzone:hover, .dropzone.drag-over {
    border-color: var(--accent);
    background: #0e1a24;
  }

  .dropzone.drag-over { transform: scale(1.01); }

  .drop-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    display: block;
    opacity: 0.6;
  }

  .drop-title {
    font-family: var(--font-hd);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
  }

  .drop-sub {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.05em;
  }

  #file-input { display: none; }

  .btn-browse {
    display: inline-block;
    margin-top: 20px;
    padding: 10px 24px;
    border: 1px solid var(--accent);
    border-radius: 6px;
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: all 0.2s;
    background: transparent;
  }

  .btn-browse:hover {
    background: var(--accent);
    color: var(--bg);
  }

  /* Preview */
  #preview-wrap {
    display: none;
    margin-top: 24px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    position: relative;
  }

  #preview-img {
    width: 100%;
    max-height: 360px;
    object-fit: cover;
    display: block;
  }

  .preview-overlay {
    position: absolute;
    top: 12px; right: 12px;
    background: rgba(8,12,16,0.85);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.72rem;
    color: var(--muted);
  }

  /* Spinner */
  #spinner {
    display: none;
    text-align: center;
    padding: 48px 0;
  }

  .spin-ring {
    width: 48px; height: 48px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .spin-label {
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.1em;
  }

  /* Results */
  #results { display: none; margin-top: 32px; }

  /* Verdict card */
  .verdict-card {
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    border: 1.5px solid;
  }

  .verdict-card.real {
    background: rgba(0,255,136,0.04);
    border-color: rgba(0,255,136,0.25);
  }

  .verdict-card.ai {
    background: rgba(255,58,92,0.04);
    border-color: rgba(255,58,92,0.25);
  }

  .verdict-card.uncertain {
    background: rgba(255,170,0,0.04);
    border-color: rgba(255,170,0,0.25);
  }

  .verdict-label {
    font-family: var(--font-hd);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.6;
  }

  .verdict-main {
    font-family: var(--font-hd);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 4px;
  }

  .real .verdict-main  { color: var(--real); }
  .ai .verdict-main    { color: var(--ai); }
  .uncertain .verdict-main { color: var(--warn); }

  .verdict-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 8px;
  }

  /* Score arc */
  .score-wrap {
    position: absolute;
    top: 24px; right: 28px;
    text-align: center;
  }

  .score-arc {
    width: 80px; height: 80px;
    transform: rotate(-90deg);
  }

  .arc-bg   { fill: none; stroke: var(--border); stroke-width: 6; }
  .arc-fill { fill: none; stroke-width: 6; stroke-linecap: round;
               transition: stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1); }

  .score-num {
    font-family: var(--font-hd);
    font-size: 1.3rem;
    font-weight: 700;
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
  }

  /* Stage pipeline */
  .pipeline {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }

  .stage-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 16px;
    transition: border-color 0.3s;
  }

  .stage-card.active   { border-color: var(--accent); }
  .stage-card.skipped  { opacity: 0.4; }
  .stage-card.verified-real { border-color: var(--real); }
  .stage-card.verified-ai   { border-color: var(--ai); }

  .stage-num {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  .stage-name {
    font-family: var(--font-hd);
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 10px;
  }

  .stage-result {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.5;
  }

  .stage-result .val {
    color: var(--text);
    font-weight: 500;
  }

  /* Signal bars */
  .signals-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
  }

  .section-title {
    font-family: var(--font-hd);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 18px;
  }

  .signal-row {
    display: grid;
    grid-template-columns: 180px 1fr 52px;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 0.72rem;
  }

  .signal-name { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .signal-bar-wrap {
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }

  .signal-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
  }

  .signal-val { text-align: right; color: var(--muted); }

  .cat-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 16px 0 10px;
    opacity: 0.6;
  }

  /* Chart */
  .chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 20px;
  }

  .chart-wrap img {
    width: 100%;
    display: block;
  }

  /* Elapsed */
  .elapsed {
    text-align: right;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 8px;
  }

  /* Error */
  .error-box {
    background: rgba(255,58,92,0.08);
    border: 1px solid rgba(255,58,92,0.3);
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 0.8rem;
    color: #ff8a9a;
    white-space: pre-wrap;
  }

  /* Model selector bar */
  .model-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 14px;
    padding: 10px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
  }

  .model-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .model-slots { display: flex; gap: 8px; flex-wrap: wrap; }

  .slot-btn {
    padding: 5px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .slot-btn:hover { border-color: var(--accent); color: var(--text); }
  .slot-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,0.07); }

  .model-hint {
    font-size: 0.68rem;
    color: var(--muted);
    margin-left: auto;
    white-space: nowrap;
  }

  /* Responsive */
  @media (max-width: 600px) {
    .pipeline { grid-template-columns: 1fr; }
    .signal-row { grid-template-columns: 130px 1fr 44px; }
    .score-wrap { position: static; margin-top: 16px; }
    .verdict-main { font-size: 1.8rem; }
  }

  /* Animations */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-up { animation: fadeUp 0.4s ease forwards; }
  .fade-up-2 { animation: fadeUp 0.4s 0.1s ease forwards; opacity: 0; }
  .fade-up-3 { animation: fadeUp 0.4s 0.2s ease forwards; opacity: 0; }
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <span class="logo">LENS</span>
    <span class="tagline">Image Authenticity Detector</span>
  </header>

  <!-- Drop Zone -->
  <div class="dropzone" id="dropzone">
    <span class="drop-icon">⬡</span>
    <div class="drop-title">Drop an image to analyze</div>
    <div class="drop-sub">JPG · PNG · WEBP · TIFF — any resolution</div>
    <button class="btn-browse" onclick="document.getElementById('file-input').click()">
      Browse files
    </button>
    <input type="file" id="file-input" accept="image/*">
  </div>

  <!-- Model selector -->
  <div class="model-bar" id="model-bar">
    <span class="model-label">MODEL</span>
    <div class="model-slots" id="model-slots">
      <button class="slot-btn active" data-slot="default">default</button>
    </div>
    <span class="model-hint" id="model-hint">Loading slots...</span>
  </div>

  <!-- Preview -->
  <div id="preview-wrap">
    <img id="preview-img" src="" alt="preview">
    <div class="preview-overlay" id="preview-info">—</div>
  </div>

  <!-- Spinner -->
  <div id="spinner">
    <div class="spin-ring"></div>
    <div class="spin-label" id="spin-label">RUNNING C2PA CHECK</div>
  </div>

  <!-- Results -->
  <div id="results"></div>

</div>

<script>
const dropzone   = document.getElementById('dropzone');
let activeSlot = 'default';

// Load available model slots on startup
async function loadModelSlots() {
  try {
    const resp  = await fetch('/models');
    const slots = await resp.json();
    const container = document.getElementById('model-slots');
    const hint      = document.getElementById('model-hint');

    if (!slots.length) {
      hint.textContent = 'No models found';
      return;
    }

    container.innerHTML = '';
    slots.forEach(s => {
      const btn = document.createElement('button');
      btn.className = 'slot-btn' + (s.slot === activeSlot ? ' active' : '');
      btn.dataset.slot = s.slot;
      btn.textContent  = s.slot;
      btn.title = `n=${s.n_samples||'?'}  RF AUC=${s.rf_auc ? s.rf_auc.toFixed(3) : '?'}${s.has_cnn ? ' + CNN' : ''}`;
      btn.addEventListener('click', () => {
        activeSlot = s.slot;
        container.querySelectorAll('.slot-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const active = slots.find(x => x.slot === activeSlot);
        if (active) hint.textContent = `n=${active.n_samples||'?'}  RF AUC=${active.rf_auc ? active.rf_auc.toFixed(3) : '?'}${active.has_cnn ? ' + CNN' : ''}`;
      });
      container.appendChild(btn);
    });

    const first = slots[0];
    hint.textContent = `n=${first.n_samples||'?'}  RF AUC=${first.rf_auc ? first.rf_auc.toFixed(3) : '?'}${first.has_cnn ? ' + CNN' : ''}`;
  } catch(e) {
    document.getElementById('model-hint').textContent = 'Could not load slots';
  }
}
loadModelSlots();


const fileInput  = document.getElementById('file-input');
const previewWrap = document.getElementById('preview-wrap');
const previewImg = document.getElementById('preview-img');
const previewInfo = document.getElementById('preview-info');
const spinner    = document.getElementById('spinner');
const spinLabel  = document.getElementById('spin-label');
const resultsDiv = document.getElementById('results');

// Drag & drop
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropzone.addEventListener('click', e => {
  if (e.target.classList.contains('btn-browse')) return;
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.style.display = 'block';
    previewInfo.textContent = `${file.name}  ·  ${(file.size/1024).toFixed(0)} KB`;
  };
  reader.readAsDataURL(file);

  // Upload & analyze
  analyzeFile(file);
}

const spinMessages = [
  'RUNNING C2PA CHECK',
  'ANALYZING FREQUENCY DOMAIN',
  'CHECKING NOISE PROFILE',
  'RUNNING ELA ANALYSIS',
  'CHECKING TEXTURE SIGNALS',
  'LOADING CNN MODEL',
  'RUNNING EFFICIENTNET',
  'CALIBRATING PREDICTIONS',
  'BLENDING MODELS',
];

async function analyzeFile(file) {
  resultsDiv.style.display = 'none';
  resultsDiv.innerHTML = '';
  spinner.style.display = 'block';

  // Cycle spinner messages
  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    msgIdx = (msgIdx + 1) % spinMessages.length;
    spinLabel.textContent = spinMessages[msgIdx];
  }, 1200);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_slot', activeSlot);

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await resp.json();
    clearInterval(msgInterval);
    spinner.style.display = 'none';
    renderResults(data);
  } catch (err) {
    clearInterval(msgInterval);
    spinner.style.display = 'none';
    resultsDiv.innerHTML = `<div class="error-box">Request failed: ${err.message}</div>`;
    resultsDiv.style.display = 'block';
  }
}

function renderResults(data) {
  if (data.error) {
    resultsDiv.innerHTML = `<div class="error-box fade-up">Error: ${data.error}\n\n${data.traceback||''}</div>`;
    resultsDiv.style.display = 'block';
    return;
  }

  const final   = data.final || {};
  const verdict = final.verdict || 'UNCERTAIN';
  const aiProb  = final.ai_prob || 0;
  const score   = final.score || 50;

  // Verdict card
  const verdictClass = verdict === 'REAL' ? 'real' : verdict === 'AI' ? 'ai' : 'uncertain';
  const verdictText  = verdict === 'REAL' ? 'AUTHENTIC' : verdict === 'AI' ? 'AI GENERATED' : 'UNCERTAIN';
  const verdictColor = verdict === 'REAL' ? 'var(--real)' : verdict === 'AI' ? 'var(--ai)' : 'var(--warn)';
  const slotLabel    = data.model_slot && data.model_slot !== 'default' ? ` · model: ${data.model_slot}` : '';
  const sourceLabel  = (final.source === 'c2pa' ? 'Verified by C2PA cryptographic signature' :
                       final.source === 'ml'   ? 'Determined by ML + forensic analysis' :
                                                 'Determined by forensic signal analysis') + slotLabel;

  // Score arc math (circumference of r=33 circle = 207.3)
  const circ   = 207.3;
  const offset = circ - (score / 100) * circ;

  let html = `
  <div class="verdict-card ${verdictClass} fade-up">
    <div class="verdict-label">Final Verdict</div>
    <div class="verdict-main">${verdictText}</div>
    <div class="verdict-sub">${sourceLabel}</div>
    <div class="verdict-sub" style="margin-top:4px; font-size:0.72rem;">
      AI probability: <span style="color:${verdictColor}">${aiProb.toFixed(1)}%</span>
      &nbsp;·&nbsp; Elapsed: ${data.elapsed_ms || '—'} ms
    </div>

    <div class="score-wrap">
      <div style="position:relative; width:80px; height:80px;">
        <svg class="score-arc" viewBox="0 0 80 80">
          <circle class="arc-bg"   cx="40" cy="40" r="33"/>
          <circle class="arc-fill" cx="40" cy="40" r="33"
            stroke="${verdictColor}"
            stroke-dasharray="${circ}"
            stroke-dashoffset="${offset}"
            id="arc-fill"/>
        </svg>
        <div class="score-num" style="color:${verdictColor}">${score}</div>
      </div>
      <div style="font-size:0.6rem; color:var(--muted); margin-top:2px; letter-spacing:0.08em;">AUTHENTIC</div>
    </div>
  </div>`;

  // Pipeline stages
  const c2pa = data.c2pa;
  const forensic = data.forensic;
  const ml = data.ml;

  const c2paClass = !c2pa ? 'skipped' :
    c2pa.verdict === 'VERIFIED_REAL' ? 'verified-real active' :
    c2pa.verdict === 'VERIFIED_AI'   ? 'verified-ai active' : 'active';

  const c2paResult = !c2pa ? '<span class="val">Not available</span>' :
    c2pa.verdict === 'NO_MANIFEST' ? '<span class="val">No manifest found</span>' :
    `<span class="val">${c2pa.verdict}</span>${c2pa.ai_tool ? `<br>Tool: <span class="val">${c2pa.ai_tool}</span>` : ''}${c2pa.camera ? `<br>Camera: <span class="val">${c2pa.camera}</span>` : ''}`;

  const forensicResult = !forensic ? '<span class="val">—</span>' :
    `AI prob: <span class="val">${forensic.ai_prob}%</span><br>Auth score: <span class="val">${forensic.auth_score}/100</span>`;

  const mlSkipped = ml && ml.stage_skipped;
  const mlResult = !ml ? '<span class="val">No model loaded</span>' :
    mlSkipped ? `<span class="val">Skipped</span><br><span style="font-size:0.68rem">${ml.skip_reason}</span>` :
    `RF: <span class="val">${ml.rf_prob}%</span>&nbsp; CNN: <span class="val">${ml.cnn_prob_cal ?? '—'}%</span><br>Blended: <span class="val">${ml.blended}%</span>${ml.uncertain ? '<br><span style="color:var(--warn)">⚠ Models disagree</span>' : ''}`;

  html += `
  <div class="pipeline fade-up-2">
    <div class="stage-card ${c2paClass}">
      <div class="stage-num">Stage 1</div>
      <div class="stage-name">C2PA Provenance</div>
      <div class="stage-result">${c2paResult}</div>
    </div>
    <div class="stage-card ${forensic ? 'active' : 'skipped'}">
      <div class="stage-num">Stage 2</div>
      <div class="stage-name">Forensic Signals</div>
      <div class="stage-result">${forensicResult}</div>
    </div>
    <div class="stage-card ${ml && !mlSkipped ? 'active' : 'skipped'}">
      <div class="stage-num">Stage 3</div>
      <div class="stage-name">ML Model</div>
      <div class="stage-result">${mlResult}</div>
    </div>
  </div>`;

  // Signals breakdown
  if (forensic && forensic.signals && forensic.signals.length > 0) {
    const cats = { ai_detection: '🤖 AI Detection', edit_detection: '✂️  Edit Detection', metadata: '📋 Metadata' };
    html += `<div class="signals-section fade-up-3"><div class="section-title">Signal Breakdown</div>`;

    for (const [catKey, catLabel] of Object.entries(cats)) {
      const catSigs = forensic.signals.filter(s => s.category === catKey);
      if (!catSigs.length) continue;
      html += `<div class="cat-label">${catLabel}</div>`;
      for (const s of catSigs) {
        const pct  = Math.round(s.score * 100);
        const col  = s.score > 0.6 ? 'var(--ai)' : s.score > 0.35 ? 'var(--warn)' : 'var(--real)';
        html += `
        <div class="signal-row">
          <div class="signal-name" title="${s.detail || ''}">${s.name}</div>
          <div class="signal-bar-wrap">
            <div class="signal-bar" style="width:${pct}%; background:${col}"></div>
          </div>
          <div class="signal-val">${s.score.toFixed(2)}</div>
        </div>`;
      }
    }
    html += `</div>`;
  }

  // Chart
  if (data.chart_b64) {
    html += `<div class="chart-wrap fade-up-3">
      <img src="data:image/png;base64,${data.chart_b64}" alt="Analysis chart">
    </div>`;
  }

  resultsDiv.innerHTML = html;
  resultsDiv.style.display = 'block';

  // Animate arc after render
  requestAnimationFrame(() => {
    const arc = document.getElementById('arc-fill');
    if (arc) {
      arc.style.strokeDashoffset = circ; // start from 0
      setTimeout(() => { arc.style.strokeDashoffset = offset; }, 50);
    }
  });
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LENS — Image Authenticity Web Interface")
    print("="*50)
    print(f"  Script dir: {SCRIPT_DIR}")
    print(f"  Open:  http://localhost:5000")
    print(f"  Stop:  Ctrl+C")
    print("="*50 + "\n")

    # Pre-load modules at startup
    try:
        _load_modules()
        print("  [OK] Pipeline modules loaded")
    except Exception as e:
        print(f"  [WARN] Module pre-load failed: {e}")
        print("         Will retry on first request")

    app.run(host="0.0.0.0", port=5000, debug=False)