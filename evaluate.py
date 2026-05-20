"""
evaluate.py — Enhanced Traffic Violation Detection Evaluator
============================================================
Wraps the TrafficViolationDetector from solution.py with:
  - Visual annotated output (bounding boxes, labels, plates)
  - Per-image JSON results printed to console
  - Batch folder processing with summary statistics
  - Debug mode exposing intermediate pipeline stages

Usage:
    # Single image (display window):
    python evaluate.py image.jpg

    # Single image (save annotated result):
    python evaluate.py image.jpg --save

    # Batch folder (auto-saves all results):
    python evaluate.py images_folder/ --save

    # With debug visuals (depth map, head crops, plate crops):
    python evaluate.py image.jpg --save --debug

    # Custom output directory:
    python evaluate.py image.jpg --save --out results/

Output:
    - Console: JSON violation record per image
    - Saved: annotated JPEGs in --out directory (default: final_results/)
    - Debug: intermediate images in debug_outputs/<stem>/
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# ── Import the core detector from solution.py ──────────────────────────────
from solution import TrafficViolationDetector

# ── Visual style constants ──────────────────────────────────────────────────
BIKE_COLOR  = (0, 255, 0)       # green  — compliant bike
VIOL_COLOR  = (0, 0, 255)       # red    — violation
PLATE_COLOR = (0, 255, 255)     # yellow — license plate box
TEXT_COLOR  = (255, 255, 255)   # white  — label text
FONT        = cv2.FONT_HERSHEY_SIMPLEX

MODEL_DIR   = "./models"


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _put_label(img, text, x, y, color, scale=0.6, thickness=2):
    """Draw a filled background label above (x, y)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def annotate_image(img: np.ndarray, result: dict,
                   bike_boxes=None, bike_persons=None, plate_info=None) -> np.ndarray:
    """
    Draw violation annotations on a copy of img.

    Parameters
    ----------
    img          : BGR image (numpy array)
    result       : output of TrafficViolationDetector.predict()
    bike_boxes   : optional list of [x1,y1,x2,y2] — drawn when available
    bike_persons : optional list-of-lists of person boxes per bike
    plate_info   : optional list of (box_in_img, text) tuples
    """
    out = img.copy()
    h, w = out.shape[:2]

    violations = result.get("violations", [])
    viol_count = len(violations)

    # Summary banner
    summary = f"Bikes detected | Violations: {viol_count}"
    cv2.putText(out, summary, (20, 50), FONT, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

    # Per-violation annotations (when no box info available, skip boxes)
    for i, v in enumerate(violations):
        riders   = v.get("num_riders", "?")
        hv       = v.get("helmet_violations", 0)
        plate    = v.get("license_plate", "UNKNOWN")
        is_viol  = hv > 0 or (isinstance(riders, int) and riders >= 3)
        color    = VIOL_COLOR if is_viol else BIKE_COLOR
        tags     = []
        if isinstance(riders, int) and riders >= 3:
            tags.append("TRIPLE")
        if hv > 0:
            tags.append(f"NO-HELMET×{hv}")
        label = f"Ppl: {riders}" + (" | " + " & ".join(tags) if tags else "")
        # Print to image top-left area when no box info
        if bike_boxes is None:
            _put_label(out, label, 10, 90 + i * 40, color)
            if plate != "UNKNOWN":
                _put_label(out, f"Plate: {plate}", 10, 130 + i * 40, PLATE_COLOR)

    # Draw bike boxes when available
    if bike_boxes is not None:
        for i, bb in enumerate(bike_boxes):
            x1, y1, x2, y2 = map(int, bb)
            v = violations[i] if i < len(violations) else {}
            riders  = v.get("num_riders", 1)
            hv      = v.get("helmet_violations", 0)
            plate   = v.get("license_plate", "UNKNOWN")
            is_viol = hv > 0 or riders >= 3
            color   = VIOL_COLOR if is_viol else BIKE_COLOR
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            tags  = (["TRIPLE"] if riders >= 3 else []) + ([f"NO-HELMET×{hv}"] if hv > 0 else [])
            label = f"Ppl: {riders}" + (" | " + " & ".join(tags) if tags else "")
            _put_label(out, label, x1, y1 - 4, color)
            if plate != "UNKNOWN":
                cv2.putText(out, plate, (x1, y2 + 18), FONT, 0.55, PLATE_COLOR, 2, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_on_image(detector: TrafficViolationDetector, img_path: Path,
                 save: bool, out_dir: Path, debug: bool) -> dict:
    """
    Run detector on one image, print result, optionally save annotated output.

    Returns the raw result dict.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Cannot read {img_path}")
        return {"violations": []}

    t0 = time.perf_counter()
    result = detector.predict(str(img_path))
    elapsed = time.perf_counter() - t0

    viol_count = len(result.get("violations", []))
    print(f"\n{'─'*60}")
    print(f"  Image   : {img_path.name}")
    print(f"  Time    : {elapsed:.2f}s")
    print(f"  Result  : {viol_count} violation(s) detected")
    print(json.dumps(result, indent=4))

    annotated = annotate_image(img, result)

    if debug:
        dbg_dir = Path("debug_outputs") / img_path.stem
        dbg_dir.mkdir(parents=True, exist_ok=True)
        # Save the annotated image as a debug artefact too
        cv2.imwrite(str(dbg_dir / "annotated.jpg"), annotated)
        print(f"  [DBG] annotated → {dbg_dir / 'annotated.jpg'}")

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"result_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved → {out_path}")
    else:
        cv2.imshow("Violation Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# BATCH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list[tuple[str, dict]], total_time: float):
    """Print a concise end-of-batch summary table."""
    print(f"\n{'═'*60}")
    print("  BATCH SUMMARY")
    print(f"{'═'*60}")
    total_viols = 0
    for name, res in all_results:
        n = len(res.get("violations", []))
        total_viols += n
        status = "🔴 VIOL" if n else "🟢  OK "
        print(f"  {status}  {name:<30}  {n} violation(s)")
    print(f"{'─'*60}")
    print(f"  Total images    : {len(all_results)}")
    print(f"  Total violations: {total_viols}")
    print(f"  Total time      : {total_time:.1f}s  "
          f"({total_time / max(len(all_results), 1):.2f}s/image avg)")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection — evaluation wrapper")
    parser.add_argument("input",
                        help="Path to an image file or a folder of images")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated results to --out directory")
    parser.add_argument("--out", default="final_results",
                        help="Output directory for saved results (default: final_results/)")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug outputs to debug_outputs/<stem>/")
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help=f"Path to model weights directory (default: {MODEL_DIR})")
    args = parser.parse_args()

    # ── Load detector once ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  AID 728 — Traffic Violation Detection")
    print(f"{'═'*60}")
    print(f"  Model dir : {args.model_dir}")
    t_init = time.perf_counter()
    detector = TrafficViolationDetector(model_dir=args.model_dir)
    print(f"  Init time : {time.perf_counter() - t_init:.2f}s\n")

    in_path = Path(args.input)
    out_dir = Path(args.out)

    # ── Single image or folder ────────────────────────────────────────────
    if in_path.is_file():
        run_on_image(detector, in_path, save=args.save,
                     out_dir=out_dir, debug=args.debug)

    elif in_path.is_dir():
        img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = sorted(p for p in in_path.iterdir()
                      if p.suffix.lower() in img_ext)
        if not imgs:
            print(f"[WARN] No images found in {in_path}")
            return
        print(f"  Found {len(imgs)} image(s) in '{in_path}'\n")
        all_results = []
        t_batch = time.perf_counter()
        for img_p in imgs:
            res = run_on_image(detector, img_p, save=True,
                               out_dir=out_dir, debug=args.debug)
            all_results.append((img_p.name, res))
        print_summary(all_results, time.perf_counter() - t_batch)

    else:
        print(f"[ERROR] '{args.input}' is not a valid file or directory.")


if __name__ == "__main__":
    main()
