#!/usr/bin/env python3
"""Open-vocabulary object detection on still frames with OWLv2.

Turns a natural-language command into image detections:

    ./detect_owlv2.py --command "pick up the tape" --images stills/scene_000_color.png

Or query several phrases at once to compare how the model scores them:

    ./detect_owlv2.py -q "blue painters tape" -q "tape dispenser" -q "multitool" \\
                      --images "stills/*_color.png"

Add --depth to also report the 3D point of each detection in the camera frame,
using the sibling *_depth.png and *_meta.json written by capture_stills.py.

Nothing here is arm-specific and nothing moves: it reads images and prints boxes.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"

# Leading command words to strip when turning an instruction into an object
# phrase. This is deliberately dumb: a first pass to prove the pipeline. Swap in
# an LLM (qwen2.5:14b is already pulled in ollama) when commands get messier.
_COMMAND_PREFIXES = [
    r"pick\s+up", r"pick", r"grab", r"grasp", r"get", r"fetch", r"take",
    r"find", r"locate", r"point\s+at", r"show\s+me", r"detect", r"where\s+is",
    r"hand\s+me", r"give\s+me", r"bring\s+me",
]
_ARTICLES = r"^(the|a|an|some|that|this)\s+"


def command_to_phrase(command):
    """'pick up the tape' -> 'tape'. Returns the object phrase."""
    s = command.strip().lower().rstrip(".!?")
    for pat in _COMMAND_PREFIXES:
        s = re.sub(r"^" + pat + r"\s+", "", s)
    s = re.sub(_ARTICLES, "", s)
    return s.strip() or command.strip()


def load_model(model_id, device, fp16):
    t0 = time.time()
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    if fp16 and device.type == "cuda":
        model = model.half()
    model = model.to(device).eval()
    return processor, model, time.time() - t0


def nms_per_query(dets, iou_thresh):
    """Greedy NMS within each query. OWLv2's post-processing does none, so a
    single object routinely comes back as a stack of near-identical boxes."""
    if not dets:
        return dets
    kept = []
    for q in {d["query"] for d in dets}:
        group = [d for d in dets if d["query"] == q]
        boxes = torch.tensor([d["box"] for d in group], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in group], dtype=torch.float32)
        idx = torchvision.ops.nms(boxes, scores, iou_thresh)
        kept.extend(group[i] for i in idx.tolist())
    kept.sort(key=lambda d: -d["score"])
    return kept


def detect(processor, model, device, fp16, pil_image, queries, threshold,
           nms_iou=0.3, topk=0):
    """Runs OWLv2. Returns (detections, elapsed_seconds).

    OWLv2 pads the input to a square before resizing, and the boxes it predicts
    are normalised against that PADDED square, not the original frame. Since the
    padding is added to the right and bottom, passing the square's side length
    as target_sizes puts the boxes straight back into original pixel
    coordinates. Passing the true (H, W) instead is the classic OWLv2 bug and
    yields boxes that are systematically squashed on one axis.
    """
    inputs = processor(text=[queries], images=pil_image, return_tensors="pt")
    inputs = {k: (v.to(device).half() if (fp16 and device.type == "cuda" and v.dtype == torch.float32)
                  else v.to(device)) for k, v in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    side = max(pil_image.height, pil_image.width)
    target_sizes = torch.tensor([[side, side]], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=threshold, target_sizes=target_sizes)[0]

    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        # Clip to the real image; padding lives outside it.
        x0 = max(0.0, min(x0, pil_image.width - 1))
        x1 = max(0.0, min(x1, pil_image.width - 1))
        y0 = max(0.0, min(y0, pil_image.height - 1))
        y1 = max(0.0, min(y1, pil_image.height - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        dets.append({
            "query": queries[int(label)],
            "score": float(score),
            "box": [x0, y0, x1, y1],
        })
    dets.sort(key=lambda d: -d["score"])
    if nms_iou > 0:
        dets = nms_per_query(dets, nms_iou)
    if topk > 0:
        dets = dets[:topk]
    return dets, elapsed


def deproject(det, depth_mm, meta):
    """Box -> 3D point in the camera frame, in metres.

    Uses the median of valid depth pixels in the middle of the box rather than
    the single centre pixel: the centre can easily land on a hole (the D455
    returns 0 where it has no stereo match) or on a background pixel seen
    through a gap in the object.
    """
    x0, y0, x1, y1 = det["box"]
    # Sample the central half of the box to stay off the edges/background.
    cx0 = int(x0 + 0.25 * (x1 - x0)); cx1 = int(x0 + 0.75 * (x1 - x0))
    cy0 = int(y0 + 0.25 * (y1 - y0)); cy1 = int(y0 + 0.75 * (y1 - y0))
    patch = depth_mm[max(0, cy0):max(1, cy1), max(0, cx0):max(1, cx1)]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    z = float(np.median(valid)) / 1000.0  # mm -> m
    u = (x0 + x1) / 2.0
    v = (y0 + y1) / 2.0
    x = (u - meta["ppx"]) / meta["fx"] * z
    y = (v - meta["ppy"]) / meta["fy"] * z
    return {"xyz_m": [x, y, z], "valid_frac": float(valid.size) / max(1, patch.size)}


# Distinct colours per query index (BGR).
_PALETTE = [(80, 220, 120), (255, 170, 60), (80, 130, 255),
            (240, 120, 240), (60, 230, 240), (200, 200, 200)]


def screen_size(default=(1920, 1080)):
    """Ask X how big the display is, so a shown window can be sized to fit."""
    try:
        import subprocess
        out = subprocess.check_output(["xdpyinfo"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if "dimensions:" in line:
                wh = line.split()[1]
                w, h = wh.split("x")
                return int(w), int(h)
    except Exception:
        pass
    return default


def _show_external(path):
    """Fall back to a desktop image viewer.

    The pip-installed cv2 on this box is a headless build (GUI: NONE), so
    cv2.imshow raises rather than opening a window - only the system C++ OpenCV
    has GTK. Handing the file to a real viewer is more robust than trying to
    swap in a GUI-enabled cv2, which would risk the working torch/cv2 install.
    """
    import shutil
    import subprocess
    for viewer in ("eog", "xdg-open", "display", "shotwell"):
        if shutil.which(viewer):
            print("Opening %s with %s on %s"
                  % (path, viewer, os.environ.get("DISPLAY", "?")))
            subprocess.Popen([viewer, path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    print("No image viewer found; annotated image is at %s" % path, file=sys.stderr)
    return False


def show_on_monitor(img, title, frac=0.9, path=None):
    """Display an image on the monitor, preferring a cv2 window and falling back
    to a desktop viewer when cv2 has no GUI support."""
    if "GUI:                           NONE" in cv2.getBuildInformation():
        return _show_external(path)

    sw, sh = screen_size()
    scale = min(frac * sw / img.shape[1], frac * sh / img.shape[0])
    win_w, win_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    except cv2.error:
        return _show_external(path)
    cv2.resizeWindow(title, win_w, win_h)
    cv2.moveWindow(title, (sw - win_w) // 2, (sh - win_h) // 2)
    print("Displaying %dx%d window on %s (q or ESC to close)"
          % (win_w, win_h, os.environ.get("DISPLAY", "?")))
    seen_visible = False
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        # The GTK backend reports not-visible for the first frame or two.
        visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
        if visible:
            seen_visible = True
        elif seen_visible:
            break
    cv2.destroyAllWindows()


def annotate(bgr, dets, queries, banner=None):
    out = bgr.copy()
    if banner:
        # Header strip so the query and outcome are readable in the image itself.
        cv2.rectangle(out, (0, 0), (out.shape[1], 42), (30, 30, 30), cv2.FILLED)
        cv2.putText(out, banner, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (240, 240, 240), 2, cv2.LINE_AA)
    if not dets:
        msg = "NO DETECTION above threshold"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cx = (out.shape[1] - tw) // 2
        cy = out.shape[0] // 2
        cv2.rectangle(out, (cx - 16, cy - th - 16), (cx + tw + 16, cy + 16),
                      (30, 30, 30), cv2.FILLED)
        cv2.putText(out, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (90, 90, 255), 3, cv2.LINE_AA)
    for d in dets:
        x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
        color = _PALETTE[queries.index(d["query"]) % len(_PALETTE)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        label = "%s %.2f" % (d["query"], d["score"])
        if d.get("xyz_m"):
            label += "  z=%.2fm" % d["xyz_m"][2]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(0, y0 - 8)
        cv2.rectangle(out, (x0, ty - th - 6), (x0 + tw + 8, ty + 4), color, cv2.FILLED)
        cv2.putText(out, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (20, 20, 20), 2, cv2.LINE_AA)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="stills/*_color.png",
                    help="image path or glob (default: stills/*_color.png)")
    ap.add_argument("-q", "--query", action="append", default=[],
                    help="object phrase to look for; repeatable")
    ap.add_argument("--command", default=None,
                    help="natural-language command, e.g. 'pick up the tape'")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="score threshold (default 0.15)")
    ap.add_argument("--nms-iou", type=float, default=0.3,
                    help="IoU for per-query NMS, 0 disables (default 0.3)")
    ap.add_argument("--topk", type=int, default=0,
                    help="keep only the N highest-scoring detections (0 = all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--fp16", action="store_true", help="half precision on GPU")
    ap.add_argument("--depth", action="store_true",
                    help="also deproject to 3D using sibling _depth.png/_meta.json")
    ap.add_argument("--out", default="detections", help="output directory")
    ap.add_argument("--show", action="store_true",
                    help="display the annotated image on the monitor (uses $DISPLAY)")
    ap.add_argument("--bench", type=int, default=0,
                    help="repeat inference N times on the first image to time it")
    args = ap.parse_args()

    queries = list(args.query)
    if args.command:
        phrase = command_to_phrase(args.command)
        print('Command: "%s"  ->  object phrase: "%s"' % (args.command, phrase))
        queries.append(phrase)
    if not queries:
        print("Give at least one --query or a --command.", file=sys.stderr)
        return 2

    paths = sorted(glob.glob(args.images)) if any(c in args.images for c in "*?[") \
        else [args.images]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("No images matched %r" % args.images, file=sys.stderr)
        return 2

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device: %s%s" % (device.type, " (fp16)" if args.fp16 else ""))
    print("Model : %s" % args.model)
    processor, model, load_s = load_model(args.model, device, args.fp16)
    print("Loaded in %.1f s\n" % load_s)

    os.makedirs(args.out, exist_ok=True)
    timings = []

    for i, path in enumerate(paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skip unreadable %s" % path)
            continue
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        dets, elapsed = detect(processor, model, device, args.fp16, pil, queries,
                               args.threshold, args.nms_iou, args.topk)
        timings.append(elapsed)

        depth_mm, meta = None, None
        if args.depth:
            dpath = path.replace("_color.png", "_depth.png")
            mpath = path.replace("_color.png", "_meta.json")
            if os.path.exists(dpath) and os.path.exists(mpath):
                depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
                with open(mpath) as f:
                    meta = json.load(f)
            else:
                print("  (no depth/meta sidecar for %s)" % os.path.basename(path))

        print("%s   %d detection(s)  [%.0f ms]"
              % (os.path.basename(path), len(dets), elapsed * 1000))
        for d in dets:
            if depth_mm is not None and meta is not None:
                p3 = deproject(d, depth_mm, meta)
                if p3:
                    d.update(p3)
            x0, y0, x1, y1 = d["box"]
            line = "   %-28s %.3f   box=(%4d,%4d)-(%4d,%4d)" % (
                d["query"], d["score"], x0, y0, x1, y1)
            if d.get("xyz_m"):
                x, y, z = d["xyz_m"]
                line += "   xyz=(%+.3f,%+.3f,%+.3f) m" % (x, y, z)
            print(line)

        banner = 'query: "%s"   |   %d hit(s) >= %.2f   |   %s' % (
            ", ".join(queries), len(dets), args.threshold, os.path.basename(path))
        vis = annotate(bgr, dets, queries, banner=banner)
        outp = os.path.join(args.out, os.path.basename(path).replace("_color", "_det"))
        cv2.imwrite(outp, vis)
        if args.show:
            show_on_monitor(vis, "OWLv2  |  %s" % ", ".join(queries),
                            path=os.path.abspath(outp))

        if args.bench and i == 0:
            print("\n  Benchmarking %d runs on this image..." % args.bench)
            runs = []
            for _ in range(args.bench):
                _, e = detect(processor, model, device, args.fp16, pil, queries,
                              args.threshold, args.nms_iou, args.topk)
                runs.append(e)
            runs = np.array(runs)
            print("  first (warm-up) %.0f ms | steady mean %.0f ms  median %.0f ms"
                  "  min %.0f ms  -> %.2f fps"
                  % (timings[0] * 1000, runs.mean() * 1000, np.median(runs) * 1000,
                     runs.min() * 1000, 1.0 / runs.mean()))

    print("\nAnnotated images written to %s/" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
