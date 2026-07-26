#!/usr/bin/env python3
"""Ground a text prompt in an RGB frame, isolate the object in 3D, and show the
whole thing in an orbitable Rerun viewer.

    ./visualize_target_3d.py -q "game controller" --stem stills/ctrl_000

Pipeline:
  1. OWLv2 grounds the prompt in the color image -> 2D box.
  2. The table plane is fitted (RANSAC) from an annulus *around* the box.
  3. Pixels inside the box that sit in front of that plane are the object
     ("segmented pixels" - depth-based, not a learned mask; see README).
  4. Those pixels are deprojected and an axis-aligned bounding box is taken over
     them - the "max" extent of the object in the camera frame.
  5. Everything is logged to Rerun: full colored cloud, highlighted object
     points, the wireframe box, and the annotated image.

The viewer is a separate window you can orbit with the mouse.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import rerun as rr


def ensure_rerun_viewer_on_path():
    """The rerun-sdk wheel ships the viewer binary but never puts it on PATH,
    so rr.init(spawn=True) fails with 'Failed to find Rerun Viewer executable'
    even though it is installed. Point PATH at the bundled copy."""
    cli_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(rr.__file__))),
                           "rerun_cli")
    binary = os.path.join(cli_dir, "rerun")
    if os.path.isfile(binary) and os.access(binary, os.X_OK):
        os.environ["PATH"] = cli_dir + os.pathsep + os.environ.get("PATH", "")
        return binary
    return None

# Reuse the detector from the sibling script rather than duplicating it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_owlv2 as det  # noqa: E402


def deproject_grid(depth_mm, meta, stride=1):
    """Full-frame deprojection. Returns (points Nx3, mask HxW bool, uv indices)."""
    h, w = depth_mm.shape[:2]
    us, vs = np.meshgrid(np.arange(0, w, stride, dtype=np.float32),
                         np.arange(0, h, stride, dtype=np.float32))
    z = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    valid = z > 0
    x = (us - meta["ppx"]) / meta["fx"] * z
    y = (vs - meta["ppy"]) / meta["fy"] * z
    pts = np.stack([x, y, z], axis=-1)
    return pts, valid


def fit_plane_ransac(pts, iters=300, thresh=0.006, seed=0):
    """Fit the dominant plane. Returns ((normal, d), inlier_mask) or (None, None).

    Used on the ring around the detection box, which on this desk is table
    surface, so the dominant plane there is the support surface.
    """
    n = len(pts)
    if n < 50:
        return None, None
    rng = np.random.default_rng(seed)
    best_plane, best_count, best_inl = None, -1, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nv)
        if norm < 1e-9:
            continue
        nv = nv / norm
        d = -float(nv.dot(p0))
        inl = np.abs(pts @ nv + d) < thresh
        c = int(inl.sum())
        if c > best_count:
            best_plane, best_count, best_inl = (nv, d), c, inl
    return best_plane, best_inl


def segment_in_box(depth_mm, meta, box, plane, margin=0.008):
    """Object mask: in-box pixels lying in front of the support plane.

    For each pixel we compute where the plane would be along that pixel's ray
    and keep the pixel only if the measured depth is closer than that by more
    than `margin`. Doing it per-ray (rather than thresholding raw depth) is what
    makes this work on a table that slants away from the camera.
    """
    h, w = depth_mm.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    mask = np.zeros((h, w), dtype=bool)
    sub = depth_mm[y0:y1, x0:x1].astype(np.float32) / 1000.0
    valid = sub > 0
    if not valid.any():
        return mask

    us, vs = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                         np.arange(y0, y1, dtype=np.float32))
    dirx = (us - meta["ppx"]) / meta["fx"]
    diry = (vs - meta["ppy"]) / meta["fy"]

    if plane is not None:
        nv, d = plane
        denom = nv[0] * dirx + nv[1] * diry + nv[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            plane_z = -d / denom
        good = valid & np.isfinite(plane_z) & (plane_z > 0)
        obj = good & (sub < plane_z - margin)
    else:
        # No plane: fall back to keeping the nearer half of the depth spread.
        zs = sub[valid]
        cut = np.percentile(zs, 60)
        obj = valid & (sub < cut)

    mask[y0:y1, x0:x1] = obj
    return mask


def reject_outliers(pts, k=2.5):
    """Drop points far from the median along each axis (stray depth speckle)."""
    if len(pts) < 20:
        return np.ones(len(pts), dtype=bool)
    med = np.median(pts, axis=0)
    dev = np.abs(pts - med)
    mad = np.median(dev, axis=0) + 1e-6
    return np.all(dev < k * 1.4826 * mad * 3.0, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True,
                    help="path stem, e.g. stills/ctrl_000 (expects _color.png etc.)")
    ap.add_argument("-q", "--query", required=True, help="object phrase to ground")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2,
                    help="cloud subsample for display (default 2)")
    ap.add_argument("--margin", type=float, default=0.008,
                    help="metres above the table a pixel must be to count (default 8mm)")
    ap.add_argument("--zmax", type=float, default=1.2,
                    help="clip the displayed cloud beyond this depth, metres (default 1.2)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-spawn", action="store_true",
                    help="do not open the viewer; just save a .rrd")
    ap.add_argument("--rrd", default=None, help="also save the recording here")
    args = ap.parse_args()

    cpath = args.stem + "_color.png"
    dpath = args.stem + "_depth.png"
    mpath = args.stem + "_meta.json"
    for p in (cpath, dpath, mpath):
        if not os.path.exists(p):
            print("missing %s" % p, file=sys.stderr)
            return 2

    color = cv2.imread(cpath, cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
    with open(mpath) as f:
        meta = json.load(f)

    # --- 1. ground the prompt -------------------------------------------------
    import torch
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model, load_s = det.load_model(det.DEFAULT_MODEL, device, args.fp16)
    pil = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    dets, infer_s = det.detect(processor, model, device, args.fp16, pil,
                               [args.query], args.threshold, nms_iou=0.3)
    print("model load %.1fs, inference %.0f ms, %d hit(s)"
          % (load_s, infer_s * 1000, len(dets)))
    for d in dets:
        print("   %-20s %.3f  box=%s" % (d["query"], d["score"],
                                         [int(v) for v in d["box"]]))
    if not dets:
        print("No detection above threshold %.2f - nothing to box." % args.threshold,
              file=sys.stderr)
        return 1
    best = dets[0]
    x0, y0, x1, y1 = best["box"]

    # --- 2. table plane from an annulus around the box ------------------------
    pts_full, valid_full = deproject_grid(depth_mm, meta, stride=1)
    h, w = depth_mm.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    rx0, ry0 = int(max(0, x0 - 0.6 * bw)), int(max(0, y0 - 0.6 * bh))
    rx1, ry1 = int(min(w, x1 + 0.6 * bw)), int(min(h, y1 + 0.6 * bh))
    ring = np.zeros((h, w), dtype=bool)
    ring[ry0:ry1, rx0:rx1] = True
    ring[int(y0):int(y1), int(x0):int(x1)] = False
    ring &= valid_full
    plane, inl = fit_plane_ransac(pts_full[ring])
    if plane is not None:
        nv, d = plane
        print("table plane: n=(%+.3f,%+.3f,%+.3f) d=%+.3f  (%d/%d ring inliers)"
              % (nv[0], nv[1], nv[2], d, int(inl.sum()), int(ring.sum())))
    else:
        print("plane fit failed; falling back to a depth percentile cut")

    # --- 3. segment + 4. axis-aligned bounding box ----------------------------
    obj_mask = segment_in_box(depth_mm, meta, best["box"], plane, args.margin)
    obj_pts = pts_full[obj_mask & valid_full]
    if len(obj_pts) < 30:
        print("Only %d object points survived segmentation - box would be "
              "meaningless. Try --margin smaller." % len(obj_pts), file=sys.stderr)
        return 1
    keep = reject_outliers(obj_pts)
    obj_pts = obj_pts[keep]

    lo = obj_pts.min(axis=0)
    hi = obj_pts.max(axis=0)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    print("object points: %d" % len(obj_pts))
    print("AABB centre  : (%+.3f, %+.3f, %+.3f) m" % tuple(center))
    print("AABB size    : %.3f x %.3f x %.3f m (WxHxD, camera frame)"
          % (2 * half[0], 2 * half[1], 2 * half[2]))

    # --- 5. log to rerun ------------------------------------------------------
    # Write the recording to a file, then open the viewer ON that file, rather
    # than using spawn=True. Two reasons: rr.save() after rr.init(spawn=True)
    # REPLACES the sink, so the data silently goes to the file and the spawned
    # viewer stays empty; and streaming over TCP races the viewer's startup.
    # File-then-open is deterministic and leaves a reusable artifact.
    rrd_path = os.path.abspath(
        args.rrd or os.path.join("detections", os.path.basename(args.stem) + ".rrd"))
    os.makedirs(os.path.dirname(rrd_path), exist_ok=True)
    rr.init("so101_grasp_target")
    rr.save(rrd_path)

    # Explicit layout: big 3D view beside the annotated image, so the viewer
    # does not have to guess and the 3D pane gets most of the window.
    try:
        import rerun.blueprint as rrb
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="point cloud + AABB", origin="/world"),
                rrb.Spatial2DView(name="detection", origin="/image"),
                column_shares=[2.0, 1.0],
            ),
            collapse_panels=True,
        ))
    except Exception as e:
        print("blueprint not applied (%s); using default layout" % e)

    # Camera optical frame: X right, Y down, Z forward.
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    cloud_pts, cloud_valid = deproject_grid(depth_mm, meta, stride=args.stride)
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[::args.stride, ::args.stride]
    # Clip to a workspace volume. Without this the cloud runs out to the far
    # wall and floor several metres away, which blows up the scene bounds and
    # leaves the viewer's default camera parked inside the desk.
    sel = cloud_valid & (cloud_pts[..., 2] < args.zmax)
    print("cloud: %d points within %.1f m (of %d valid)"
          % (int(sel.sum()), args.zmax, int(cloud_valid.sum())))
    rr.log("world/cloud", rr.Points3D(positions=cloud_pts[sel],
                                      colors=rgb[sel], radii=0.0015))
    rr.log("world/object", rr.Points3D(positions=obj_pts,
                                       colors=np.tile([255, 60, 60], (len(obj_pts), 1)),
                                       radii=0.0025))
    rr.log("world/bbox", rr.Boxes3D(centers=[center], half_sizes=[half],
                                    colors=[[60, 255, 120]],
                                    labels=["%s  %.2f" % (best["query"], best["score"])]))
    banner = 'query: "%s"  |  %.2f  |  AABB %.0fx%.0fx%.0f mm' % (
        best["query"], best["score"], 2000 * half[0], 2000 * half[1], 2000 * half[2])
    vis = det.annotate(color, [best], [args.query], banner=banner)
    overlay = vis.copy()
    overlay[obj_mask] = (0.35 * overlay[obj_mask] +
                         0.65 * np.array([60, 60, 255])).astype(np.uint8)
    rr.log("image/annotated", rr.Image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))

    out_png = os.path.join("detections", os.path.basename(args.stem) + "_seg3d.png")
    os.makedirs("detections", exist_ok=True)
    cv2.imwrite(out_png, overlay)
    print("wrote %s" % out_png)

    # Make sure every logged chunk has hit disk before the viewer opens it.
    try:
        rr.flush(blocking=True)
    except Exception:
        time.sleep(3.0)
    print("wrote %s (%.1f MB)" % (rrd_path, os.path.getsize(rrd_path) / 1e6))

    if not args.no_spawn:
        viewer = ensure_rerun_viewer_on_path()
        if not viewer:
            print("Rerun viewer binary not found; open %s manually." % rrd_path,
                  file=sys.stderr)
            return 1
        subprocess.Popen([viewer, rrd_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Rerun viewer opened on %s" % os.environ.get("DISPLAY", "?"))
        print("  orbit: left-drag   pan: middle-drag (or shift+left)   zoom: scroll")
    return 0


if __name__ == "__main__":
    sys.exit(main())
