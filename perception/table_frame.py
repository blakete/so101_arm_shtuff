#!/usr/bin/env python3
"""Estimate the table plane and derive the camera->table transform.

The D455 looks down at the desk at an angle, so everything logged in the camera
frame renders tilted: the table is a slope, "up" is meaningless, and an
axis-aligned box around a flat object picks up the tilt as fake extent. This
module recovers the support plane and builds a right-handed table frame

    origin : foot of the perpendicular from the camera onto the table
    +Z     : table normal, pointing up out of the surface toward the camera
    +Y     : camera's optical axis projected onto the table (away from camera)
    +X     : Y x Z, completing a right-handed frame

Applying the inverse of the camera's rotation puts the table flat in the
viewer, so Rerun's up axis is the table's up axis.

    ./table_frame.py --stem stills/pc_000 --show
    ./table_frame.py --stem stills/pc_000 --json table_frame.json

Depth only - no model, no torch, so this is fast and importable from anywhere.
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
    """The rerun-sdk wheel ships the viewer binary but never puts it on PATH."""
    cli_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(rr.__file__))), "rerun_cli")
    binary = os.path.join(cli_dir, "rerun")
    if os.path.isfile(binary) and os.access(binary, os.X_OK):
        os.environ["PATH"] = cli_dir + os.pathsep + os.environ.get("PATH", "")
        return binary
    return None


def deproject_grid(depth_mm, meta, stride=1):
    """Deproject a depth image to a HxWx3 point grid. Returns (points, valid)."""
    h, w = depth_mm.shape[:2]
    us, vs = np.meshgrid(np.arange(0, w, stride, dtype=np.float32),
                         np.arange(0, h, stride, dtype=np.float32))
    z = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    valid = z > 0
    x = (us - meta["ppx"]) / meta["fx"] * z
    y = (vs - meta["ppy"]) / meta["fy"] * z
    return np.stack([x, y, z], axis=-1), valid


def fit_plane_ransac(pts, iters=400, thresh=0.006, seed=0, max_score_pts=40000):
    """Dominant plane by RANSAC. Returns ((normal, d), inlier_mask) over `pts`.

    Scoring is done on a random subset when the cloud is large; the winning
    plane is then re-scored against every point.
    """
    n = len(pts)
    if n < 50:
        return None, None
    rng = np.random.default_rng(seed)
    score_pts = pts if n <= max_score_pts else pts[rng.choice(n, max_score_pts,
                                                              replace=False)]
    best, best_count = None, -1
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(n, 3, replace=False)]
        nv = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nv)
        if norm < 1e-9:
            continue
        nv = nv / norm
        d = -float(nv.dot(p0))
        count = int((np.abs(score_pts @ nv + d) < thresh).sum())
        if count > best_count:
            best, best_count = (nv, d), count
    if best is None:
        return None, None
    nv, d = best
    return best, np.abs(pts @ nv + d) < thresh


def refine_plane(pts):
    """Least-squares plane through points (SVD). RANSAC picks the inliers with a
    3-point sample, which is noisy; refitting to all of them sharpens the normal
    considerably, and the normal is what the whole rotation hangs on."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    nv = vt[-1]
    nv = nv / np.linalg.norm(nv)
    return nv, -float(nv.dot(c))


def estimate_table_frame(depth_mm, meta, zmax=1.5, thresh=0.006, seed=0):
    """Fit the support plane and build the camera->table transform.

    Returns a dict with the plane, the rotation/translation, and human-readable
    geometry (camera height, tilt, roll). Raises RuntimeError if no plane fits.
    """
    pts_grid, valid = deproject_grid(depth_mm, meta, stride=2)
    sel = valid & (pts_grid[..., 2] < zmax) & (pts_grid[..., 2] > 0)
    pts = pts_grid[sel]
    if len(pts) < 500:
        raise RuntimeError("only %d valid points within %.1f m" % (len(pts), zmax))

    plane, inliers = fit_plane_ransac(pts, thresh=thresh, seed=seed)
    if plane is None:
        raise RuntimeError("RANSAC found no plane")
    nv, d = refine_plane(pts[inliers])

    # Orient the normal to point from the table toward the camera (the camera
    # sits at the origin, so it must be on the positive side of the plane).
    if d < 0:
        nv, d = -nv, -d

    inlier_frac = float(inliers.mean())
    height = float(d)                 # camera's perpendicular distance to table
    origin = -d * nv                  # foot of that perpendicular, on the plane

    # Build the in-plane axes from the camera's optical axis so the frame is
    # deterministic rather than arbitrary.
    z_t = nv
    optical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = optical - optical.dot(z_t) * z_t
    if np.linalg.norm(fwd) < 1e-6:
        # Camera looking straight down the normal: any in-plane axis will do.
        fwd = np.array([1.0, 0.0, 0.0]) - z_t[0] * z_t
    y_t = fwd / np.linalg.norm(fwd)
    x_t = np.cross(y_t, z_t)
    x_t /= np.linalg.norm(x_t)

    # Columns are the table axes expressed in camera coordinates, so R_ct maps
    # table -> camera; its transpose maps camera -> table.
    R_ct = np.stack([x_t, y_t, z_t], axis=1)
    R_tc = R_ct.T
    t_tc = -R_tc @ origin

    tilt = 90.0 - float(np.degrees(np.arccos(np.clip(abs(optical.dot(z_t)), -1, 1))))
    cam_x = np.array([1.0, 0.0, 0.0])
    roll = 90.0 - float(np.degrees(np.arccos(np.clip(abs(cam_x.dot(z_t)), -1, 1))))

    return {
        "plane_normal_cam": nv.tolist(),
        "plane_d": float(d),
        "inlier_fraction": inlier_frac,
        "inlier_count": int(inliers.sum()),
        "points_considered": int(len(pts)),
        "camera_height_m": height,
        "optical_axis_tilt_deg": tilt,
        "camera_roll_deg": roll,
        "table_origin_in_cam": origin.tolist(),
        "R_cam_to_table": R_tc.tolist(),
        "t_cam_to_table": t_tc.tolist(),
        "R_table_to_cam": R_ct.tolist(),
        "convention": "p_table = R_cam_to_table @ p_cam + t_cam_to_table; "
                      "+Z is table up, +Y is optical axis projected onto table",
    }


def to_table(pts, frame):
    """Apply the camera->table transform to an Nx3 (or ...x3) array.

    Done explicitly in numpy rather than via rr.Transform3D so there is no
    ambiguity about Rerun's column-major Mat3x3 convention - a silently
    transposed rotation would look plausible and be wrong.
    """
    R = np.asarray(frame["R_cam_to_table"], dtype=np.float32)
    t = np.asarray(frame["t_cam_to_table"], dtype=np.float32)
    return pts.reshape(-1, 3) @ R.T + t


def table_grid(extent=0.6, step=0.1, z=0.0):
    """Line strips for a grid on the table plane, in table coordinates."""
    strips = []
    n = int(round(extent / step))
    for i in range(-n, n + 1):
        c = i * step
        strips.append([[-extent, c, z], [extent, c, z]])
        strips.append([[c, -extent, z], [c, extent, z]])
    return strips


def log_table_frame(frame, entity="world"):
    """Log the table axes and a reference grid, in table coordinates."""
    rr.log(entity + "/grid",
           rr.LineStrips3D(table_grid(), colors=[[90, 90, 110]], radii=0.0006))
    rr.log(entity + "/table_axes",
           rr.Arrows3D(origins=[[0, 0, 0]] * 3,
                       vectors=[[0.15, 0, 0], [0, 0.15, 0], [0, 0, 0.15]],
                       colors=[[230, 70, 70], [70, 230, 70], [70, 130, 255]],
                       labels=["table X", "table Y", "table Z (up)"]))
    # Where the camera sits, expressed in the table frame.
    cam_origin = to_table(np.zeros((1, 3), dtype=np.float32), frame)[0]
    R = np.asarray(frame["R_cam_to_table"], dtype=np.float32)
    rr.log(entity + "/camera_axes",
           rr.Arrows3D(origins=[cam_origin] * 3,
                       vectors=(R * 0.1).T,
                       colors=[[255, 140, 140], [140, 255, 140], [140, 180, 255]],
                       labels=["cam X", "cam Y", "cam Z (optical)"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True,
                    help="capture stem, e.g. stills/pc_000")
    ap.add_argument("--zmax", type=float, default=1.5,
                    help="ignore points beyond this depth when fitting (default 1.5)")
    ap.add_argument("--thresh", type=float, default=0.006,
                    help="RANSAC inlier distance, metres (default 0.006)")
    ap.add_argument("--json", default=None, help="write the transform here")
    ap.add_argument("--show", action="store_true",
                    help="open a table-aligned Rerun view of the cloud")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no-align", action="store_true",
                    help="with --show, log the raw camera frame for comparison")
    args = ap.parse_args()

    dpath, cpath, mpath = (args.stem + "_depth.png", args.stem + "_color.png",
                           args.stem + "_meta.json")
    for p in (dpath, cpath, mpath):
        if not os.path.exists(p):
            print("missing %s" % p, file=sys.stderr)
            return 2
    depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(cpath, cv2.IMREAD_COLOR)
    with open(mpath) as f:
        meta = json.load(f)

    try:
        frame = estimate_table_frame(depth_mm, meta, args.zmax, args.thresh)
    except RuntimeError as e:
        print("table frame estimation failed: %s" % e, file=sys.stderr)
        return 1

    nv = frame["plane_normal_cam"]
    print("plane normal (camera frame) : (%+.4f, %+.4f, %+.4f)" % tuple(nv))
    print("plane offset d              : %+.4f m" % frame["plane_d"])
    print("RANSAC inliers              : %d / %d  (%.1f%%)"
          % (frame["inlier_count"], frame["points_considered"],
             100 * frame["inlier_fraction"]))
    print("camera height above table   : %.3f m" % frame["camera_height_m"])
    print("optical axis vs table plane : %.1f deg" % frame["optical_axis_tilt_deg"])
    print("camera roll vs table        : %.1f deg" % frame["camera_roll_deg"])
    print("R_cam_to_table:")
    for row in frame["R_cam_to_table"]:
        print("   [%+.4f %+.4f %+.4f]" % tuple(row))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(frame, f, indent=2)
        print("wrote %s" % args.json)

    if not args.show:
        return 0

    rrd = os.path.abspath(os.path.join(
        "detections", os.path.basename(args.stem) + "_tableframe.rrd"))
    os.makedirs(os.path.dirname(rrd), exist_ok=True)
    rr.init("so101_table_frame")
    rr.save(rrd)

    pts, valid = deproject_grid(depth_mm, meta, stride=args.stride)
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[::args.stride, ::args.stride]
    sel = valid & (pts[..., 2] < args.zmax)
    cloud = pts[sel]
    colors = rgb[sel]

    if args.no_align:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        rr.log("world/cloud", rr.Points3D(cloud, colors=colors, radii=0.0015))
    else:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/cloud",
               rr.Points3D(to_table(cloud, frame), colors=colors, radii=0.0015))
        log_table_frame(frame)

    try:
        rr.flush(blocking=True)
    except Exception:
        time.sleep(3.0)
    viewer = ensure_rerun_viewer_on_path()
    if viewer:
        subprocess.Popen([viewer, rrd], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Rerun viewer opened on %s" % os.environ.get("DISPLAY", "?"))
    else:
        print("viewer not found; recording at %s" % rrd, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
