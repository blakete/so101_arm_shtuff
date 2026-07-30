#!/usr/bin/env python3
"""rig_rerun.py - live Rerun visualization of the SO-101 rig on exopack.

Streams into a Rerun viewer:
  - D455 color + wrist camera as 2D panels
  - D455 depth as an RGB-colored 3D point cloud
  - the D455 pose (pinhole frustum) in a world frame anchored to the table:
    the dominant plane is RANSAC-fitted from the first depth frames, world +Z
    is the table normal (up), origin is where the optical axis meets the table
  - a live virtual SO-101: the so101_new_calib URDF (so101_model/) driven by
    the follower's servo positions, read straight off the STS bus (read-only
    port of sts_bus.h using stdlib termios - no pyserial). URDF zero pose =
    all servos centered at 2048 ticks, which is exactly lerobot's new
    calibration convention. Joint angles also plot in a time-series panel.

The D455 is static-mounted, so its pose is logged once after the plane fit;
wire in cuVSLAM/DLIO odometry here if the camera ever moves. The arm base is
placed on the table along the camera's look direction (facing back at it);
trim with --arm-dist/--arm-x/--arm-y/--arm-yaw until it overlaps the cloud.

  python3 rig_rerun.py [--serve] [--wrist-dev PATH] [--no-wrist] [--no-arm]
"""

import argparse
import glob as globmod
import json
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr

WRIST_DEV = ("/dev/v4l/by-id/"
             "usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0")

COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 848, 480     # D455 native stereo resolution
FPS = 30
CLOUD_STEP = 4                  # decimation of the aligned depth grid
CLOUD_EVERY = 2                 # log the cloud every Nth frame
DEPTH_MIN, DEPTH_MAX = 0.15, 4.0            # meters
PLANE_THRESH = 0.008            # RANSAC inlier distance, meters
# Workspace crop in the table frame: without it, returns from the office
# background blow up the 3D view's auto-bounds and the table becomes a speck.
CROP_XY, CROP_Z = 0.8, (-0.03, 0.6)         # meters around the world origin

# SO-101 follower servo bus (see arm_monitor.h: the CH34x by-id path is the
# only name that survives replugs; 5AE6082981 is the follower board).
FOLLOWER_GLOB = "/dev/serial/by-id/*1a86*5AE6082981*"
ANY_ARM_GLOB = "/dev/serial/by-id/*1a86*"
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "so101_model", "so101_new_calib.urdf")
# Servo IDs 1..6 from the base out, names matching the URDF joints.
JOINT_IDS = [("shoulder_pan", 1), ("shoulder_lift", 2), ("elbow_flex", 3),
             ("wrist_flex", 4), ("wrist_roll", 5), ("gripper", 6)]
TICKS_CENTER, RAD_PER_TICK = 2048, 2.0 * math.pi / 4096.0

# Per-joint zero/sign calibration. Captured via --calibrate (pose the real arm
# over the frozen virtual one, then `touch /tmp/so101_capture`); hand-edit
# "sign" to -1 for any joint that mirrors. Hot-reloaded on file change.
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "so101_calib.json")
CAPTURE_TRIGGER = "/tmp/so101_capture"


def load_calib(path):
    calib = {name: {"zero": TICKS_CENTER, "sign": 1.0} for name, _ in JOINT_IDS}
    try:
        with open(path) as f:
            for name, c in json.load(f).items():
                if name in calib:
                    calib[name].update(c)
        print("calibration loaded: %s" %
              {n: (c["zero"], c["sign"]) for n, c in calib.items()}, flush=True)
    except (IOError, ValueError):
        pass
    return calib
WARMUP_FRAMES = 15              # let auto-exposure settle before the fit


class WristCam(threading.Thread):
    """Latest-frame grabber so MJPG decode never stalls the D455 loop."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.lock = threading.Lock()
        self.frame = None       # most recent BGR frame
        self.stamp = 0.0
        self.running = True

    def run(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = None
                    time.sleep(1.0)
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()   # yanked camera: reads fail forever, re-open
                cap = None
                continue
            with self.lock:
                self.frame = frame      # cv2 allocates a fresh array per read
                self.stamp = time.time()

    def latest(self):
        with self.lock:
            return self.frame, self.stamp


class StsBus:
    """Read-only Feetech STS3215 client, a straight port of sts_bus.h.
    Only PING/READ are implemented, so it can never write a servo register."""

    def __init__(self):
        self.fd = -1

    def open(self, path, baud=termios.B1000000):
        self.close()
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as e:
            return "%s: %s" % (path, e.strerror)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[0] = 0                                        # iflag: raw
            attrs[1] = 0                                        # oflag: raw
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0                                        # lflag: raw
            attrs[4] = attrs[5] = baud
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except termios.error as e:
            os.close(fd)
            return str(e)
        self.fd = fd
        return None

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def read_block(self, sid, addr, length, timeout=0.02):
        if self.fd < 0:
            return None
        pkt = bytes([0xFF, 0xFF, sid, 0x04, 0x02, addr, length])
        pkt += bytes([~sum(pkt[2:]) & 0xFF])
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
            os.write(self.fd, pkt)
        except OSError:
            return None
        want = 6 + length
        buf = b""
        poller = select.poll()
        poller.register(self.fd, select.POLLIN)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not poller.poll(max(0, int((deadline - time.monotonic()) * 1e3))):
                break
            try:
                buf += os.read(self.fd, 64)
            except OSError:
                return None
            # Half-duplex bus: scan past a possible echo of our own packet.
            for i in range(len(buf) - want + 1):
                if (buf[i] == 0xFF and buf[i + 1] == 0xFF and buf[i + 2] == sid
                        and buf[i + 3] == length + 2):
                    frame = buf[i:i + want]
                    if (~sum(frame[2:want - 1]) & 0xFF) == frame[want - 1]:
                        return frame[5:5 + length]
        return None

    def read_pos(self, sid):
        b = self.read_block(sid, 56, 2)  # present position, 0..4095
        return None if b is None else (b[0] | (b[1] << 8))


class ArmPoller(threading.Thread):
    """Polls the six servos on a background thread; mirrors arm::Monitor's
    resilience (re-resolve the port, drop a dead fd after repeated misses)."""

    def __init__(self, port_glob):
        super().__init__(daemon=True)
        self.port_glob = port_glob
        self.lock = threading.Lock()
        self.ticks = {}         # joint name -> raw ticks 0..4095
        self.ok = False
        self.status = "starting"
        self.running = True

    def run(self):
        bus = StsBus()
        fails = 0
        while self.running:
            if bus.fd < 0:
                ports = (sorted(globmod.glob(self.port_glob))
                         or sorted(globmod.glob(ANY_ARM_GLOB)))
                err = bus.open(ports[0]) if ports else "no CH34x driver board"
                if bus.fd < 0:
                    with self.lock:
                        self.ok, self.status = False, err
                    time.sleep(1.0)
                    continue
                with self.lock:
                    self.status = ports[0]
                fails = 0
            got = {}
            for name, sid in JOINT_IDS:
                ticks = bus.read_pos(sid)
                if ticks is not None:
                    got[name] = ticks
            if got:
                fails = 0
                with self.lock:
                    self.ticks.update(got)
                    self.ok = True
            else:
                # A yanked USB device reads nothing forever; drop and re-open.
                fails += 1
                with self.lock:
                    self.ok = False
                if fails >= 10:
                    bus.close()
                    fails = 0
            time.sleep(0.01)
        bus.close()

    def snapshot(self):
        with self.lock:
            return dict(self.ticks), self.ok


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_mat(r, p, y):
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def rot_axis(axis, q):
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _origin(el):
    o = el.find("origin")
    xyz = [float(t) for t in (o.get("xyz", "0 0 0") if o is not None
                              else "0 0 0").split()]
    rpy = [float(t) for t in (o.get("rpy", "0 0 0") if o is not None
                              else "0 0 0").split()]
    return np.array(xyz), rpy


class So101Model:
    """Just enough URDF to drive Rerun: nested entities mirror the kinematic
    chain, so each per-frame Transform3D is exactly the URDF joint transform."""

    def __init__(self, urdf_path, root_entity="world/arm"):
        self.dir = os.path.dirname(urdf_path)
        root = ET.parse(urdf_path).getroot()
        self.visuals = {}       # link name -> [(xyz, rpy, mesh path)]
        for link in root.findall("link"):
            vis = []
            for v in link.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is None:
                    continue
                xyz, rpy = _origin(v)
                vis.append((xyz, rpy,
                            os.path.join(self.dir, mesh.get("filename"))))
            self.visuals[link.get("name")] = vis

        by_parent = {}
        for j in root.findall("joint"):
            xyz, rpy = _origin(j)
            axis_el = j.find("axis")
            axis = np.array([float(t) for t in
                             (axis_el.get("xyz") if axis_el is not None
                              else "0 0 1").split()])
            jd = {"name": j.get("name"), "type": j.get("type"),
                  "xyz": xyz, "R": rpy_mat(*rpy), "axis": axis,
                  "parent": j.find("parent").get("link"),
                  "child": j.find("child").get("link")}
            by_parent.setdefault(jd["parent"], []).append(jd)

        self.path = {"base_link": root_entity + "/base_link"}
        self.chain = []
        stack = ["base_link"]
        while stack:
            link = stack.pop()
            for jd in by_parent.get(link, []):
                self.path[jd["child"]] = self.path[link] + "/" + jd["child"]
                jd["entity"] = self.path[jd["child"]]
                self.chain.append(jd)
                stack.append(jd["child"])

    def log_static(self):
        for link, vis in self.visuals.items():
            if link not in self.path:
                continue
            for i, (xyz, rpy, mesh) in enumerate(vis):
                ent = "%s/visual%d" % (self.path[link], i)
                rr.log(ent, rr.Transform3D(translation=xyz,
                                           mat3x3=rpy_mat(*rpy)), static=True)
                rr.log(ent, rr.Asset3D(path=mesh), static=True)
        for jd in self.chain:
            if jd["type"] == "fixed":
                rr.log(jd["entity"], rr.Transform3D(translation=jd["xyz"],
                                                    mat3x3=jd["R"]), static=True)
            else:
                # Amber arrow marks each articulation axis.
                rr.log(jd["entity"] + "/axis",
                       rr.Arrows3D(vectors=[jd["axis"] * 0.04],
                                   colors=[[255, 190, 0]]), static=True)

    def log_pose(self, angles):
        for jd in self.chain:
            if jd["type"] != "revolute":
                continue
            R = jd["R"] @ rot_axis(jd["axis"], angles.get(jd["name"], 0.0))
            rr.log(jd["entity"],
                   rr.Transform3D(translation=jd["xyz"], mat3x3=R))


def fit_plane_ransac(pts, iters=300, thresh=PLANE_THRESH, seed=0):
    """Dominant plane through pts (Nx3). Returns unit normal n and offset d
    with n.x = d for points on the plane, refined by SVD over the inliers."""
    rng = np.random.default_rng(seed)
    best_inl, best_count = None, -1
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inl = np.abs(pts @ n - n @ p0) < thresh
        count = int(inl.sum())
        if count > best_count:
            best_inl, best_count = inl, count
    q = pts[best_inl]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[2]
    return n, float(n @ c), best_inl


def world_from_camera(n, d):
    """Table-anchored world frame from a plane in camera coords: returns A
    (rows = world axes in camera coords) and c0 (world origin in camera
    coords), so p_world = A @ (p_cam - c0)."""
    if d > 0:                   # make the normal point from table to camera
        n, d = -n, -d
    # Origin: the optical axis hits the table; degenerate view -> closest point.
    c0 = (d / n[2]) * np.array([0.0, 0.0, 1.0]) if abs(n[2]) > 0.2 else d * n
    xw = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(xw) < 1e-6:
        xw = np.array([0.0, 1.0, 0.0]) - n[1] * n
    xw /= np.linalg.norm(xw)
    yw = np.cross(n, xw)
    return np.stack([xw, yw, n]), c0


def jpeg(img_rgb):
    try:
        return rr.Image(img_rgb).compress(jpeg_quality=80)
    except Exception:
        return rr.Image(img_rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="web viewer instead of a native window")
    ap.add_argument("--wrist-dev", default=WRIST_DEV)
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--no-arm", action="store_true",
                    help="skip the servo bus and the virtual arm")
    ap.add_argument("--calibrate", action="store_true",
                    help="freeze the virtual arm at URDF zero; pose the real "
                         "arm to match it, then `touch %s` to capture the "
                         "servo zero offsets" % CAPTURE_TRIGGER)
    ap.add_argument("--urdf", default=URDF_PATH)
    ap.add_argument("--arm-dist", type=float, default=0.40,
                    help="base distance from world origin along the view dir")
    ap.add_argument("--arm-x", type=float, default=None,
                    help="override base x in the table frame")
    ap.add_argument("--arm-y", type=float, default=None,
                    help="override base y in the table frame")
    ap.add_argument("--arm-yaw", type=float, default=None,
                    help="override base yaw in degrees (default: face camera)")
    args = ap.parse_args()

    rr.init("exopack_rig")
    if args.serve:
        rr.serve(open_browser=False)
    else:
        try:
            rr.spawn(memory_limit="4GB")
        except TypeError:
            rr.spawn()

    try:
        import rerun.blueprint as rrb
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="table frame"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/world/camera/image",
                                      name="D455 RGB"),
                    rrb.Spatial2DView(origin="/wrist/image", name="wrist cam"),
                    rrb.TimeSeriesView(origin="/joints",
                                       name="joint angles (deg)"),
                    row_shares=[3, 3, 2],
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        ))
    except Exception as e:
        print("blueprint skipped:", e)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Static wireframe spanning the whole workspace (the cloud crop volume).
    # This pins the 3D view's auto-fit bounds: without it the eye keeps
    # re-centering on the fluttering per-frame cloud extent and the entire
    # scene appears to jitter until the user first orbits the view.
    rr.log("world/workspace",
           rr.Boxes3D(centers=[[0.0, 0.0, (CROP_Z[0] + CROP_Z[1]) / 2.0]],
                      half_sizes=[[CROP_XY, CROP_XY,
                                   (CROP_Z[1] - CROP_Z[0]) / 2.0]],
                      colors=[[130, 130, 130, 70]]),
           static=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
    profile = pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color) \
                  .as_video_stream_profile().get_intrinsics()
    print("D455 up: fx=%.1f fy=%.1f  depth scale %g m" %
          (intr.fx, intr.fy, depth_scale), flush=True)

    rr.log("world/camera/image",
           rr.Pinhole(image_from_camera=[[intr.fx, 0, intr.ppx],
                                         [0, intr.fy, intr.ppy],
                                         [0, 0, 1]],
                      resolution=[COLOR_W, COLOR_H],
                      image_plane_distance=0.15),
           static=True)

    wrist = None
    if not args.no_wrist:
        wrist = WristCam(args.wrist_dev)
        wrist.start()

    arm, model = None, None
    calib = load_calib(CALIB_PATH)
    calib_mtime = os.path.getmtime(CALIB_PATH) if os.path.exists(CALIB_PATH) \
        else 0.0
    calibrating = args.calibrate
    if not args.no_arm:
        model = So101Model(args.urdf)
        model.log_static()
        arm = ArmPoller(FOLLOWER_GLOB)
        arm.start()
        print("Arm    : %s (%d joints, %d meshes)" %
              (args.urdf, sum(j["type"] == "revolute" for j in model.chain),
               sum(len(v) for v in model.visuals.values())), flush=True)
        if calibrating:
            print("CALIBRATION: pose the real arm to overlap the frozen "
                  "virtual arm (watch the cloud), then run:  touch %s"
                  % CAPTURE_TRIGGER, flush=True)

    # Precomputed back-projection rays for the decimated aligned-depth grid.
    vs, us = np.mgrid[0:COLOR_H:CLOUD_STEP, 0:COLOR_W:CLOUD_STEP]
    ray_x = ((us - intr.ppx) / intr.fx).astype(np.float32)
    ray_y = ((vs - intr.ppy) / intr.fy).astype(np.float32)

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    fitted = False
    last_wrist = 0.0
    t0 = time.time()
    n_frame = 0
    while not stop["now"]:
        try:
            frames = align.process(pipe.wait_for_frames())
        except RuntimeError as e:
            print("wait_for_frames:", e, flush=True)
            continue
        depth_f, color_f = frames.get_depth_frame(), frames.get_color_frame()
        if not depth_f or not color_f:
            continue
        depth = np.asanyarray(depth_f.get_data())
        color = np.asanyarray(color_f.get_data())
        n_frame += 1
        rr.set_time_seconds("t", time.time() - t0)

        z = depth[::CLOUD_STEP, ::CLOUD_STEP].astype(np.float32) * depth_scale
        valid = (z > DEPTH_MIN) & (z < DEPTH_MAX)
        pts = np.stack([ray_x * z, ray_y * z, z], axis=-1)[valid]

        if not fitted and n_frame >= WARMUP_FRAMES and len(pts) > 5000:
            sample = pts[np.random.default_rng(0).choice(
                len(pts), min(len(pts), 20000), replace=False)]
            n, d, _ = fit_plane_ransac(sample)
            A, c0 = world_from_camera(n, d)
            rr.log("world/camera",
                   rr.Transform3D(translation=-A @ c0, mat3x3=A), static=True)
            # Grey slab under the inlier footprint marks the fitted table.
            pw = (pts - c0) @ A.T
            on = np.abs(pw[:, 2]) < PLANE_THRESH
            (x0, y0), (x1, y1) = (np.percentile(pw[on, :2], p, axis=0)
                                  for p in (2, 98))
            quad = np.array([[x0, y0, 0], [x1, y0, 0],
                             [x1, y1, 0], [x0, y1, 0]], dtype=np.float32)
            rr.log("world/table",
                   rr.Mesh3D(vertex_positions=quad,
                             triangle_indices=[[0, 1, 2], [0, 2, 3]],
                             vertex_colors=np.tile([95, 90, 85], (4, 1))),
                   static=True)
            cam_h = float((-A @ c0)[2])
            print("table plane locked: normal %s, camera %.3f m above, "
                  "patch %.2fx%.2f m" % (np.round(n, 3), cam_h,
                                         x1 - x0, y1 - y0), flush=True)
            if model is not None:
                # Base on the table along the camera's horizontal look
                # direction, facing back toward the camera.
                p_cam = -A @ c0
                horiz = np.array([-p_cam[0], -p_cam[1], 0.0])
                horiz /= max(np.linalg.norm(horiz), 1e-6)
                # The arm is the only tall thing on the far side of the table,
                # so the far decile of above-table points centers on its base.
                bx = by = None
                tall = pw[(pw[:, 2] > 0.03) & (pw[:, 2] < 0.35)
                          & (np.abs(pw[:, 0]) < CROP_XY)
                          & (np.abs(pw[:, 1]) < CROP_XY)]
                if len(tall) > 300:
                    s = tall[:, :2] @ horiz[:2]
                    far = tall[(s > 0.15) & (s > np.percentile(s, 80))]
                    if len(far) > 100:
                        bx, by = (float(v) for v in np.median(far[:, :2],
                                                              axis=0))
                        print("arm base estimated from cloud", flush=True)
                if bx is None:
                    bx, by = (float(horiz[0]) * args.arm_dist,
                              float(horiz[1]) * args.arm_dist)
                if args.arm_x is not None:
                    bx = args.arm_x
                if args.arm_y is not None:
                    by = args.arm_y
                yaw = math.radians(args.arm_yaw) if args.arm_yaw is not None \
                    else math.atan2(-horiz[1], -horiz[0])
                rr.log("world/arm",
                       rr.Transform3D(translation=[bx, by, 0.0],
                                      mat3x3=rot_z(yaw)), static=True)
                print("arm base at (%.2f, %.2f), yaw %.0f deg" %
                      (bx, by, math.degrees(yaw)), flush=True)
            fitted = True

        rr.log("world/camera/image/rgb", jpeg(color))
        if fitted and n_frame % CLOUD_EVERY == 0:
            cols = color[::CLOUD_STEP, ::CLOUD_STEP][valid]
            pw = (pts - c0) @ A.T
            keep = ((np.abs(pw[:, 0]) < CROP_XY) & (np.abs(pw[:, 1]) < CROP_XY)
                    & (pw[:, 2] > CROP_Z[0]) & (pw[:, 2] < CROP_Z[1]))
            # Explicit radius: default points are 1 device pixel, invisible
            # on the 6K desktop.
            rr.log("world/points",
                   rr.Points3D(pw[keep], colors=cols[keep], radii=0.0025))

        if wrist is not None:
            frame, stamp = wrist.latest()
            if frame is not None and stamp > last_wrist:
                last_wrist = stamp
                rr.log("wrist/image",
                       jpeg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if arm is not None:
            ticks, ok = arm.snapshot()
            if calibrating:
                model.log_pose({})  # frozen at URDF zero as the match target
                if ok and os.path.exists(CAPTURE_TRIGGER):
                    for name, t in ticks.items():
                        calib[name]["zero"] = int(t)
                    with open(CALIB_PATH, "w") as f:
                        json.dump(calib, f, indent=2)
                    calib_mtime = os.path.getmtime(CALIB_PATH)
                    os.remove(CAPTURE_TRIGGER)
                    calibrating = False
                    print("calibration captured -> %s : %s" %
                          (CALIB_PATH,
                           {n: c["zero"] for n, c in calib.items()}),
                          flush=True)
            elif ok:
                # Hot-reload hand-edits (e.g. sign flips) without a restart.
                if n_frame % 30 == 0 and os.path.exists(CALIB_PATH):
                    m = os.path.getmtime(CALIB_PATH)
                    if m > calib_mtime:
                        calib_mtime = m
                        calib = load_calib(CALIB_PATH)
                angles = {
                    name: calib[name]["sign"] * (t - calib[name]["zero"])
                    * RAD_PER_TICK for name, t in ticks.items()}
                model.log_pose(angles)
                for name, q in angles.items():
                    rr.log("joints/" + name, rr.Scalar(math.degrees(q)))

        if n_frame % 300 == 0:
            print("frame %d  %.1f fps  %d pts" %
                  (n_frame, n_frame / (time.time() - t0), len(pts)),
                  flush=True)

    if wrist is not None:
        wrist.running = False
    if arm is not None:
        arm.running = False
    pipe.stop()
    print("stopped after %d frames" % n_frame, flush=True)


if __name__ == "__main__":
    sys.exit(main())
