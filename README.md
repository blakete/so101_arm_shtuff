# so101_arm_shtuff

Scratch workspace for the SO-101 arm + D455 setup on `exopack` (Jetson, L4T 35.4.1).

## d455_viewer

C++ viewer that streams RGB + depth from the Intel RealSense D455 and shows
them side by side in one window, with a live SO-101 joint-position panel
alongside.

- Color at 1280x720 @ 30 fps, depth at 848x480 @ 30 fps. Falls back to
  librealsense defaults if that profile combination is unavailable.
- Depth is aligned to the color frame by default and colorized by librealsense.
- The window auto-sizes to 92% of the detected screen and centers itself.

### Layout

Default is a 2x2 grid, cycled with `l`:

```
kGrid (default)              kRow                      kStack
+----------+----------+      +-------+-------+-----+   +-------+-----+
| follower |  RGB     |      |  RGB  | depth |arms |   |  RGB  |arms |
+----------+----------+      +-------+-------+-----+   +-------+     |
| leader   |  depth   |                                | depth |     |
+----------+----------+                                +-------+-----+
```

In `kGrid` all four cells are the camera's 1280x720, so the canvas is 2560x1440
- 16:9, which matches exopack's screen almost exactly. Depth is resampled to the
color cell size in this layout so the cells stay square with each other even
when alignment is toggled off.
- Overlay shows live fps, alignment mode, colormap, and the depth in meters at
  the center crosshair.

## Joint panel

`sts_bus.h` speaks the Feetech STS/SCS serial protocol to each arm's six STS3215
servos at 1 Mbaud. One 8-byte read per servo from address 56 returns position,
speed, load, voltage and temperature in a single transaction.

### Multiple arms

Every CH34x driver board under `/dev/serial/by-id/*1a86*` is treated as an arm
and gets its own poll thread and panel, stacked in the right-hand column. The
boards on exopack:

| CH343 serial | role |
| --- | --- |
| `5AE6082981` | follower |
| `5AE6085251` | leader / teleop |

That mapping lives in `label_for()` in `arm_monitor.h`; an unrecognised board is
labelled with its serial. `--port <dev>` restricts the viewer to one arm.

Device paths are always resolved through `/dev/serial/by-id/`, never `ttyACMn`.
The by-id name derives from the CH343's serial number, so it follows a given
board across replugs, cable swaps, and USB hubs — the `ttyACMn` index does not.

`arm_monitor.h` runs each arm's polling on its own thread (~45 Hz measured per
arm, with two arms attached) and hands the render loop a mutex-guarded snapshot. This matters: each USB-CDC round trip
costs a millisecond or two, so polling six servos inline in the camera loop
would cost roughly a third of the framerate. If the port cannot be opened the
thread retries once a second, so plugging the arm in mid-run picks it up, and an
absent arm just shows "no response" rows without touching the video. After ~10
consecutive failed polls it closes the port and re-resolves the path, so an
unplug/replug recovers on its own — a yanked USB device otherwise leaves a stale
fd that fails forever.

**Read-only.** The bus client implements PING and READ only — there is no code
path that writes a servo register, so it cannot change torque, limits, IDs, or
commanded position.

### Two caveats on what the numbers mean

- **Angles are uncalibrated.** Degrees are computed as offset from the servo's
  mid-travel point (2048 of 4096 ticks) in the servo's own frame. No homing
  offsets or per-joint direction flips are applied, so this is "where each
  servo is", not "where a LeRobot arm model thinks the joint is".
- **Joint names are assumed** from the standard SO-101 convention for IDs 1-6.
  Six servos at IDs 1-6 is confirmed; which physical joint each ID drives is
  not independently verified here.

### Bus ownership

Only one process should hold the servo port at a time. Running `--dump-joints`
while the viewer is up (or alongside a LeRobot session) means two readers
interleaving on the same half-duplex bus. Stop one before starting the other.

### Why depth is 848x480 and not 1280x720

848x480 is the D455's native stereo resolution. Requesting 1280x720 depth
visibly thins out returns on low-texture surfaces (a plain white table came
back mostly empty, with no return at all at the center pixel). Alignment
resamples depth onto the 1280x720 color grid regardless, so requesting the
sensor's native profile costs nothing and keeps the depth image dense.

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Dependencies (already present on `exopack`): librealsense2 2.56.3 under
`/usr/local` and system OpenCV 4.5.4 (`libopencv-dev`, GTK2 highgui backend).
Both are found through pkg-config; `/usr/local/lib` is baked into the binary's
RPATH so no `LD_LIBRARY_PATH` is needed.

### Run

From the desktop session:

```bash
./build/d455_viewer
```

Over SSH, the window has to be pointed at the physical X session (`:1` on
seat0):

```bash
./run_on_desktop.sh          # equivalent to DISPLAY=:1 ./build/d455_viewer
```

### Options

| flag | effect |
| --- | --- |
| `--scale 0.1..1.0` | fraction of the screen the window fills (default `0.92`) |
| `--fullscreen` | start in true fullscreen |
| `--port <dev>` | servo bus device (default `/dev/ttyACM0`) |
| `--no-arm` | skip the servo bus entirely; camera only |
| `--dump-joints` | print joint readings to stdout for 5 s and exit (no camera, no window) |
| `--headless` | no window; grab 30 frames and write `/tmp/d455_headless.png` |

### Keys

| key | action |
| --- | --- |
| `q` / `Esc` | quit |
| `a` | toggle depth→color alignment |
| `c` | cycle depth colormap (Jet / WhiteToBlack / BlackToWhite / Hue) |
| `l` | cycle layout (grid → row → stacked) |
| `j` | toggle the joint panel |
| `f` | toggle fullscreen |
| `s` | save `d455_snapshot_N.png` of the current view |

### Performance note

The camera always delivers 30 fps; the displayed rate is limited by how many
pixels the Jetson has to scale up each frame. Measured on exopack's 6144x3456
display with the 2x2 grid (2560x1440 canvas):

| `--scale` | window | displayed |
| --- | --- | --- |
| 0.92 (default) | 5652x3179 (18 MP) | 12.5 fps |
| 0.6 | 3686x2073 (7.6 MP) | 18.5 fps |

The cost is the window upscale, not the camera or the servo polling — halving
the displayed pixel count buys back roughly half the lost framerate. Use
`--scale` to pick your point on that curve.

### Headless check

`./build/d455_viewer --headless` grabs 30 frames, reports the achieved fps, and
writes `/tmp/d455_headless.png` — useful for confirming the camera works over
SSH without a display.

## Hardware notes

- D455: serial `241122301570`, FW `5.16.0.1`, negotiated USB 3.2.
- SO-101 arms: two CH343 USB-serial bridges (`1a86:55d3`), serials `5AE6082981`
  (follower) and `5AE6085251` (leader/teleop). Both answer PING on six servos at
  IDs 1–6, 1 Mbaud.
- All twelve servos report **5.4 V** supply (STS3215 nominal is 7.4 V) and a flat
  0.0% load, consistent with the buses running on logic power only. Encoders read
  fine; the arms will not hold position or move like this. Deferred, not fixed.
- A dead USB cable will enumerate the CH343 fine while passing no servo traffic:
  the leader showed a healthy `/dev/serial/by-id` entry but zero PING replies
  across every baud from 9600 to 1 M until the cable was swapped. If a board
  appears but no servo answers, suspect the cable before the servos.
