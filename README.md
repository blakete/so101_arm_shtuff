# so101_arm_shtuff

Scratch workspace for the SO-101 arm + D455 setup on `exopack` (Jetson, L4T 35.4.1).

## d455_viewer

C++ viewer that streams RGB + depth from the Intel RealSense D455 and shows
them side by side in one window.

- Color and depth at 848x480 @ 30 fps (D455 native depth width — no FOV crop),
  falling back to librealsense defaults if that profile is unavailable.
- Depth is aligned to the color frame by default and colorized by librealsense.
- Overlay shows live fps, alignment mode, colormap, and the depth in meters at
  the center crosshair.

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

### Keys

| key | action |
| --- | --- |
| `q` / `Esc` | quit |
| `a` | toggle depth→color alignment |
| `c` | cycle depth colormap (Jet / WhiteToBlack / BlackToWhite / Hue) |
| `s` | save `d455_snapshot_N.png` of the current view |

### Headless check

`./build/d455_viewer --headless` grabs 30 frames, reports the achieved fps, and
writes `/tmp/d455_headless.png` — useful for confirming the camera works over
SSH without a display.

## Hardware notes

- D455: serial `241122301570`, FW `5.16.0.1`, negotiated USB 3.2.
- SO-101 arm: CH343 USB-serial bridge (`1a86:55d3`) on `/dev/ttyACM0`, stable
  path `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6082981-if00`; 6 Feetech
  servos answer PING at IDs 1–6 at 1 Mbaud.
