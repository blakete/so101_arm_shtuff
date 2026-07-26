# perception

Language-conditioned object grounding for the SO-101, step one: turn a phrase
like *"pick up the tape"* into image boxes and a 3D point in the camera frame.

Two scripts, both offline-first — no arm involvement, nothing moves.

| script | what it does |
| --- | --- |
| `capture_stills.py` | grabs RGB + aligned depth + intrinsics from the D455 to `stills/` |
| `detect_owlv2.py` | runs OWLv2 open-vocabulary detection on those stills |

## Environment

The stack is pinned by the Jetson, not by preference:

- **Python 3.8 is mandatory.** `torch 2.1.0a0+...nv23.06` is NVIDIA's JetPack 5
  wheel and only exists for 3.8. `python3.9` is on the box but has no CUDA torch.
- **`transformers==4.46.3`** — the last release supporting Python 3.8 (4.47
  raised the floor to 3.9). Installed to user site alongside everything else;
  `torch`, `cv2`, `numpy`, `PIL` all live in `~/.local/lib/python3.8/site-packages`,
  so a venv would *hide* them unless it inherits user site.
- Install was additive only: `regex`, `safetensors`, `tokenizers`, `transformers`.
  `numpy` was pinned to `1.23.5` during install so the resolver could not bump it
  out from under torch.
- `/` had ~9 GB free. `owlv2-base-patch16-ensemble` is ~600 MB in `~/.cache/huggingface`.

To reproduce:

```bash
python3 -m pip install --user "transformers==4.46.3" "numpy==1.23.5"
```

## Capture

Only one process can hold the D455, so **stop `d455_viewer` first**.

```bash
pkill -x d455_viewer
./capture_stills.py --tag tape -n 10 --interval 2.0
```

Each shot writes `_color.png` (8-bit BGR), `_depth.png` (16-bit, millimetres,
aligned to color) and `_meta.json` (intrinsics + depth scale). Because depth is
aligned, pixel (u,v) means the same thing in both images — that is what makes
the box→3D step a lookup rather than a registration problem. 40 frames are
discarded up front so auto-exposure settles; without that the first shots are
too dark to be useful.

## Detect

```bash
# natural-language command
./detect_owlv2.py --command "pick up the tape" --images stills/scene_000_color.png --depth

# explicit phrases, several at once
./detect_owlv2.py -q "blue roll of painters tape" -q "computer mouse" --images "stills/*_color.png"

# time it
./detect_owlv2.py --command "pick up the tape" --images stills/scene_000_color.png --bench 10
```

Useful flags: `--threshold` (default 0.15), `--nms-iou` (0.3), `--topk`,
`--fp16`, `--cpu`, `--depth`.

### Two implementation details that matter

**OWLv2 pads to a square before resizing**, and its boxes are normalised against
that padded square, not the original frame. Passing the true `(H, W)` as
`target_sizes` — the obvious thing, and what most snippets do — yields boxes
systematically squashed on one axis. Since padding is added right and bottom,
passing the square's side length puts boxes straight back into original pixel
coordinates.

**OWLv2 does no NMS.** Raw output gave 18 boxes for one query on one image,
mostly stacked duplicates of the same two objects. `nms_per_query()` applies
greedy NMS within each query and cut that to 3.

## Measured results

Scene: blue painter's-tape roll, Scotch tape dispenser, multitool, small PCB,
mouse, keyboard, arm gripper. Deliberately contains **two** tapes.

### Latency (AGX Orin, 1280x720 input, batch 1)

| precision | first call | steady mean | fps |
| --- | --- | --- | --- |
| fp32 | 2833 ms | **631 ms** | 1.58 |
| fp16 | 2733 ms | **587 ms** | 1.70 |

Model load ~2.8 s warm. **fp16 buys only 7%** — the eager-mode ViT at 960x960 is
not the shape that benefits from half precision here. If this needs to be
faster, TensorRT is the lever, not dtype. At ~0.6 s it is fine for
command-triggered grounding and far too slow for per-frame tracking.

### Accuracy — single query `"tape"`, all 6 stills

| rank | object | score range | box stability |
| --- | --- | --- | --- |
| 1 | blue painter's tape | 0.724 – 0.754 | ±1 px, xyz ±1 mm |
| 2 | Scotch tape dispenser | 0.518 – 0.560 | ±1 px |
| 3 | multitool (false positive) | 0.150 – 0.159 | — |

Both real tapes rank top-2 on every frame, with a clean gap to the false
positive. A threshold around 0.3 separates them reliably.

### The important negative result

Elaborate, specific phrases performed **worse** than the bare word `"tape"`:

| query | top hit | correct? |
| --- | --- | --- |
| `computer mouse` | 0.605 → mouse | yes |
| `blue roll of painters tape` | 0.522 → blue roll | yes |
| `clear tape dispenser` | 0.509 → **multitool** | no |
| `clear tape dispenser` | 0.361 → dispenser | yes, but 2nd |
| `multitool` | 0.237 → **small PCB** | no |
| `robot gripper` | nothing above threshold | no |

**OWLv2 scores are not calibrated across different query phrases.** Ranking
*within* one query is meaningful and stable; comparing scores *between* queries
is not. Practical consequence: use one phrase and rank its hits, rather than
racing several phrases and taking the global argmax. Disambiguating "which
tape" is better handled downstream (colour, size, position) than by hoping a
more descriptive prompt wins on score.

## Limitations

- **No hand-eye calibration.** The 3D points are in the *camera* frame. Useless
  for grasping until the camera→arm-base transform exists.
- **A box centroid is not a grasp pose.** No orientation. Needs at least a
  principal axis, i.e. segmentation (MobileSAM prompted by the box).
- **No abstention policy.** "Nothing found" and "three candidates" must become
  first-class outcomes before anything actuates.
- Depth valid fraction on this scene was ~54% of pixels; `deproject()` takes the
  median of valid pixels in the central half of the box to avoid holes, but a
  fully-invalid box returns `None` and must be handled.
