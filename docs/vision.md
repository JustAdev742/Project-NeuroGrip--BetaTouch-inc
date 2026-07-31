# Vision

## The abstraction comes first

The requirement was explicit: *do not tightly couple the software to one model.*

Every backend implements one `Protocol` — frame in, `VisionResult` out — and
declares what it can do:

```python
class VisionCapability(Flag):
    DETECTION | CLASSIFICATION | DEPTH | SEGMENTATION | GRASP | GESTURE | TRACKING
```

Consumers check capabilities, never types. A backend that cannot produce grasps
simply omits `GRASP`, and `neurogrip.ai.grasp` falls back to the
affordance-driven planner. Adding segmentation or gesture recognition later means
shipping a backend that sets the flag, not editing the pipeline.

Backends register themselves at import:

```python
register_backend("my_model", lambda config, **kw: MyBackend(config))
```

and are selected by `[vision] backend = "my_model"`. An unknown name falls back
to the null backend with a warning rather than aborting startup — an
unrecognised model string in a config file must not prevent someone from using
their hand.

---

## HGGD-MCU

The configured backend for this build.

**Background.** HGGD (*Efficient Heatmap-Guided 6-DoF Grasp Detection in
Cluttered Scenes*, Chen et al., IEEE RA-L 2023) detects grasps in two stages: a
global network predicts a dense **graspable heatmap** plus coarse attributes, and
a local network refines candidates around the peaks. The heatmap is the useful
idea here — it answers "where on this object would a grasp work?" directly,
rather than making us infer it from a bounding box.

`hggd-mcu` is the microcontroller/edge profile: a single quantised network,
monocular input, and the local refinement folded into per-anchor regression
heads. It runs in a few milliseconds on the Pi-class host, which is what makes a
30 Hz assistive loop feasible.

### Output contract

One forward pass over a `1 × 1 × H × W` greyscale input, at stride `S`:

| Head | Shape | Meaning |
|---|---|---|
| `heatmap` | `h × w` | Graspability, 0..1 |
| `angle` | `h × w × A` | Grasp-axis orientation, `A` bins over [0, π) |
| `width` | `h × w` | Gripper opening, normalised to image width |
| `quality` | `h × w` | Predicted grasp success |
| `class` | `h × w × C` | Object class logits (shares the backbone) |

### Decoding

`decode_heatmap` is independent of the inference runtime and unit-tested against
hand-built tensors:

1. **Peak finding** — 3×3 non-maximum suppression above `score_threshold`.
2. **Padding rejection** — peaks on or against the letterbox border are dropped.
   The pad boundary is a synthetic step edge; without this it is the most
   "graspable" structure in every frame.
3. **Sub-cell refinement** — a parabolic fit across the peak's neighbours. At
   stride 8 one cell is ~5 % of the image width; without this the grasp point
   visibly quantises.
4. **Attribute decode** — orientation from the angle-bin argmax, opening from the
   width head, score from heatmap × quality.
5. **Coordinate mapping** — model space back to source-image space through the
   letterbox geometry.
6. **Grasp NMS** — suppression in the joint position/orientation space, with the
   180° symmetry of a grasp axis handled correctly.

### Runtimes

The network is reached through `InferenceSession`, so the same decoding serves:

| Session | Requires | Use |
|---|---|---|
| `OnnxInferenceSession` | `onnxruntime` | Deployment |
| `TfliteInferenceSession` | `tflite_runtime` | int8 on constrained targets |
| `ClassicalHeatmapSession` | nothing | Fallback, and CI |

**The classical session is not a stub.** It computes a real edge-density
graspability heatmap in the *same tensor layout* the network would produce:
Sobel gradients accumulated per cell, graspability rising with edge energy and
falling where the gradient is isotropic (texture, not an edge), orientation from
the dominant gradient direction rotated 90° (fingers should close *across* an
edge, not along it), and width from the structure-tensor coherence.

It is a genuine — if modest — classical baseline, and it is what keeps the system
usable when weights are missing. The degradation is recorded, logged and shown on
the diagnostics screen, so it is never silent.

### Depth

Reference HGGD consumes RGB-D. This build has a monocular camera, so metric depth
comes from size priors (`vision/depth.py`) and is attached after decoding:

```
distance = real_height × focal_length_px / apparent_height_px
```

The catch is that the prior is a *class* prior: bottles range from 15 to 30 cm.
The estimator therefore reports a distance **and an honest error bar**, and its
confidence collapses for unknown classes. Downstream, depth only ever modulates
approach speed and grip aperture — never whether a grasp happens — so a wrong
estimate degrades comfort, not safety.

`TODO(hardware)`: with a depth camera fitted, feed a real depth plane in as a
second input channel and retire the size-prior estimator.

### Installing weights

```
models/
├── manifest.toml
└── hggd_mcu/
    └── hggd_mcu_int8.onnx
```

`ModelRegistry` verifies presence and SHA-256. Checksums matter beyond tidiness:
silently loading a *different* model than the one the affordance thresholds were
tuned against would change the hand's behaviour with no visible cause.

---

## Other backends

| Backend | Capabilities | Purpose |
|---|---|---|
| `hggd_mcu` | detection, classification, grasp | The build's model |
| `onnx_detector` | detection, classification | A generic YOLO-style detector — the second concrete example, proving the abstraction is not shaped around HGGD alone |
| `mock` | everything | Simulation, driven by scene ground truth with configurable error |
| `null` | none | No camera, vision disabled, or a backend that failed to load |

### The mock is allowed to cheat; the real ones are not

`SimulatedCamera` attaches ground truth to `Frame.metadata["scene"]`. Only
`MockVisionBackend` reads it, and only so it can emit *deliberately imperfect*
detections:

```toml
[vision.mock]
confidence_noise = 0.05
false_negative_rate = 0.05   # frames where the object is simply not seen
label_error_rate = 0.03      # plausible confusions: a can looks like a cup
```

Perfect vision is the least useful thing to test against. Fusion, target
selection and the "AI unsure → user still in control" paths only get exercised
when vision is *sometimes wrong*.

`test_real_backends_do_not_read_the_simulation_ground_truth` asserts that
HGGD-MCU never looks at the truth, so the simulation cannot flatter it.

---

## Pipeline

Runs in its own rate group (20 Hz), decoupled from the 200 Hz control loop.
Control never waits for vision; it reads the latest cached result and checks its
age. That decoupling is why a slow model degrades the *quality* of assistance
rather than the *responsiveness* of the hand.

### Tracking

Per-frame detections flicker: seen at 0.9, missed entirely, then reported as a
different class. Choosing a grasp from that produces a hand that changes its mind
mid-reach. The tracker provides:

- **identity** across frames via greedy IoU association;
- **persistence** through brief occlusion (usually by the user's own hand);
- **label voting** — the reported class is the majority over recent history, so
  one odd frame cannot change what the hand thinks it is holding;
- **age**, which fusion uses as a stability term.

Greedy IoU rather than Hungarian assignment or a Kalman filter: with a handful of
objects and a hand-mounted camera, association is rarely ambiguous, and the
simpler algorithm is easier to reason about when something does go wrong.

### Target selection

`VisionResult.primary` is *not* the most confident detection. It ranks by
confidence (45 %), centrality (35 %), track stability (10 %) and size (10 %),
because the hand-mounted camera points where the user is reaching — and the most
confident thing in frame is often a large background object they have no interest
in.

---

## Preprocessing

**Letterbox, never stretch.** A distorted image changes an object's apparent
aspect ratio, which is one of the cues used to tell an upright bottle from one
lying down. `LetterboxInfo` carries the geometry so every predicted coordinate
can be un-padded before it is compared with anything from the original frame.

NumPy fast path where available, pure-Python fallback otherwise — the fallback
exists so the pipeline runs in CI, not because resizing 720p in a Python loop is
a good idea.
