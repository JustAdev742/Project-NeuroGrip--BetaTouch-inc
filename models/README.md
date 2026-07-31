# Models

Model weights are **not** in version control. This directory holds the manifest
that declares what should be here, and the files themselves once installed.

```
models/
├── manifest.toml
├── hggd_mcu/
│   └── hggd_mcu_int8.onnx      # not in git
└── gestures/
    └── user_lda.json           # per-user, not shipped
```

## Checking what is installed

```bash
neurogrip diagnose      # the "Models" self-test
```

## What happens when a model is missing

Nothing breaks. The system degrades and says so:

| Missing | Consequence |
|---|---|
| `hggd_mcu` | The backend falls back to the classical edge-density graspability session. Grasp *quality* drops; the hand works. The diagnostics screen shows the degradation. |
| `gestures` | The threshold classifier is used, which needs no training data and is the default anyway. |

That is deliberate. A prosthetic hand that stops working because a model file is
absent would be a worse failure than one that grasps slightly less well.

## Training a per-user gesture model

1. Record labelled sessions:
   ```bash
   neurogrip record var/close.emg --seconds 60 --label close
   neurogrip record var/open.emg  --seconds 60 --label open
   neurogrip record var/rest.emg  --seconds 60 --label rest
   ```
   (The training exercises also label automatically as they run.)

2. Fit an LDA over the Hudgins feature set and write the JSON model described in
   `neurogrip/emg/gestures.py::LinearGestureClassifier`.

3. Point the configuration at it:
   ```toml
   [emg]
   classifier = "linear"
   model_path = "models/gestures/user_lda.json"
   ```

`TODO(training)`: `tools/train_gestures.py` is not yet written. The recording and
replay halves — the parts that need to live in the runtime — are done.
