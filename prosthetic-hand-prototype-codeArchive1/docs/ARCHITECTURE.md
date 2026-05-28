# Architecture Overview

This prototype follows a simple pipeline:

1) Camera captures frames from the palm view.
2) Vision model detects the object and returns a label plus confidence.
3) EMG provides user intent (open, close, hold, cancel).
4) Grip selector fuses EMG intent with vision confidence and mapping rules.
5) Hand controller drives 5 servos to the target grip.
6) Dashboard shows live telemetry and system state.

Mock mode replaces camera, vision, EMG, and servo hardware with simulated services while keeping the rest of the system identical.
