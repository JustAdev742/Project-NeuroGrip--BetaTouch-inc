# Documentation

**Getting started**

| | |
|---|---|
| [installation.md](installation.md) | From clone to running, with or without hardware |
| [demonstration.md](demonstration.md) | Showing this to someone in ten minutes |

**Understanding it**

| | |
|---|---|
| [architecture.md](architecture.md) | Layers, invariants, runtime model, and why each decision was made. **Start here.** |
| [safety.md](safety.md) | Hazard analysis, the response ladder, and an honest list of this design's limits |
| [fusion.md](fusion.md) | The seven gates, walked through one at a time |
| [emg.md](emg.md) | Signal chain, calibration, intent, recording and replay |
| [vision.md](vision.md) | Backends, HGGD-MCU, tracking, depth |
| [modes.md](modes.md) | What actually differs between the four modes |
| [training.md](training.md) | The exercises and what each one trains |

**Working on it**

| | |
|---|---|
| [api.md](api.md) | The interfaces you implement to swap hardware or models |
| [testing.md](testing.md) | What the suite covers, and how to test perception |
| [development.md](development.md) | Conventions, testing strategy, how to extend |
| [hardware.md](hardware.md) | Bill of materials, wiring, assembly, bring-up |
| [protocol.md](protocol.md) | NGP v1 wire format |
| [configuration.md](configuration.md) | Every setting, and the layering rules |

## If you read only one thing

The rule the whole design serves:

> **The AI never replaces the user. The user is always in control.**
> The user decides *when*. The AI decides *how*.

[architecture.md](architecture.md) explains how that is enforced structurally
rather than by convention, and [fusion.md](fusion.md) shows the exact code path.
