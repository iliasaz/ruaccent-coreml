# repro/ — on-device validation harness (ANEProbe)

Validates each converted CoreML model for **CPU-vs-Neural-Engine numeric parity** + latency against
the onnxruntime logit oracles. Mac ≠ iPhone ANE (carried learning), so we check both: this Apple
Silicon Mac's ANE first (cheap), then the iPhone/iPad ANE (authoritative).

## What it checks
For every oracle case it runs the model under `.cpuOnly` and `.cpuAndNeuralEngine` and reports:
- **cpu==oracle** — CoreML CPU argmax/decision matches the onnxruntime ground truth.
- **ane==cpu** — the Neural Engine produces the same argmax/decision as CPU (the thing that breaks on device).
- **|Δcpu,ane|** — max logit divergence between the two compute paths (fp16 ANE drift).
- **latency** — per-prediction ms on the ANE path.

## Run on this Mac (uses the Mac Neural Engine)
```bash
python3 repro/export_oracles.py                      # -> repro/oracles.json (from converter/fixtures/refs)
cd repro && swift build -c release
./.build/release/ANEProbe oracles.json ../converter/_work/coreml          # all models
./.build/release/ANEProbe oracles.json ../converter/_work/coreml M2_omograph   # one model
```
Latest Mac-ANE result: **all 4 models PASS** (cpu==oracle and ane==cpu on every case; |Δ| ≤ 2.3e-2;
M1 0.2ms / M2 ~1.5–2.6ms / M3,M4 0.1ms).

## Run on iPhone/iPad (authoritative ANE)
The probe logic is plain CoreML — it deploys unchanged inside an iOS app target. Recipe (see the
`ios-device-testing` / `ios-deploy` skills and chatterbox's device notes):

1. **Wrap** `Sources/ANEProbe/main.swift`'s logic in a tiny SwiftUI app (a button → run → write the
   report to the app container). Add `oracles.json` + the `.mlpackage`(s) as bundle resources.
2. **Bundle ONE model at a time.** In an Xcode `PBXFileSystemSynchronizedRootGroup` project, a model
   in a *subfolder* still bundles — exclude others by moving them **outside** the synced source folder.
3. **Build + install + launch** on a connected, **unlocked** device via `xcodebuild` + `devicectl`
   (`CODE_SIGNING_ALLOWED` per your signing). Capture logs with `idevicesyslog` **and** the report file
   written to the app container (os.Logger `.notice` isn't reliably captured).
4. If a probe app crashes, an **Xcode CoreML performance report** is a crash-free way to get per-op
   device placement (CPU/GPU/ANE) + latency.

**Pass bar:** `ane==cpu` (argmax/decision) on every case for every model — i.e. the iPhone ANE
reproduces the rendered stress. fp16 |Δ| is expected; only argmax/decision flips matter.
