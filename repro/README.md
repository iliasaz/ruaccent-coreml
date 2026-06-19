# repro/ — on-device validation harness (RuaccentProbe)

Validates each converted CoreML model for **numeric parity + latency across all CoreML compute
units** against the onnxruntime logit oracles. Named **RuaccentProbe** (distinct from
chatterbox-coreml's `ANEProbe`) so the two harnesses' logs are never conflated. Mac ≠ iPhone ANE
(carried learning), so we check both: this Apple Silicon Mac first (cheap), then the iPhone/iPad ANE.

## What it checks
For every oracle case it runs the model under **`.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, `.all`**
and reports, per unit:
- **oracle-match** — argmax/decision matches the onnxruntime ground truth.
- **Δ** — max logit divergence vs the `.cpuOnly` baseline (fp16 / accelerator drift).
- **latency** — avg ms over 5 runs (after warmup).

## Run on this Mac (exercises Mac GPU + Neural Engine)
```bash
python3 repro/export_oracles.py                      # -> repro/oracles.json (from converter/fixtures/refs)
cd repro && swift build -c release
./.build/release/RuaccentProbe oracles.json ../converter/_work/coreml          # all models, all units
./.build/release/RuaccentProbe oracles.json ../converter/_work/coreml M2_omograph   # one model
```
Latest Mac result: **all 4 models PASS on every compute unit** (cpu/gpu/ane/all match the oracle on
every case). Δ vs cpu ≤ 7e-2 (gpu, harmless — argmax stable). Latency: M1/M3/M4 0.2 ms, M2 1–2.5 ms.

## Run on iPhone/iPad (authoritative ANE)
The probe logic is plain CoreML — it deploys unchanged inside an iOS app target. Recipe (see the
`ios-device-testing` / `ios-deploy` skills and chatterbox's device notes):

1. **Wrap** `Sources/RuaccentProbe/main.swift`'s logic in a tiny SwiftUI app (a button → run → write
   the report to the app container). Add `oracles.json` + the `.mlpackage`(s) as bundle resources.
2. **Bundle ONE model at a time.** In an Xcode `PBXFileSystemSynchronizedRootGroup` project, a model
   in a *subfolder* still bundles — exclude others by moving them **outside** the synced source folder.
3. **Build + install + launch** on a connected, **unlocked** device via `xcodebuild` + `devicectl`
   (`CODE_SIGNING_ALLOWED` per your signing). Capture logs with `idevicesyslog` **and** the report file
   written to the app container (os.Logger `.notice` isn't reliably captured).
4. If a probe app crashes, an **Xcode CoreML performance report** is a crash-free way to get per-op
   device placement (CPU/GPU/ANE) + latency.

**Pass bar:** every compute unit matches the oracle argmax/decision on every case for every model.
fp16 |Δ| is expected; only argmax/decision flips matter.
