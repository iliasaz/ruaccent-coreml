# CLAUDE.md

On-device **Russian word stress (accentuation)** for Apple Silicon — a CoreML port of
[RUAccent](https://github.com/Den4ikAI/ruaccent), shipped as the Swift package
**`RUAccentCoreML`**. Consumed by `chatterbox-coreml` (multilingual TTS) behind the
`RussianStressing` protocol; usable standalone. macOS 15+ / iOS 18+, **arm64**.

**Read [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) first** — it has the phased
plan, locked decisions, the integration contract, and the carried CoreML learnings.

Canonical repo: **https://github.com/iliasaz/ruaccent-coreml** (private). `origin` only.
Upstream Python: **https://github.com/Den4ikAI/ruaccent** (model variant `turbo3.1`;
confirm artifacts + **license** in Phase 0). Sister project (the consumer + source of
the learnings): `~/Developer/chatterbox-coreml`.

## Build & test
```bash
swift build
swift test
```
Keep the package **green and binary-free** from day one (the scaffold builds; add real
behavior per the plan's phases). Verify both platforms when touching anything
CoreML/Swift-runtime: `xcodebuild ... -destination 'platform=macOS'` and
`'generic/platform=iOS Simulator'` (`CODE_SIGNING_ALLOWED=NO`).

## Layout
- `Sources/RUAccentCoreML/` — the package (`RussianStressing` protocol + `RUAccent`).
- `converter/` — Python → CoreML conversion + fixture-dump scripts. **Outputs gitignored.**
- `Resources/` (Phase 3) — packed `accents`/`omographs` dictionary extract (~11 MB).
- `repro/` (Phase 1+) — on-device validation harness (**RuaccentProbe**; named distinctly from
  chatterbox's `ANEProbe` to avoid log conflation).
- `docs/MIGRATION_PLAN.md` — the plan.

## Working style (carried from chatterbox — these were learned the hard way)
- **Evidence-based, ablation-driven.** Verify what was *actually* tested (real inputs,
  bundled/loaded artifacts, exact config) before interpreting a failure. Change ONE
  variable at a time. Never act on an **unproven premise** or an **assumed side effect**.
- **Mac CoreML correctness ≠ iPhone ANE** — always device-test. Bundle **one model at a
  time** in the probe; in Xcode synchronized-root projects, exclude others by moving them
  **outside** the source folder (a subfolder still bundles). An **Xcode CoreML perf
  report** gives crash-free per-op placement + latency.
- **fp16 norm overflow** (RMSNorm/LayerNorm): harden with rescale-in-fp16 or fp32 norm.
  **ANE fused attention drops `attn_mask` at q_len≫1** → manually decompose SDPA. Keep
  traced shapes **static ints**. **Palettize** for size (skip the output head). Don't
  commit binaries.

## Conventions
- HF auth: machine is `iliasaz`; `HF_HOME=/Users/ilia/Downloads/HF_HOME` (export it
  explicitly in non-interactive shells). `.venv`: coremltools 9.0 / torch 2.8 / Python 3.13.
- Don't bump dependency versions or commit without being asked. Match surrounding style.
- The integration boundary is the `RussianStressing` protocol + `StressNotation`; default
  output is `U+0301` **after** the stressed vowel; **manual stress in the input always wins**.
