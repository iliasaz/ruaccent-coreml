# RUAccentCoreML

On-device **Russian word stress (ударение / accentuation)** for Apple Silicon — a
CoreML port of [RUAccent](https://github.com/Den4ikAI/ruaccent), packaged as a
reusable **Swift Package**. Built to be consumed by
[`chatterbox-coreml`](https://github.com/iliasaz/chatterbox-coreml) (multilingual
on-device TTS) behind a small protocol, and usable standalone.

- Platforms: **macOS 15+ / iOS 18+, arm64**.
- No network at runtime; models + a packed dictionary ship on-device.
- Status: **complete and shipping.** All four models are converted and running on
  device, 58 tests / 7 suites pass, and end-to-end golden parity against the Python
  reference is **52/52** (`InternalPipelineTests.endToEndGoldenParity`). It is the
  Russian stress path in `chatterbox-coreml`'s multilingual pipeline.
  [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) records how it got there.

## Interface (integration contract)

```swift
import RUAccentCoreML

let accentor = try await RUAccent(modelDirectory: modelsURL)   // or nil = dict/passthrough-only
let stressed = try accentor.stress("мой тёплый дом")           // → "мо́й тёплый до́м"  (U+0301 after the vowel)
```

`stress(_:notation:)` resolves in priority order — **manual mark in the input always
wins** → accents dictionary → context homograph model → neural OOV accentor — and
restores `ё`. Default notation is `U+0301` *after* the vowel (TTS convention);
`.plusBeforeVowel` reproduces RUAccent's native `+`-before output.

Consumers depend on the `RussianStressing` protocol so a dictionary-only fallback can
be swapped for the full model path.

## Layout

| Path | Purpose |
|---|---|
| `Sources/RUAccentCoreML/` | the Swift package (protocol + `RUAccent`) |
| `docs/MIGRATION_PLAN.md` | phased port plan + carried CoreML learnings |
| `converter/` | Python → CoreML conversion + fixture-dump scripts (outputs gitignored) |
| `repro/` | on-device validation harness (RuaccentProbe), added in Phase 1 |

## Upstream

RUAccent (Python): https://github.com/Den4ikAI/ruaccent — model variant `turbo3.1`.

**Licence: MIT**, settled and verified at HEAD (2026-09-05). Upstream relicensed
from Creative Commons to MIT in commit `39543da` (2026-07-17, *"Change license from
Creative Commons to MIT"*), and the weights this port repackages —
[`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator) — are tagged
`license:mit` and are not gated. Redistribution of the converted artifacts is fine
with attribution; see [`NOTICE`](NOTICE).

A caution worth keeping, since it cost real time on the consuming project: the
`ruaccent-src/` checkout in this working tree is **pinned at `704bd30`
(2024-10-24)**, whose `LICENSE` is still the pre-relicense CC BY-NC-ND text. Reading
a licence out of a pinned vendored copy tells you that pin's state, not the
project's. Check the upstream ref.
