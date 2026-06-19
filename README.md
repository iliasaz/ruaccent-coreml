# RUAccentCoreML

On-device **Russian word stress (ударение / accentuation)** for Apple Silicon — a
CoreML port of [RUAccent](https://github.com/Den4ikAI/ruaccent), packaged as a
reusable **Swift Package**. Built to be consumed by
[`chatterbox-coreml`](https://github.com/iliasaz/chatterbox-coreml) (multilingual
on-device TTS) behind a small protocol, and usable standalone.

- Platforms: **macOS 15+ / iOS 18+, arm64**.
- No network at runtime; models + a packed dictionary ship on-device.
- Status: **scaffold** — interface fixed, port in progress. See
  [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md).

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

RUAccent (Python): https://github.com/Den4ikAI/ruaccent — model variant of interest
`turbo3.1`. Confirm exact model artifacts, dictionaries, and **license** in Phase 0.
