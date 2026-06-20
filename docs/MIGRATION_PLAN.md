# RUAccent → CoreML migration plan

Port [RUAccent](https://github.com/Den4ikAI/ruaccent) (Russian lexical stress) to an
on-device **CoreML** pipeline, shipped as the reusable Swift package **`RUAccentCoreML`**
(this repo). It is consumed by [`chatterbox-coreml`](https://github.com/iliasaz/chatterbox-coreml)
(multilingual TTS) as a SwiftPM dependency behind the `RussianStressing` protocol, and
is usable standalone. Target: **macOS 15+ / iOS 18+, arm64**, no runtime network.

> This plan is written to be picked up by a **fresh agent session**. It front-loads the
> hard-won CoreML learnings from the chatterbox **turbo** and **multilingual** migrations
> so we don't rediscover them. Track on GitHub as an epic + one issue per phase.

---

## Upstream (confirm in Phase 0 — do not assume)

- **Repo:** https://github.com/Den4ikAI/ruaccent (Python). Models are published on
  HuggingFace; the variant named in the chatterbox plan is **`turbo3.1`**.
- **License:** RUAccent's and the models' licenses **must be confirmed** before
  redistributing weights/dictionaries inside this package.
- **What RUAccent does (verify against the code):** places lexical stress on Russian
  text, restores `ё`, and disambiguates **homographs** (same spelling, different stress,
  e.g. за́мок/замо́к) using sentence context. Native output marks stress with **`+`
  before the stressed vowel**.
- **Likely pipeline (confirm):** normalize/tokenize → for each word:
  1. **accents dictionary** lookup (precomputed stress for known words) → use it;
  2. if the word is a **homograph** (in the omographs dictionary) → run the
     **context disambiguation model** over the sentence;
  3. if **OOV** → run the **neural accentor model** to predict the stressed form;
  4. **`ё` restoration**.
- **Components to port:** (a) neural accentor model, (b) homograph disambiguation
  model, (c) dictionaries (`accents`, `omographs`), (d) tokenizer/preprocessing.

The exact model architectures (char-level seq2seq? token classifier? BERT-class
context model?), tensor I/O, and tokenizer are **the first thing to nail in Phase 0**.

---

## Locked decisions

1. **Separate reusable SPM package** (this repo); chatterbox-coreml depends on it. Keep
   the package **binary-free**: nothing large is committed. The on-device bundle — the four
   CoreML `.mlpackage`s, the packed `.rapack` dictionaries (full `accents`, 21.75 MiB), and
   the tokenizer files — is hosted in the **private HF repo `iliasaz/ruaccent-coreml`** and
   downloaded on-device by `ModelRepository` (swift-transformers `Hub`), mirroring how
   chatterbox-coreml hosts `iliasaz/chatterbox-turbo-coreml`. A consumer can also supply a
   local directory via `RUAccent(modelDirectory:)`. (Earlier plan: ship a ~11 MB dict extract
   as a committed package resource — superseded by HF hosting once the full dict was locked.)
2. **Output contract:** `U+0301` combining acute placed **after** the stressed vowel by
   default (TTS convention). `.plusBeforeVowel` option reproduces RUAccent's `+` for parity.
3. **Manual stress always wins** — an existing `U+0301` (or `+`) in the input is preserved.
4. **Resolution order:** manual → accents dictionary → homograph(context) model → neural
   OOV model → `ё`. The dictionary is the primary source; models are fallbacks.
5. **CoreML fp16, ANE-resident**, validated **on device** (Mac parity ≠ iPhone ANE).
6. **Sync `stress(_:)`, async `init`** — model load is async; per-call stressing is sync
   and fast (dictionary + small models).

---

## Phases (each closes only when its exit criterion is met + recorded in its issue)

### Phase 0 — Investigate & pin upstream
Clone RUAccent into `ruaccent-src/` (gitignored). Identify the exact `turbo3.1` accentor
+ homograph model artifacts, their PyTorch graphs and I/O, the tokenizer, and the
dictionaries; document the real pipeline. Confirm licenses. Stand up a `.venv`
(coremltools 9.0 / torch 2.8 / Python 3.13, matching chatterbox). Write a Python script
that runs RUAccent and dumps a `(input → stressed)` **fixture set** = the ground truth.
**Exit:** architecture documented here; fixtures generated; license cleared.

### Phase 1 — Convert the neural accentor → CoreML
PyTorch → CoreML fp16. Carry the T3 learnings below (manual SDPA if it's a transformer
with q_len≫1; fp16 norm hardening; static shapes; inline positional encodings). Validate
Mac-CPU parity vs PyTorch, then **on device** with a small RuaccentProbe (`repro/`) probe.
**Exit:** accentor stress-accuracy ≥ agreed target on **iPhone ANE** (not just Mac).

### Phase 2 — Convert the homograph disambiguation model → CoreML
Same methodology (it's context-driven, likely a small BERT-class classifier).
**Exit:** homograph accuracy parity vs Python on device.

### Phase 3 — Pack dictionaries + Swift loader
Pack the full `accents`/`omographs`/`yo` dicts into mmap-able `.rapack` v2 (`converter/pack_dicts.py`;
`docs/SWIFT_DICT_PACK.md`), write the Swift loader + lookup (`Sources/RUAccentCoreML/Dict/`). The packs
ship via the private HF bundle (not committed), downloaded by `ModelRepository`. **Exit:** Swift
dictionary lookup byte-matches Python ✔ (DictReaderTests).

### Phase 4 — Swift preprocessing / tokenizer
Char-level Russian tokenization + the deterministic Unicode steps (lowercase, NFKD,
`ё`/`й` handling, word/punctuation splitting) the model expects.
**Exit:** Swift token stream matches the Python tokenizer on the fixture set.

### Phase 5 — Swift API + resolution
Implement `RUAccent: RussianStressing`: manual → dict → homograph → neural → `ё`, with
`StressNotation` output. **Exit:** end-to-end `stress(_:)` matches Python RUAccent on the
fixture set within agreed accuracy (incl. homographs `мой→мо́й`, `русский`, `тёплый`,
`ёжик`, `йога`, and context pairs).

### Phase 6 — Size + latency
Palettize the model(s) (8-bit k-means; skip the output head). Measure on-device latency
and package size. **Exit:** acceptable latency/size on iPhone for a sentence.

### Phase 7 — Integrate into chatterbox-coreml
Add as a SwiftPM dependency behind the chatterbox Phase-4 stress interface; packed
dictionary as the fallback. **Exit:** chatterbox synthesizes Russian with correct stress
end-to-end on device.

---

## Carried CoreML learnings (from chatterbox turbo + multilingual)

**Process / methodology — the meta-lesson (cost real hours):**
- **Evidence-based, ablation-driven.** Before interpreting *any* failure, verify *what was
  actually tested* — the real inputs, the artifacts actually bundled/loaded, the exact
  config. Change **one variable at a time** and attribute the delta. Never act on an
  **unproven premise** or an **assumed side effect** (it caused a multi-hour misdiagnosis
  where a different bundled model was the real crasher). Resolve ambiguous logs to the
  real identity before concluding.
- **Mac CoreML correctness ≠ iPhone ANE.** Mac CPU/GPU/ANE can all be perfect while the
  device diverges or won't compile. **Always device-test** with a small probe app.
- **On-device probe gotchas:** bundle **one model at a time**; in an Xcode
  `PBXFileSystemSynchronizedRootGroup` project, a model in a *subfolder* (e.g. `_hold/`)
  **still bundles** — exclude by moving it **outside the synced source folder**. An
  **Xcode CoreML performance report** is a crash-free way to get per-op device placement
  + latency when a probe app crashes.

**Conversion / numerics:**
- **fp16 norm overflow.** If the model has RMSNorm/LayerNorm, fp16 `mean(x²)` overflows
  when activations exceed ~256 → `inf`→`rsqrt 0` → **zeroed outputs**. HF norms upcast to
  fp32 internally but `compute_precision=FLOAT16` erases that. Fix with a rescale-in-fp16
  norm (see chatterbox `HardenedRMSNorm`) or keep the norm fp32.
- **ANE drops the explicit `attn_mask` for fused attention at q_len≫1** — manually
  decompose SDPA (`matmul(q,kᵀ)·scale + mask → softmax → matmul`) for any encoder/prefill
  attention; q_len=1 can keep fused.
- **Keep ALL traced shapes static Python ints** — any `.shape`-derived size becomes an
  `aten::Int` the CoreML torch frontend rejects. Inline positional encodings if the HF
  rotary/embedding helpers emit dynamic ops.
- **Palettize** weights (8-bit k-means) for size; **skip the output/classification head**
  (`OptimizationConfig.set_op_name(name, None)`); k-means is deterministic so shared
  weights still dedup across merged functions.
- **Don't commit binaries** (`.mlpackage`/`.npy`/weights) — gitignored here; dicts ship
  as small packed resources.

**Env / conventions:**
- This machine is authed to HuggingFace as `iliasaz`; `HF_HOME=/Users/ilia/Downloads/HF_HOME`
  (**export it explicitly** in non-interactive shells or files land in `~/.cache`).
- Match chatterbox's `.venv`: coremltools 9.0, torch 2.8.0, Python 3.13.
- Device recipe (from chatterbox CLAUDE.md): build/install via `xcodebuild` + `devicectl`;
  capture os.Logger with `idevicesyslog` **and** a file written to the app container
  (os.Logger `.notice` isn't reliably captured); device must be **unlocked** to launch.

---

## Integration contract (what chatterbox-coreml consumes)

```swift
public protocol RussianStressing: AnyObject {
    func stress(_ text: String, notation: StressNotation) throws -> String  // manual mark always wins
}
public enum StressNotation: Sendable { case combiningAcute /* U+0301 after vowel */, plusBeforeVowel }

let accentor = try await RUAccent(modelDirectory: modelsURL)   // nil = dictionary/passthrough-only fallback
let stressed = try accentor.stress("мой тёплый дом")           // "мо́й тёплый до́м"
```

chatterbox depends on the **protocol**, so the dictionary-only fallback and the full
model path are interchangeable.
