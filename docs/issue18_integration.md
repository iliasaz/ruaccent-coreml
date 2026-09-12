# Phase 7 — Integrate `ruaccent-coreml` stress into chatterbox-coreml

A copy-pasteable wiring guide for a `chatterbox-coreml` maintainer. Every step below is grounded in
the verified findings (real models run; trial build compiled clean). Where chatterbox's inline doc
sketch (`MTLTextTokenizer.swift:24-28`) is wrong, this guide says so and gives the correct version.

---

## 1. What this is

`RUAccentCoreML` is an on-device, CoreML port of [RUAccent](https://github.com/Den4ikAI/ruaccent)
that places Russian lexical stress (and restores `ё`). It plugs straight into chatterbox's existing
`RussianStressing` seam in `MTLTextTokenizer` — the same injection point already used by
`ManualRussianStress` (identity) and `DictionaryRussianStress` (per-word dict fallback). Wiring it as
the Phase-7 neural source means the multilingual TTS path (`language == "ru"`) gets full
dictionary + homograph + neural-OOV stress instead of relying on caller-supplied marks. The package
is pure Swift + SPM, binary-free (weights live on a private HF repo), macOS 15 / iOS 18, arm64.

---

## 2. Add the SwiftPM dependency

The runtime currently lives **only** on branch `coreml-model-conversions` (not `main`, not tagged —
`git tag` is empty). Pick one of three dependency forms for chatterbox's `Package.swift`.

```swift
// chatterbox-coreml/Package.swift — add ONE to the `dependencies:` array

// (a) DEV default — local path, no GitHub auth, works today:
.package(path: "../ruaccent-coreml"),

// (b) BRANCH — where the runtime lives right now (private repo → needs git creds, see below):
.package(url: "https://github.com/iliasaz/ruaccent-coreml.git", branch: "coreml-model-conversions"),

// (c) TAGGED — switch to this once the runtime is merged to main and tagged:
.package(url: "https://github.com/iliasaz/ruaccent-coreml.git", from: "0.1.0"),
```

Add the product to the **`ChatterboxCoreML` target's** `dependencies:`:

```swift
.product(name: "RUAccentCoreML", package: "ruaccent-coreml"),
```

> **SPM gotcha:** the *product* name is `RUAccentCoreML` (declared in `ruaccent-coreml/Package.swift:11`)
> but the *package identity* in `.product(package:)` is `ruaccent-coreml` (the repo/dir name,
> lowercase-hyphenated). They differ — a common mistake.

**Private-repo auth (options b/c only):** the GitHub repo is private over HTTPS (no SSH remote
configured). SPM needs git credentials to fetch it — an Xcode-signed-in GitHub account, a credential
helper, or `~/.netrc`:

```
machine github.com login <github-username> password <github-PAT-with-repo-scope>
```

This GitHub auth is **separate** from the HF token (§5). The local-path option (a) sidesteps it and
is the friction-free dev default.

**Shared transitive dep — verified safe.** Both packages depend on `swift-transformers` (`from: "1.3.0"`)
and both `Package.resolved` pin the *identical* revision `2fa33e1…` / 1.3.3. SPM dedups to one copy;
both consume only its `Hub` product. No conflict. (Confirmed by the trial build's resolved graph.)

---

## 3. The adapter

The two `RussianStressing` protocols are **different types** and an adapter is mandatory:

| | chatterbox | RUAccentCoreML |
|---|---|---|
| decl | `MTLTextTokenizer.swift:33-35` | `RussianStressing.swift:21-27` |
| shape | `protocol RussianStressing: Sendable`<br>`func stress(_ nfkdText: String) -> String` | `protocol RussianStressing: AnyObject`<br>`func stress(_ text:, notation:) throws -> String` |
| | sync, non-throwing, **Sendable** | throwing, has `notation`, **not Sendable** |

The naive sketch already in chatterbox (`MTLTextTokenizer.swift:24-28`,
`accentor.stress(nfkdText)`) **does not compile and is semantically wrong** — wrong arity, missing
`try`, missing notation, no Sendable bridge, and it feeds raw NFKD straight in (corrupts `ё`/`й`,
see below). Use this verified adapter instead. **Put it inside the `ChatterboxCoreML` module** so the
unqualified `RussianStressing` binds to chatterbox's protocol (both modules also declare
`ModelRepository` and `RUAccentError`, so qualify any of those you reference).

```swift
// chatterbox-coreml/Sources/ChatterboxCoreML/RuAccentStress.swift
import Foundation
import RUAccentCoreML

/// Bridges `ruaccent-coreml`'s `RUAccent` to chatterbox's `RussianStressing`.
///
/// `MTLTextTokenizer` hands us text that is already **lowercased + NFKD** (so `ё`→`е`+U+0308,
/// `й`→`и`+U+0306), and afterwards replaces `"+"`→U+0301 **in place** (MTLTextTokenizer.swift:126-130).
/// Two correctness traps, both empirically verified against the real models:
///
///  1. NOTATION — ask for `.combiningAcute` (U+0301 *after* the vowel). RUAccent's other notation
///     (`.plusBeforeVowel`, e.g. `"м+ой"`) would, after chatterbox's in-place `"+"`→U+0301 replace,
///     land the acute on the *preceding consonant* (`"м́ой"`). With `.combiningAcute` RUAccent already
///     emits the acute, so chatterbox's replace is a harmless no-op. (.combiningAcute is the default.)
///
///  2. NFKD round-trip — RUAccent's `normalize` accepts only *composed* Cyrillic and STRIPS bare
///     combining marks (U+0308/U+0306/U+0301). Feeding raw NFKD corrupts `ё`/`й` words
///     (`"мой"`→`"мои́"`, `"самолёт"`→`"самолё́т"`). So NFC-compose before stressing and re-NFKD the
///     result, matching what chatterbox's grapheme tokenizer was trained on.
///
/// On any failure we return the input unchanged so TTS degrades to caller-supplied marks.
///
/// `@unchecked Sendable` is sound: `RUAccent` is a `final class` wrapping load-once, immutable CoreML
/// models; `stress(_:notation:)` is a read-only inference path with no mutable shared state.
struct RuAccentStress: RussianStressing, @unchecked Sendable {
    let accentor: RUAccent

    func stress(_ nfkdText: String) -> String {
        let composed = nfkdText.precomposedStringWithCanonicalMapping        // NFC for RUAccent
        guard let stressed = try? accentor.stress(composed, notation: .combiningAcute) else {
            return nfkdText                                                  // fail-open
        }
        return stressed.decomposedStringWithCompatibilityMapping            // back to NFKD
    }
}
```

**Why each line — proven by running the real models:**

- **NFKD round-trip is REQUIRED (not passthrough).** `normalize` (`Normalize.swift:30-43,46-52`) deletes
  bare U+0308/U+0306/U+0301. Verified corruption with raw-NFKD passthrough vs. correct NFC-roundtrip:
  `мой` → `мои́` (wrong) vs `мо́й` ✓; `самолёт` → `самолё́т` (wrong) vs `самолёт` ✓;
  `война` → `во́ина` (wrong) vs `война́` ✓; `ёж` → `ё́ж` (wrong) vs `ёж` ✓.
- **`.combiningAcute` is mandatory (not `.plusBeforeVowel`).** With `.plusBeforeVowel`, after
  chatterbox's in-place `+`→U+0301 the acute lands on the *consonant*: `мой` → `м́ой`,
  `молоко` → `молоќо`. With `.combiningAcute` RUAccent emits U+0301 already and chatterbox's replace
  is a no-op.
- **Return NFKD (`.decomposedStringWithCompatibilityMapping`).** chatterbox NFKD'd before calling and
  its grapheme tokenizer is NFKD-trained. At scalar level the round-trip is well-formed:
  `всё́` → `в с е(U+0435) U+0308 U+0301`; `мо́й` → `м о U+0301 и(U+0438) U+0306`.
- **`@unchecked Sendable` compiled clean** in the trial build and satisfies chatterbox's
  `RussianStressing: Sendable`. Plain words and manual-mark words are identical under passthrough and
  round-trip (`молоко`→`молоко́`; manual `сл+ово`→`сло́во`, `сло́во`(U+0301)→`сло́во` preserved).

---

## 4. Wire it at model load

`ChatterboxCoreMLModel.load(from:russianStress:)` already defaults to `ManualRussianStress()`
(`ChatterboxCoreMLModel.swift:75-78`) and plumbs the source into the multilingual tokenizer
(`…:124`). Replace the default with the adapter.

**Default pattern (i): RUAccent downloads its own private bundle**, sharing chatterbox's HF token and
HF_HOME (§5):

```swift
// token + hfHome are the SAME values chatterbox already uses for its own model download.
let token = TokenStore.load()                                   // keychain on device; env/file fallback
let accentor = try await RUAccentCoreML.RUAccent(
    downloadingFrom: RUAccentCoreML.ModelRepository.defaultRepoId,   // "iliasaz/ruaccent-coreml"
    hfHome: effectiveHFHome,                                    // SAME cache root as chatterbox's model
    hfToken: token.isEmpty ? nil : token                        // nil → env/file fallback still fires
)
let model = try await ChatterboxCoreMLModel.load(
    from: URL(fileURLWithPath: modelPath),
    russianStress: RuAccentStress(accentor: accentor)
)
```

`init(downloadingFrom:hfHome:hfToken:configuration:progress:)` is at `RUAccent.swift:146-156`.

**Offline / local-dev alternative (no network, no token)** — `converter/_work` already holds
`coreml/`, `dictpack/`, `nn/`, exactly the layout `init(modelDirectory:)` expects
(`RUAccent.swift:125-139`):

```swift
let accentor = try RUAccentCoreML.RUAccent(
    modelDirectory: URL(fileURLWithPath: "/path/to/ruaccent-coreml/converter/_work"))
```

**Both honor manual stress.** A caller-supplied `+` or U+0301 in the input is preserved per-word and
survives the NFC round-trip (`ManualStress.swift`, `RUAccent.swift:166-179`). The stresser only runs
for `language == "ru"` (`MTLTextTokenizer.swift:128`), so non-`ru` text is untouched — and RUAccent's
normalize stripping disallowed punctuation is harmless on that path.

---

## 5. HF token & cache

One `iliasaz` account owns **both** private repos (`iliasaz/chatterbox-turbo-coreml` and
`iliasaz/ruaccent-coreml`), so a single HF read token authorizes both downloads. RUAccent's token
resolution order (`ModelRepository.swift:130-141`) is **identical** to chatterbox's — pass whatever
chatterbox already resolves:

```
explicit hfToken param → HF_TOKEN env → HUGGING_FACE_HUB_TOKEN env
                       → (via HubApi) $HF_HOME/token, ~/.cache/huggingface/token
```

- **iOS/macOS app:** token from the Keychain via `TokenStore.load()`; pass `token.isEmpty ? nil : token`
  so the env/file fallback still fires when empty.
- **CLI/dev:** `HF_TOKEN` env or `~/.cache/huggingface/token`.

> Verify the token's *scope* before shipping: a fine-grained PAT could be limited to one repo. It must
> have read access to **both** `iliasaz/*` repos.

**Shared cache.** Both `ModelRepository`s resolve the hub base identically: `<hfHome>/hub`, with the
same env order `HF_HUB_CACHE → HF_HOME+/hub → ~/.cache/huggingface/hub`
(`ruaccent ModelRepository.swift:35-54`). Pass the **same** `effectiveHFHome` to both downloads and
they write sibling dirs under one cache (`<hub>/models/iliasaz/chatterbox-turbo-coreml` and
`…/ruaccent-coreml`) — no duplication, no conflict. If you hand RUAccent a different HF_HOME than
chatterbox uses, both still work but land in separate caches (no sharing).

---

## 6. Verify

1. **Sanity sentences.** Feed a few `ru` strings through the multilingual path and confirm these
   verified outputs (after chatterbox's `+`→U+0301, shown composed):

   | input | expected |
   |---|---|
   | `мой` | `мо́й` |
   | `война` | `война́` |
   | `самолёт` | `самолёт` (ё preserved, no extra mark) |
   | `ёж` | `ёж` |
   | `её ёлка под окном` | `её ёлка по́д окно́м` |
   | `на двери висит замок` | `на двери́ виси́т замо́к` (homograph picks lock = замо́к) |
   | `сл+ово тут` (manual) | `сло́во тут` (manual stress kept) |

2. **Live-hub test (opt-in).** ruaccent's hub round-trip test is gated behind an env flag
   (`Tests/RUAccentCoreMLTests/ModelRepositoryTests.swift:8,16`):

   ```bash
   RUACCENT_HUB_TEST=1 HF_TOKEN=<token> swift test
   ```

3. **Sentence-level parity test (recommended).** Add a chatterbox test over restored-`ё` words (M4)
   asserting the adapter's NFKD output survives the round-trip — this is the one case the naive sketch
   broke.

4. **Issue #18 exit criterion.** On-device stress matches the ruaccent runtime within the agreed
   accuracy; bundle size and per-call latency are acceptable as one-time TTS text-prep. (4-model ANE
   per-call latency is unmeasured; a homograph-heavy sentence runs M2 per ambiguous word — measure on
   device if it matters.)

**Optional fallback ordering.** If you want graceful degradation when the bundle isn't present, keep
`DictionaryRussianStress` as a fallback: try to construct `RuAccentStress`, and on failure fall back
to the dictionary source (both honor manual marks). The adapter itself already fails open per-call
(returns input unchanged on any error).

---

## 7. Status & links

- **RUAccentCoreML runtime** lives on branch **`coreml-model-conversions`** (not `main`, untagged).
  52/52 golden-parity tests green; builds macOS 15 / iOS 18. Merge/tag before pinning a versioned dep.
- **Trial build:** a throwaway chatterbox worktree with the local-path dep + this adapter
  **compiled clean** (`swift build` → `Build complete!`), `@unchecked Sendable` satisfied, no
  name-collision errors, swift-transformers deduped to the shared revision.
- **Model bundle:** private HF repo `iliasaz/ruaccent-coreml` (downloaded via `ModelRepository`,
  `RUAccent.swift:146-156`).
- **Cross-links:** ruaccent epic [iliasaz/ruaccent-coreml#1]; this is chatterbox **issue #18**
  (Phase 7).

### Open questions for the chatterbox team

- **`ё` + U+0301 redundancy.** For a stressed monosyllable like `всё`, RUAccent emits `ё`+U+0301
  (NFKD: `е`+U+0308+U+0301) — the acute is arguably redundant since `ё` already implies stress.
  Confirm the grapheme tokenizer treats `ё`+U+0301 the same as bare `ё`; if not, the adapter can drop
  a U+0301 that immediately follows `ё`. (Multisyllabic ё-words like `самолёт` get **no** extra mark.)
- **`RUAccent.Configuration` passthrough.** The adapter uses whatever `Configuration` the injected
  `RUAccent` was built with (default = all models on). Decide whether to expose dict-only / model
  toggles to the TTS layer.
- **On-device latency.** The 4-model ANE pipeline's per-call cost as TTS text-prep is unmeasured;
  homograph-heavy sentences run M2 per ambiguous word. Measure on device if it's on the hot path.
- **Concurrency.** `@unchecked Sendable` is sound for serial use; if you drive many concurrent
  `stress(...)` across actors, serialize or wrap `RUAccent` in an actor (not stress-tested here).

### Checklist

- [ ] Add the `.package(...)` dep + `.product(name: "RUAccentCoreML", package: "ruaccent-coreml")`.
- [ ] Add `Sources/ChatterboxCoreML/RuAccentStress.swift` (the §3 adapter).
- [ ] Build a `RUAccent` (download with shared token/HF_HOME, or `modelDirectory:` offline).
- [ ] Pass `RuAccentStress(accentor:)` to `ChatterboxCoreMLModel.load(from:russianStress:)`.
- [ ] Run the §6 sanity sentences + (optional) parity test; confirm `ё`/`й` words survive.
- [ ] Before shipping a versioned dep: merge runtime to `main`, tag, switch to option (c).
