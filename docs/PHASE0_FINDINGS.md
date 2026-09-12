# Phase 0 — RUAccent upstream, pinned

Authoritative result of Phase 0 (issue #2): the real architecture, the exact model I/O,
the license position, and the ground-truth fixtures every later phase validates against.
Supersedes the "likely pipeline (confirm)" guesses in `MIGRATION_PLAN.md`.

Upstream pinned at **RUAccent `1.5.8.3`**, model variant **`turbo3.1`**, run as
`process_all(use_dictionary=True, tiny_mode=False)` — the canonical full pipeline and our
**parity oracle**. Source clone in `ruaccent-src/` (gitignored); HF artifacts from
[`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator).

---

## 1. License (the gating decision)

> **⚠️ SUPERSEDED — corrected 2026-09-05. The table below is what was true when this
> was written; both of its licence readings are now wrong, and neither should be
> quoted.** Upstream `Den4ikAI/ruaccent` **relicensed to MIT** in commit `39543da`
> (2026-07-17, *"Change license from Creative Commons to MIT"*), and
> [`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator) is now tagged
> **`license:mit`**, not `apache-2.0`. Verified at HEAD via the GitHub and HF APIs.
> There is no NonCommercial or NoDerivatives constraint on anything here, so the
> "clean-reimplementation" position is no longer a *legal requirement* — it remains
> an accurate description of how the port was built, and the code is MIT © Ilia
> Sazonov either way. See [`NOTICE`](../NOTICE). The entry is left standing as a
> dated record; **note also that the `ruaccent-src/` clone is pinned at `704bd30`
> (2024-10-24), so its `LICENSE` file still shows the old CC text** — that pin is
> exactly how the stale reading survived.

| | What | License | Implication |
|---|---|---|---|
| **Weights + dictionaries** | HF repo `ruaccent/accentuator` | **`apache-2.0`** (HF card tag) | The artifacts we convert + redistribute are OK, **including commercially**, with attribution. |
| **Python code** | GitHub `Den4ikAI/ruaccent` | **CC BY-NC-ND 4.0** (`LICENSE` file) — *but* `pyproject.toml` says `Apache Software License` | Repo is internally contradictory. NoDerivatives/NonCommercial would bind the **code**, not the Apache-tagged weights. |

**Working position:** redistribute the **Apache-2.0** weights/dictionaries; **reimplement** the
pipeline in Swift/Python from the spec below (do **not** copy the CC-licensed Python verbatim).
The README routes commercial questions to the author's Telegram, so a written confirmation is
available if certainty is wanted (recommended, since the consumer `chatterbox-coreml` may ship
commercially). **→ User decision: accept the Apache-weights + clean-reimpl position, or get
written clarification before redistributing.**

---

## 2. The real pipeline (`process_all_internal`, per sentence)

```
text
 → normalize: re.sub("[^a-zA-Z0-9\s а-яёА-ЯЁ —.,!?:;…«»„""''(){}\[\]-]", "", text)   # strip disallowed
 → TextPreprocessor.split_by_sentences  (razdel.sentenize — rule-based)
 → for each sentence:
     words, remaining_text = split_by_words(sentence)         # regex; keeps inter-word spans
     stress_usages = M3.predict_stress_usage(sentence)        # per-word STRESS / NO_STRESS / PUNCT
     words = _process_yo(words, sentence):                    # ё restoration
         if 'е' in sentence: yo_pred = M4.predict_yo_homographs(sentence)   # per-word YO / NO_YO / PUNCT
         word = yo_words.get(word, word)                      # dict (85 568)
         if yo_pred[i]=="YO": word = yo_homographs.get(word, word)          # dict (637)
     words = _process_omographs(words):                       # homograph disambiguation
         for word in omographs dict (19 741): mark ' <w>word</w> ',
             M2.classify(text, hypotheses) → pick argmax prob_true variant
     words = _process_accent(words, stress_usages):           # stress placement
         skip if '+' already present
         if stress_usages[i]=="STRESS":
             s = accents.get(word.lower())                    # dict lookup first
             if found: copy '+' positions onto original-case word
             elif >1 vowel and no punctuation: word = M1.put_accent(word)   # neural OOV accentor
     join(remaining_text, words) → delete_spaces_before_punc
```

**Resolution order (locked-decision #4 confirmed):** manual `+` wins → **M3 gates** whether a word
is stressed at all → ё (M4 + dicts) → omograph (M2 + dict) → accent (dict, else M1). Output marks
stress with **`+` before the stressed vowel** and restores `ё`.

**Empirically confirmed behaviors** (from `fixtures.json`):
- M3 genuinely gates: `мой тёплый дом → мой тёплый д+ом` — only `дом` is marked; `мой`/`тёплый`
  get NO_STRESS in-sentence (yet isolated `мой → м+ой`). Dropping M3 would **over-stress**.
- Context homographs resolve correctly: `…висит замок → …зам+ок` (lock) vs
  `…старинный замок → …з+амок` (castle).
- ё restoration + stress compose: `ежик нашел гриб → +ёжик наш+ёл гр+иб`.

**Out of scope:** `RuleEngine` (koziev `rupostagger`/`rulemma`, `python-crfsuite`) is **loaded but
never called** in `process_all` (only `rule_accent.load(...)` is referenced; `.accentuate(...)`
is dead). No need to port the CRF tagger. (Verified by grep + neutralized output-neutrally in the
dumper.)

---

## 3. The four neural models (exact ONNX I/O — from `converter/fixtures/model_io.json`)

All inputs `int64`, outputs `float`; shapes `[batch, seq]` → logits. No upstream safetensors —
**weights ship as ONNX only**.

| | Model dir | Arch | Shape | onnx | Inputs | Output (logits) | Labels |
|---|---|---|---|---|---|---|---|
| **M1** | `nn/nn_accent` | `RoFormerForTokenClassification` (rotary) | 4L · 128h · 8heads · ffn 256 · max_pos 60 | 0.8 MB | `input_ids, attention_mask, token_type_ids` | `[B, S, 3]` | `NO / STRESS_PRIMARY / STRESS_SECONDARY` |
| **M2** | `nn/nn_omograph/turbo3.1` | `DebertaForSequenceClassification` (disentangled attn, rel-pos) | 6L · 768h · 12heads · ffn 3072 · max_pos 512 · `type_vocab=0` | **359 MB** | `input_ids, attention_mask` | `[B, 2]` | binary: P(hypothesis true) |
| **M3** | `nn/nn_stress_usage_predictor` | `BertForTokenClassification` (absolute pos) | 3L · 312h · 12heads · ffn 600 · max_pos 2048 | 116 MB | `input_ids, attention_mask, token_type_ids` | `[B, S, 3]` | `NO_STRESS / PUNCT / STRESS` |
| **M4** | `nn/nn_yo_homograph_resolver` | `DistilBertForTokenClassification` | vocab 5031 · max_pos 512 | 14 MB | `input_ids, attention_mask` | `[B, S, 3]` | `NO_YO / PUNCT / YO` |

Per-model ANE trap (carried learnings): M1 rotary→inline static; **M2 disentangled+relative
attention = the dominant risk**; M3 manual-SDPA attn_mask at q_len≫1 + huge embedding (83828×312);
M4 easiest (DistilBERT, no token_type_ids, no pooler).

`token_type_ids` for M1/M3 is **single-segment all-zeros** in this pipeline (the dumper injects
it; numerically identical to the author's older-transformers behavior).

### Tokenizers (4, all need byte-exact Swift ports)
- **M1** `CharTokenizer` — `list(text)`, vocab.txt size 45 (incl. `U+0301`!), `[bos]/[eos]/[pad]/[unk]`, lowercased.
- **M2** `RobertaTokenizer` — byte-level BPE, `vocab.json` (60258) + `merges.txt` + **10 002 added `+word` tokens**, pair input `(text, hypothesis)`.
- **M3** `BertTokenizer` — wordpiece, vocab 83828; uses `return_offsets_mapping` for subword→word `AVERAGE` aggregation (stays in Swift, not the model).
- **M4** `DistilBertTokenizer` — wordpiece, vocab 5031; same offset/aggregation pattern.

### Dictionaries (`dictionary/`)
| file | entries | gz | role |
|---|---|---|---|
| `accents.json.gz` | — | 21 MB | full accent dict (`use_dictionary=True`, our oracle) |
| `accents_nn.json.gz` | 110 826 | 0.85 MB | NN-mode dict (smaller, slightly different outputs) |
| `omographs.json.gz` | 19 741 | 0.22 MB | word → list of `+`-marked variants (+ hardcoded `коса`) |
| `yo_words.json.gz` | 85 568 | 0.55 MB | word → `ё`-word |
| `yo_homographs.json.gz` | 637 | 0.01 MB | context-dependent `ё` forms |

**Decision (resolved 2026-06-19):** ship the **full `accents` dict** (`use_dictionary=True`, 21 MB gz /
175 MB JSON / 3.19 M entries) — exact parity with the fixtures oracle, accepting the larger package
(supersedes the plan's ~11 MB target). WS-F packs a compact on-device form (e.g. sorted keys + a
prefix-trie / FST, or a memory-mapped sorted blob) rather than raw JSON.

---

## 4. Ground-truth artifacts (the numeric oracle)

Generated by `converter/dump_ruaccent_fixtures.py` (committed; outputs partly gitignored):
- `converter/fixtures/fixtures.json` — **42** end-to-end `(input → +marked output)` pairs across
  homograph-context, ё-restoration, OOV-neural, stress-usage, multi-sentence, edge groups.
- `converter/fixtures/model_io.json` — exact ONNX I/O for all four models.
- `converter/fixtures/refs/*.npz` + `manifest.json` — **14** per-model `(tokenized input → logits)`
  oracles (M1×4, M2×4, M3×3, M4×3) for op-level numeric-equivalency checks of each CoreML model.
  Tolerance target: logits max-abs-diff < 1e-3 on Mac CPU vs ONNX, then device-validate.

---

## 5. Parallel workstreams (dependency map)

```
Phase 0 (done) ─┬─ WS-A  M1 accent  (RoFormer)     ─┐
                ├─ WS-B  M2 omograph (DeBERTa, hard)├─ each: rebuild HF module ← ONNX weights
                ├─ WS-C  M3 stress   (BERT)         │        → trace(static) → ct.convert fp16
                ├─ WS-D  M4 yo        (DistilBERT)   ┘        → Mac parity vs refs → DEVICE parity
                ├─ WS-E  Tokenizers ×4 (Swift)  ─ pair each with its model
                ├─ WS-F  Dictionaries pack + Swift loader
                └─ WS-G  Preprocessing/orchestration (normalize, razdel port, word-split, postproc, resolution)
                                         │
   converge →  WS-H  Swift API (RUAccent: RussianStressing) → end-to-end parity vs fixtures.json
                                         │
              → size+latency (palettize) → integrate into chatterbox-coreml
```

WS-A…G are **independent and concurrent**. The four model conversions share one recipe
(rebuild-from-config → static trace → fp16 → parity), differing only in the arch-specific ANE trap,
so they pipeline well. **Recommended first conversion: M1** (smallest, self-contained) to lock the
toolchain, then **M4** (locks the BERT-family recipe), then M3, then **M2** (hardest, last).

## 6. Scope changes vs. the original plan / issues
- Issues #2–#9 assume **two** models. Need conversion issues for **M3 (stress-usage)** and
  **M4 (ё-homograph)**, and explicit tokenizer (#×4) + preprocessing (razdel) tracking.
- Conversion is **ONNX → rebuild-from-config**, not "PyTorch → CoreML" (no upstream PyTorch).
- Dictionary size budget (~11 MB) needs revisiting against the 21 MB-gz full dict.
