# Per-model CoreML conversion playbooks

Distilled from a 6-component fan-out (per-model spec → adversarial verify against the **actual
ONNX graphs, configs, tokenizer.json, and dumped fixtures**). Pairs with `PHASE0_FINDINGS.md`.
Shared recipe for every model: **rebuild HF module from `config.json` → map ONNX initializers
into `state_dict` → fp32-validate vs onnxruntime → harden for fp16 → trace static → `ct.convert`
fp16 → Mac parity → ANE device-validate**. coremltools 9 has no ONNX frontend and there are no
upstream safetensors, so the rebuild-and-load path is mandatory (don't use onnx2torch — ANE-hostile).

Universal fp16 gotcha (carried + reconfirmed in every graph): masked positions are filled with
`finfo(fp32).min` ≈ −3.4e38 → −inf in fp16 → `nan`. Use a **finite −1e4** additive mask inside a
**manually-decomposed SDPA**, and patch the **inner encoder** instance (see
`[[coreml-fp16-mask-gotcha]]`). Convert **fp32 first** to prove the graph, then fp16.

---

## M1 — accent (RoFormer, OOV accentor) ✅ converted (Mac parity)
- **Arch:** `RoFormerForTokenClassification`, 4L/128h/8heads, head_dim 16, ffn 256, exact-erf GELU,
  **rotary on Q,K only** (`rotary_value=false`), char vocab 45, max_pos 60, 3 labels. `relative_attention=true`
  in config is a **no-op** (no rel-bias tensor in `big.onnx`; HF uses rotary exclusively — do NOT add DeBERTa rel-bias).
- **I/O:** `input_ids,[attention_mask=ones],[token_type_ids=zeros]` int32 `[1,S]` → `logits [1,S,3]` (raw).
- **Source:** fp32 `big.onnx` (the int8 `model.onnx` diverges ≤0.49). 25 anonymous `onnx::MatMul_*` Linear
  weights mapped by **bias-adjacency** + transposed.
- **Done:** `converter/convert_accent_coreml.py`. GATE1 torch-vs-ONNX 1.2e-05; GATE2 fp16 argmax-match;
  GATE3 rendered-stress == int8 upstream. fp16 pkg 1.1 MB. **Remaining: ANE device test**; consider
  manual-SDPA + explicit interleaved-rotary rewrite + `RangeDim(3,60)` if ANE rejects the stock trace.

## M4 — yo_homograph (DistilBERT, е→ё) — *recommended 2nd conversion*
- **Arch:** `DistilBertForTokenClassification`, **dim 264** (not 768), hidden 792, 12 heads → head_dim 22,
  3 layers, vocab 5031, max_pos 512, exact-erf GELU, LayerNorm eps **1e-12**, **scale = 1/√22 ≈ 0.2132**,
  3 labels {NO_YO,PUNCT,YO}. No token_type_ids, no pooler.
- **I/O:** `input_ids, attention_mask` int32 `[1,S]` → `logits [1,S,3]` (raw; softmax+AVERAGE word-aggregation in Swift).
- **Weights:** 35 named map 1:1; 19 `onnx::MatMul_*` by bias-adjacency. fp16 ≈ 7 MB. Mask `Where(==0, −3.4e38)`.
- Locks the BERT-family recipe (embeddings + LayerNorm + manual-SDPA + token-cls head) before M3/M2.

## M3 — stress_usage (BERT, *whether to stress*) — gates the pipeline
- **Arch:** `BertForTokenClassification`, 3L/312h/12heads/head_dim 26/ffn 600, vocab **83828**, max_pos 2048,
  type_vocab 2, exact-erf GELU, eps 1e-12, absolute pos, 3 labels {NO_STRESS,PUNCT,STRESS}.
- **I/O:** `input_ids, attention_mask, token_type_ids=zeros` int64→int32 `[1,S]` → `logits [1,S,3]` (raw).
- **Role:** per-word aggregated label; `_process_accent` gates word *i* by **positional** `stress_usages[i]=="STRESS"`.
- **Risk:** 19 **anonymous** `onnx::MatMul_*` incl. classifier → bias-adjacency mapping is load-bearing (strict-by-name fails).
  word_embeddings 83828×312 = **89.9% of params** → fp16 ≈ 58 MB; **palettize the embedding**, skip the head.

## M2 — omograph (homograph disambiguation) — **NOT the hard one after all**
- **Headline (confirmed 3 ways):** `relative_attention=false`, `pos_att_type=null`, zero rel-embedding
  initializers → **DeBERTa disentangled attention is dead code**. It's a **vanilla BERT absolute-position
  encoder**. Only surviving DeBERTa-v1 quirk: **fused in_proj** Linear 768→2304 (no bias) split into Q/K/V,
  with **separate q_bias and v_bias** added after split (**no k_bias**).
- **Arch:** 6L/768h/12heads/ffn 3072, vocab **60258** (50256 base + 10002 `+word`/`<w>`/`</w>` added), max_pos 512,
  num_labels 2, erf-GELU, eps **1e-7**.
- **I/O:** `input_ids, attention_mask` int32 `[1,S]` (NO token_type_ids) → `logits [1,2]` (raw; index 1 = P(hypothesis correct)).
  Pair-encoded `<s> marked_sentence </s></s> hypothesis </s>`, `<w>=60256 </w>=60257`, **pad id = 1** (tokenizer),
  not config's 0. Decision = argmax of logit[1] across a word's K variants (invariant to upstream's global-softmax bug).
- **Size:** fp32 359 MB; word_embeddings 60258×768 = **185 MB (51.6%)** + layers ~170 MB → **palettize 8-bit**
  (skip head); fixed-S functions (S=128, fallback 256, cap 512) merged via `ct.utils`.

## Tokenizers (×4, pure-Swift, byte-exact)
- **All four `tokenizer.json` are present locally** (no M2 blocker). Validate against re-dumped golden refs.
- **M1 char:** lowercase → iterate **unicode scalars** → vocab.txt line index; OOV→`[unk]=1`; `[bos]=2…[eos]=3`.
- **M2 RoBERTa BPE:** GPT-2 byte→unicode map; **added-token longest-match trie** (10002 + `<w>/</w>`) *before* BPE;
  byte-level space `Ġ` (id 225) can precede `<w>` — replicate. Raw case.
- **M3/M4 BERT/DistilBERT WordPiece:** `do_lower_case=false`, `strip_accents=null`, `##` continuation; **must also
  emit `offset_mapping` + `special_tokens_mask`** to the caller for `collect_pre_entities`/`aggregate_words`
  (subword→word **AVERAGE** aggregation, `is_subword` by scalar-length compare). M3 raw sentence; M4 `sentence.lower()`, gated on `'е'∈sentence`.
- **⚠ Oracle gap:** `dump_ruaccent_fixtures.py` currently calls the tokenizers without
  `return_offsets_mapping`/`return_special_tokens_mask`, so the refs lack them — **re-dump** before tokenizer parity work.

## Dictionaries + preprocessing/orchestration (pure-Swift)
- **`accents.json` (full) = 175.5 MB JSON / 20.95 MB gz / 3,194,879 entries** — the ~11 MB plan target is
  unrealistic for full parity → **ship `accents_nn` (110 826 entries, 0.85 MB gz)** unless full-dict parity is required.
  Plus `omographs` (19 741, str→[str]), `yo_words` (85 568), `yo_homographs` (637).
- **`text_split.py` is dead code** (confirmed): `process_all` uses `TextPreprocessor.split_by_sentences` →
  **`razdel.sentenize`** (regex + 11 rules + segment loop) which must be **ported verbatim**, incl. the
  offset-reconstruction in `split_by_sentences`.
- Port `split_by_words` (regex; `' - '`→`' ~ '`; lower-for-match/slice-from-original), `delete_spaces_before_punc`,
  `fix_capital`, `count_vowels`, `has_punctuation`, and `process_all_internal` in the **exact resolution order**
  (manual `+` wins → M3 gate → ё(M4+dicts) → omograph(M2+dict) → accent(dict|M1) → postproc).
- Output is `+`-before-vowel; the package's `RussianStressing` then maps to `U+0301`-after-vowel.
