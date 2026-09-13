# SWIFT_TOKENIZERS.md — Tokenizer Port Spec (RUAccent → RUAccentCoreML)

Self-contained Swift implementation spec for the **three tokenizer families** used by
RUAccent's four neural models. A Swift dev should be able to implement **byte-exact**
tokenizers from this doc alone. Every id/size/string below was read directly from the
artifact files and cross-checked by running the upstream `transformers` tokenizers
(`.venv/bin/python`, transformers 5.12, **fast**
tokenizers). All probe outputs in this doc are reproduced verbatim from those runs.

Artifact roots (gitignored, present locally):

| Model | Family | Dir |
|---|---|---|
| **M1** accent (RoFormer token-clf) | **CharTokenizer** | `converter/_work/nn/nn_accent/` |
| **M2** omograph (DeBERTa-v1 seq-clf) | **ByteLevel-BPE (Roberta)** | `converter/_work/nn/nn_omograph/turbo3.1/` |
| **M3** stress-usage (BERT token-clf) | **WordPiece (Bert)** | `converter/_work/nn/nn_stress_usage_predictor/` |
| **M4** yo-homograph (DistilBERT token-clf) | **WordPiece (DistilBert)** | `converter/_work/nn/nn_yo_homograph_resolver/` |

### ONNX/CoreML model I/O (verified by `onnxruntime`)

| Model | input tensors | output | notes |
|---|---|---|---|
| M1 | `input_ids`, `attention_mask`, `token_type_ids` | `logits [B, S, 3]` | **token_type_ids required**, zero-filled |
| M2 | `input_ids`, `attention_mask` | `logits [B, 2]` | **no token_type_ids** (`type_vocab_size: 0`) |
| M3 | `input_ids`, `attention_mask`, `token_type_ids` | `logits [B, S, 3]` | **token_type_ids required**, zero-filled |
| M4 | `input_ids`, `attention_mask` | `logits [B, S, 3]` | **no token_type_ids** (DistilBERT) |

> **token_type_ids decision (resolves the open question in the brief):** M1 and M3 ONNX
> graphs declare a `token_type_ids` input → the Swift caller MUST supply it, always
> **zero-filled** (single-sequence; type_vocab_size 2 but never uses segment 1). M2 and M4
> ONNX graphs do **not** declare `token_type_ids` → do NOT pass it. This matches the HF
> tokenizer `model_input_names`: M3 = `[input_ids, token_type_ids, attention_mask]`;
> M1/M2/M4 effectively need only `input_ids`/`attention_mask` for type ids (M1 needs zeros
> for the graph). If your CoreML conversion baked token_type_ids into the M1/M3 signature,
> feed an all-zeros Int32 tensor of the same length as `input_ids`.

---

## 1. CharTokenizer (M1 — accent model)

**Source:** `ruaccent-src/ruaccent/char_tokenizer.py`,
`converter/_work/nn/nn_accent/{vocab.txt, tokenizer_config.json, special_tokens_map.json, config.json}`.

This is a tiny custom `PreTrainedTokenizer` subclass (slow/Python tokenizer; not a fast
`tokenizers` model). The whole algorithm is in the source file and is trivial to port.

### 1.1 Vocab — exact, 45 entries (0-based ids = line number − 1)

`config.json`: `"vocab_size": 45`. `vocab.txt` is **45 lines**, one token per line, id = 0-based
line index. **Full table** (reproduce exactly; note hidden/non-ASCII entries):

| id | token | note |
|---|---|---|
| 0 | `[pad]` | pad |
| 1 | `[unk]` | unk |
| 2 | `[bos]` | bos |
| 3 | `[eos]` | eos |
| 4 | `'` | U+0027 apostrophe |
| 5 | `-` | U+002D hyphen-minus |
| 6 | `.` | U+002E |
| 7 | `?` | U+003F |
| 8 | `` ` `` | U+0060 backtick |
| 9 | `c` | **latin** c (U+0063) |
| 10 | `e` | **latin** e (U+0065) |
| 11 | `́` | **U+0301 COMBINING ACUTE ACCENT** (a combining mark, not a letter) |
| 12 | `а` | Cyrillic а (U+0430) |
| 13 | `б` | |
| 14 | `в` | |
| 15 | `г` | |
| 16 | `д` | |
| 17 | `е` | |
| 18 | `ж` | |
| 19 | `з` | |
| 20 | `и` | |
| 21 | `й` | |
| 22 | `к` | |
| 23 | `л` | |
| 24 | `м` | |
| 25 | `н` | |
| 26 | `о` | |
| 27 | `п` | |
| 28 | `р` | |
| 29 | `с` | |
| 30 | `т` | |
| 31 | `у` | |
| 32 | `ф` | |
| 33 | `х` | |
| 34 | `ц` | |
| 35 | `ч` | |
| 36 | `ш` | |
| 37 | `щ` | |
| 38 | `ъ` | |
| 39 | `ы` | |
| 40 | `ь` | |
| 41 | `э` | |
| 42 | `ю` | |
| 43 | `я` | |
| 44 | `ё` | Cyrillic ё (U+0451) |

**Special-token strings + ids** (`special_tokens_map.json` / `tokenizer_config.json`):
`pad_token="[pad]"` id 0, `unk_token="[unk]"` id 1, `bos_token="[bos]"` id 2,
`eos_token="[eos]"` id 3. **All lowercase in brackets** — note these are NOT the BERT-style
`[PAD]`/`[CLS]`; they are different strings. `do_lower_case: true`.

### 1.2 Algorithm (exact)

Input is the **lowercased single word** supplied by `AccentModel.put_accent`:
`lower_word = word.lower()` then `tokenizer(lower_word)`. Inside the tokenizer
(`do_lower_case=true`):

1. `_tokenize(text)`: `text = text.lower()`; return `list(text)` — i.e. split into **Unicode
   scalars / characters** (Python `list(str)` = list of code points). One token per character.
   - **Swift note:** iterate `text.unicodeScalars` (Python `list(str)` enumerates code points,
     NOT grapheme clusters). Combining marks like U+0301 are their own element.
2. `_convert_token_to_id(token)`: lowercase the token again (no-op here), then
   `vocab.get(token, vocab["[unk]"])`. **Unknown char → id 1 (`[unk]`).**
3. `build_inputs_with_special_tokens`: `[bos_id] + ids + [eos_id]` = `[2] + ids + [3]`.
4. `attention_mask` = all 1s, length = len(input_ids). `token_type_ids` = all 0s
   (`create_token_type_ids_from_sequences` → `(len+2)*[0]`).

No padding is applied for inference (single word, batch size 1). No `max_length` truncation
in practice (`model_max_length` is effectively infinite; `config.max_position_embeddings`=60,
but words are short).

### 1.3 Verified examples (probe output)

```
'замок'  -> chars ['з','а','м','о','к']                ids [2, 19, 12, 24, 26, 22, 3]
'ЗАмок'  -> chars ['з','а','м','о','к'] (lowercased)   ids [2, 19, 12, 24, 26, 22, 3]
'кофе'   -> chars ['к','о','ф','е']                    ids [2, 22, 26, 32, 17, 3]
'test'   -> chars ['t','e','s','t']                    ids [2, 1, 10, 1, 1, 3]   # t,s -> [unk]=1; e -> 10
"пример'"-> chars ['п','р','и','м','е','р',"'"]        ids [2, 27, 28, 20, 24, 17, 28, 4, 3]
```
(`t` and `s` are latin and **not** in vocab → `[unk]`=1; only latin `c`,`e` are present.)

### 1.4 Downstream consumption (tokenizer-relevant only)

`AccentModel.put_accent(word)` (`accent_model.py`): runs the model on the lowercased word;
`logits` shape `[1, S, 3]` where `S == len(input_ids)` (verified: `замок` → S=7 =
`[bos] з а м о к [eos]`). Per position: `score = max(softmax(logits))`, `label =
argmax(logits)` mapped via `id2label` = `{0:"NO", 1:"STRESS_PRIMARY", 2:"STRESS_SECONDARY"}`.
`render_stress` walks positions `i = 0..S-1` over the **original** (non-lowercased) `word`
characters; when `label != "NO"` and `label != "STRESS_SECONDARY"` and `score >= 0.55`, it
inserts `"+"` **before** `word[i-1]` (the previous character; position 0 = `[bos]`, so the
first real char maps to model index 1, and `word[index-1]` is that char). Net effect: a `+`
is placed **before** the stressed vowel. Marking the stressed vowel/`+` placement is
AccentModel post-processing (Agent C/D territory); included here only to fix the index
alignment: **model logit index `i` aligns to `word[i-1]`** because of the `[bos]` prefix.

---

## 2. WordPiece — Bert (M3) & DistilBert (M4)

**Sources:** `stress_usage_model.py`, `yo_homograph_model.py`; per-model
`{vocab.txt, tokenizer.json, tokenizer_config.json, special_tokens_map.json, config.json}`.

Both are **fast** HF tokenizers (`is_fast: True`, verified). The byte-exact reference is the
`tokenizers` Rust pipeline encoded in each `tokenizer.json`. M3 and M4 share **identical**
normalizer / pre-tokenizer / post-processor / WordPiece settings; they differ only in vocab
and in whether `token_type_ids` is emitted (M3 yes, M4 no).

### 2.1 Special tokens (read from each model's files)

Identical strings & ids for both M3 and M4 (verified from `vocab.txt` line order and
`tokenizer.json.added_tokens`):

| token | id | role |
|---|---|---|
| `[PAD]` | 0 | pad |
| `[UNK]` | 1 | unk |
| `[CLS]` | 2 | cls (start) |
| `[SEP]` | 3 | sep (end) |
| `[MASK]` | 4 | mask (unused at inference) |

`tokenizer_config.json` (both): `do_lower_case: false`, `do_basic_tokenize: true`,
`strip_accents: null`, `tokenize_chinese_chars: true`, `never_split: null`.

Vocab sizes (verified): **M3 = 83828** (`vocab.txt` 83828 lines, `config.vocab_size 83828`);
**M4 = 5031** (`vocab.txt` 5031 lines, `config.vocab_size 5031`). In both, `vocab.txt` line
index (0-based) **is** the token id.

### 2.2 `tokenizer.json` structure (identical for M3 & M4, verified)

```jsonc
normalizer:    { "type":"BertNormalizer", "clean_text":true, "handle_chinese_chars":true,
                 "strip_accents":null, "lowercase":false }
pre_tokenizer: { "type":"BertPreTokenizer" }
model:         { "type":"WordPiece", "unk_token":"[UNK]",
                 "continuing_subword_prefix":"##", "max_input_chars_per_word":100 }
post_processor:{ "type":"TemplateProcessing",
                 single: [CLS]:0  A:0  [SEP]:0
                 pair:   [CLS]:0  A:0  [SEP]:0  B:1  [SEP]:1
                 special_tokens: [CLS]=ids[2], [SEP]=ids[3] }
decoder:       { "type":"WordPiece", "prefix":"##", "cleanup":true }
```
Only single-sequence is used downstream (a full sentence). `pair` exists but is never
exercised by M3/M4.

### 2.3 Pipeline step 1 — `BertNormalizer` (`clean_text=true`, `handle_chinese_chars=true`, `lowercase=false`, `strip_accents=null`)

Applied to the raw input string (M4 receives an already-lowercased sentence —
`_process_yo` calls `sentence.lower()`; M3 receives the sentence as-is, casing preserved).
`BertNormalizer` does, in order, on the code-point stream:

1. **clean_text:** for each char `c`: drop `c` if it is char `0x0` (null), `0xFFFD`
   (replacement), or a **control** char (Unicode category starts with `C`, **except** that
   `\t`, `\n`, `\r` are treated as whitespace, not dropped). Replace any **whitespace** char
   (`\t`, `\n`, `\r`, or Unicode `White_Space`/category `Zs`) with a single ASCII space
   `' '` (0x20). No collapsing of runs here — each whitespace char → one space.
2. **handle_chinese_chars (`tokenize_chinese_chars`):** wrap every CJK ideograph code point
   with surrounding spaces (`' ' + c + ' '`). The CJK ranges are the standard BERT set:
   `[0x4E00–0x9FFF]`, `[0x3400–0x4DBF]`, `[0x20000–0x2A6DF]`, `[0x2A700–0x2B73F]`,
   `[0x2B740–0x2B81F]`, `[0x2B820–0x2CEAF]`, `[0xF900–0xFAFF]`, `[0x2F800–0x2FA1F]`.
   **Russian (Cyrillic U+0400–U+04FF), latin, and digits are NOT in these ranges → untouched.**
   For RUAccent's Russian-only inputs this step is effectively a no-op, but implement it for
   parity safety.
3. **strip_accents = null:** with `BertNormalizer`, `strip_accents` defaults to **the value
   of `lowercase`**, which is `false` here → **accents are NOT stripped.** Combining marks
   and precomposed accented letters pass through unchanged (verified: `café` stays `café`,
   `ё` stays `ё`). Do **not** NFD-decompose or remove `Mn` marks.
4. **lowercase = false:** **no lowercasing.** Casing is preserved (verified: `ЗАМОК`,
   `Замок`, `замок` tokenize differently).

### 2.4 Pipeline step 2 — `BertPreTokenizer` (= BasicTokenizer splitting + offset tracking)

Operates on the normalized string and produces pre-token spans (each carries a (start,end)
**char offset** into the normalized string, which for these inputs equals an offset into the
original because normalization is length-preserving for Cyrillic/latin/digits). Rules:

1. **Whitespace split:** split on runs of ASCII space (the only whitespace left after
   clean_text). Spaces are consumed (not emitted as tokens).
2. **Punctuation split:** within each whitespace-delimited chunk, split off **every
   punctuation char as its own pre-token**. "Punctuation" = a char that is **either** ASCII
   punctuation in the ranges `[33–47]` (`!"#$%&'()*+,-./`), `[58–64]` (`:;<=>?@`),
   `[91–96]` (`` [\]^_` ``), `[123–126]` (`{|}~`), **or** any char whose Unicode general
   category starts with `P` (Pc, Pd, Ps, Pe, Pi, Pf, Po). Each such char becomes a standalone
   pre-token; the runs between them are pre-tokens too.
3. **CJK:** already space-wrapped by the normalizer, so each CJK char is its own pre-token.
4. **Digits and letters are NOT separated from each other** by the pre-tokenizer — only
   whitespace and punctuation split. So `123` (whitespace-isolated) is one pre-token, and
   `слово123буква` is a **single** pre-token (digits+letters glued) that WordPiece later
   splits internally.

**Exact effect on Cyrillic / digits / latin (verified with M3):**
```
'Привет, мир!'    -> ['Привет', ',', 'мир', '!']          offs (0,6)(6,7)(8,11)(11,12)
'дом-музей'       -> ['дом', '-', 'музей']                 # '-' is punctuation
'число 123 и 4.5' -> ['число','123','и','4','.','5']       # '.' splits 4 and 5
'café naïve'      -> ['café','naïve']                       # accents kept (naïve later -> [UNK])
'ёлка'            -> ['ёлка']                                # ё kept
'ЗАМОК Замок замок'-> ['ЗАМОК','Замок','замок']             # casing preserved
'слово123буква'   -> ['слово123буква']                      # one pre-token (no digit/letter split)
```

### 2.5 Pipeline step 3 — `WordpieceTokenizer` (greedy longest-match, `##`, `max_input_chars_per_word=100`)

For each pre-token string `w` (as a sequence of Unicode chars), against the model vocab:

1. If `len(chars(w)) > max_input_chars_per_word` (**100**) → emit a single `[UNK]` for the
   whole pre-token; stop.
2. Greedy **longest-match-first**, left to right:
   - `start = 0`. While `start < len(chars)`: set `end = len(chars)`; try substring
     `chars[start:end]`. For `start > 0`, prefix the candidate with `"##"` before vocab
     lookup. Shrink `end` by 1 until the (prefixed) candidate is in the vocab or `end == start`.
   - If no substring matched (`end == start`) → the **entire word** is unknown: discard any
     pieces collected and emit a single `[UNK]`; stop.
   - Else append the matched piece's id, set `start = end`, continue.
   - The **first** piece of a word has no prefix; **continuation** pieces are stored with the
     literal `"##"` prefix (e.g. `##лка`), and that `##` is part of the token string in the
     vocab. `continuing_subword_prefix` = `"##"`.
3. `unk_token` = `[UNK]` (id 1).

**Verified subword examples (M4, lowercased inputs):**
```
'ёжик'        -> ['ё','##жи','##к']
'съел'        -> ['съ','##е','##л']
'абрикос'     -> ['аб','##рик','##о','##с']
'непонятное'  -> ['непонят','##ное']
'словцо'      -> ['слов','##цо']
'абракадабра' -> ['аб','##рак','##а','##да','##бра']
```
**Verified (M3):** `'ёлка' -> ['ё','##лка']`, `'ЗАМОК' -> ['ЗА','##МО','##К']`,
`'слово123буква' -> ['слово','##12','##3','##бук','##ва']`.

### 2.6 Pipeline step 4 — post-processor + tensors

Single sequence layout: `[CLS] <wordpieces...> [SEP]` → ids `[2] + ids + [3]`.

- **`attention_mask`**: all 1s, length = len(input_ids) (no padding at inference; one sentence
  per call, batch size 1).
- **`token_type_ids`** (`type_id`): all 0 (single sequence). **Emit & feed for M3** (ONNX
  requires it); **do NOT emit for M4** (ONNX has no such input).
- **`special_tokens_mask`**: 1 at `[CLS]`/`[SEP]` positions, 0 elsewhere.
- **`offset_mapping`**: per token a `(start, end)` char span into the **original sentence**.
  Special tokens (`[CLS]`,`[SEP]`) get `(0, 0)`. For real tokens the fast tokenizer carries
  the byte/char span from the pre-tokenizer through WordPiece: the **first** subword of a
  pre-token starts at the pre-token's start; each subsequent subword's start = previous
  subword's end; the `##` prefix is **not** counted in the offset (offsets index the original
  text, which has no `##`). End of the last subword = pre-token end. (Verified e.g. `ёлка` →
  `ё`=(0,1), `##лка`=(1,4); `двери`=(9,14); `.`=(14,15).)

**Verified M3 full encode** `"Замок на двери."`:
```
ids     [2,     55574,  548,  31629,  18,   3]
tokens  [CLS]   Замок   на    двери   .     [SEP]
offsets (0,0)  (0,5)  (6,8)  (9,14)(14,15) (0,0)
stm     1       0      0      0       0     1
attn    1       1      1      1       1     1
type    0       0      0      0       0     0   (M3 only; M4 omits)
```

### 2.7 Subword aggregation used downstream (CONFIRMED branch + algorithm)

This is the load-bearing parity logic in `stress_usage_model.py` /
`yo_homograph_model.py` (`collect_pre_entities` → `aggregate_words` → `aggregate_word`).
The model produces per-token softmax score vectors; aggregation collapses subwords back to
words and assigns one label per word.

**Branch CONFIRMED (was an open question in the brief):** these are **fast WordPiece**
tokenizers and `continuing_subword_prefix` **is** `"##"` (non-empty) for both M3 and M4
(verified: `tok._tokenizer.model.continuing_subword_prefix == "##"`). Therefore in
`collect_pre_entities` the code takes the **first** branch:

```python
if getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None):
    is_subword = len(word) != len(word_ref)      # <-- THIS branch runs for M3 & M4
else:
    is_subword = sentence[start-1:start] != " " if start > 0 else False   # NOT taken
```
where `word = convert_ids_to_tokens(id)` (the token string, **including** any `##`) and
`word_ref = sentence[start:end]` (the original text slice from `offset_mapping`).
So **`is_subword == True` iff the token string is longer than its source slice**, which is
exactly "the token carries a `##` prefix" (the `##` adds 2 chars; the slice has no `##`).

> Concretely: a continuation token like `##ное` has `len("##ное")=5` vs `len("ное")=3` →
> subword. A leading token like `непонят` has `len==len` → not subword. (Verified.)

**UNK special case:** if `input_ids[idx] == unk_token_id` (1), then `word = word_ref`
(replace the `[UNK]` string with the actual source text) and `is_subword = False`. So an
`[UNK]` always **starts** a new word group.

`special_tokens_mask[idx]` truthy → token is **skipped** entirely (CLS/SEP never enter
pre_entities).

**`aggregate_words` (grouping):** iterate pre_entities in order. Start a new `word_group`
with the first entity. For each next entity: if `is_subword` → append to the current group;
else → **flush** the current group via `aggregate_word`, then start a new group with this
entity. Flush the last group at the end. Net: **one leading (non-subword) token + its
following subword tokens = one word.**

**`aggregate_word` with strategy `"AVERAGE"` (the only strategy used; both models call
`aggregate_words(..., "AVERAGE")`):**
1. `scores = np.stack([e["scores"] for e in group])` → shape `[n_subwords, n_labels]`
   (n_labels = 3).
2. `average_scores = np.nanmean(scores, axis=0)` → **element-wise mean of the per-token
   softmax probability vectors** (NaN-safe; scores here are finite so it's a plain mean).
3. `entity_idx = argmax(average_scores)`; `entity = id2label[str(entity_idx)]`;
   `score = average_scores[entity_idx]`.
4. The returned `word` = `convert_tokens_to_string([e["word"] for e in group])` (joins
   subwords, stripping `##` and inserting spaces per WordPiece decoder — used only for
   debugging/labels; **the consumer only reads `entity`**).

> **Important:** the **softmax is per-token** (computed in `predict_*` before aggregation:
> `scores = softmax(logits, axis=-1)` per position), and AVERAGE means the **mean of those
> softmax vectors**, then argmax. Do NOT average logits.

**Downstream use of the result:** `ruaccent.py` calls
`extract_entities(predict_*())` → list of just the `"entity"` strings, **one per word, in
sentence order**, then indexes it positionally against the `words` list
(`stress_usages[i]`, `yo_predictions[i]`). **Parity requirement:** the number and order of
aggregated word-entities must line up 1:1 with `TextPreprocessor.split_by_words` output.
(Word-splitting itself is a separate agent's spec; this tokenizer doc guarantees the entity
list ordering matches token order.)

### 2.8 `id2label` maps (exact, from each `config.json`)

- **M3** (`nn_stress_usage_predictor/config.json`):
  `{"0":"NO_STRESS", "1":"PUNCT", "2":"STRESS"}` (label2id is the inverse).
  Downstream: a word with entity `"STRESS"` is eligible for accenting; `"NO_STRESS"`/`"PUNCT"`
  are not. (In `_process_accent`, `if stress_usages[i] == "STRESS": ...`.)
- **M4** (`nn_yo_homograph_resolver/config.json`):
  `{"0":"NO_YO", "1":"PUNCT", "2":"YO"}`.
  Downstream (`_process_yo`): a word with entity `"YO"` triggers a `yo_homographs` dict lookup.

### 2.9 M3 vs M4 differences summary

| | M3 (stress-usage / Bert) | M4 (yo / DistilBert) |
|---|---|---|
| vocab size | 83828 | 5031 |
| input casing | sentence as-is (not lowercased) | sentence **lowercased** by caller (`sentence.lower()`) |
| `token_type_ids` | **emit + feed** (ONNX needs it) | **omit** (ONNX has none) |
| id2label | NO_STRESS / PUNCT / STRESS | NO_YO / PUNCT / YO |
| everything else | identical normalizer/pre-tok/WordPiece/post-proc | identical |

---

## 3. ByteLevel-BPE — Roberta tokenizer (M2 — omograph)

**Sources:** `omograph_model.py`; `nn_omograph/turbo3.1/{tokenizer.json,
tokenizer_config.json, special_tokens_map.json, vocab.json, merges.txt, added_tokens.json,
config.json}`.

Fast tokenizer (`is_fast: True`). Model architecture is **DeBERTa-v1**
(`DebertaForSequenceClassification`, `type_vocab_size: 0`), but `tokenizer_class` is
**`RobertaTokenizer`** — a GPT-2-style **ByteLevel BPE** with Roberta special tokens and a
`RobertaProcessing` post-processor. Sequence-classification, output `logits [B, 2]`.

### 3.1 Special tokens + ids (verified)

| token | id | role |
|---|---|---|
| `<s>` | 0 | bos **and** cls |
| `<pad>` | 1 | pad |
| `</s>` | 2 | eos **and** sep |
| `<unk>` | 3 | unk |
| `<mask>` | 4 | mask (unused at inference; `lstrip:true`) |

(`tokenizer_config.json`: `bos=cls=<s>`, `eos=sep=</s>`, `pad=<pad>`, `unk=<unk>`,
`add_prefix_space:false`, `trim_offsets:true`.) These five are ids 0–4 in **both** the base
`vocab.json` and the `tokenizer.json.added_tokens` (as `special:true`).

### 3.2 Vocab layout (verified counts)

- **Base BPE vocab** (`vocab.json`, also `tokenizer.json.model.vocab`): **50256** entries,
  ids `0..50255`. Keys are byte-level-encoded strings (GPT-2 unicode mapping, §3.4).
- **Merges** (`merges.txt`): file has `#version: 0.2` header on line 1 then **49995** merge
  rules (space-separated pairs); `tokenizer.json.model.merges` holds the same **49995** rules
  (no header). First real merge: `ĥ Ð`.
- **Added tokens** (`added_tokens.json`, also the non-special entries of
  `tokenizer.json.added_tokens`): **10002** non-special added tokens, ids `50256..60257`,
  **plus** the 5 special tokens → `tokenizer.json.added_tokens` has **10007** entries total.
- **Total vocab_size = 50256 + 10002 + (5 specials already counted in 0..4) = 60258**
  (matches `config.vocab_size: 60258`). The 5 specials occupy ids 0–4 inside the base 50256;
  the 10002 extra added tokens occupy 50256–60257.

**What the 10002 added tokens are:** **stress-marked Russian word forms** — e.g.
`п+осле`(50256), `уж+е`(50257), `был+а`(50258), `к+оса`(53101), `+августовские`(58646),
`ясн+ы`(...) — i.e. the omograph **hypothesis variants** are present as **single atomic
tokens**, with the `+` stress mark embedded. They have `normalized:true, special:false`.

### 3.3 `<w>` / `</w>` markers — DEDICATED TOKENS, not byte-BPE'd (parity-critical, CONFIRMED)

`_process_omographs` wraps the target word in the sentence:
`t[position] = ' <w>' + t[position] + '</w> '` and the joined sentence (with literal
`<w>...</w>`) is the **first** sequence (A) to the tokenizer.

- `<w>` is an **added token** id **60256**; `</w>` is **id 60257** (`normalized:true,
  special:false`). They are **NOT** in the base BPE `vocab.json`. The fast tokenizer extracts
  added tokens **before** byte-level pre-tokenization, so they always encode as exactly one id.
- **Verified:** `tok("<w>") -> [60256]`, `tok("</w>") -> [60257]`,
  `tok(" <w>коса</w> ") -> [225, 60256, 296, 6907, 60257, 225]`
  (`225`=`Ġ` space token; `296 6907` = byte-BPE of `коса`).
- **Swift implication:** maintain an added-token table; greedily match `<w>`/`</w>` (and any
  hypothesis variant that exists as an added token, e.g. `к+оса`→53101) as whole tokens before
  running byte-level BPE on the residual text. Do **not** byte-BPE the literal `<` `w` `>`
  characters.

### 3.4 Byte→unicode mapping (GPT-2 `bytes_to_unicode`)

`pre_tokenizer: {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true,
"use_regex":true}`. The base vocab keys are produced by the standard GPT-2 byte→unicode
table. Build it once (256 entries):

```
visible bytes B = [0x21..0x7E] ∪ [0xA1..0xAC] ∪ [0xAE..0xFF]     # 188 printable bytes
for b in B:                    u[b] = chr(b)
n = 0
for b in 0..255 not in B:      u[b] = chr(256 + n); n += 1        # remaining 68 bytes
```
This yields e.g. byte `0x20`(space)→`'Ġ'`(U+0120), `0xD0`→`'Ð'`, `0xBA`→`'º'`, `0x0A`→`'Ċ'`.
**Verified:** UTF-8 of `коса` = bytes `D0 BA D0 BE D1 81 D0 B0` maps to `'ÐºÐ¾ÑģÐ°'`, and
`296`+`6907` decode (via inverse table) back to `коса`. Encoding direction (Swift):
1. UTF-8-encode the text piece into bytes.
2. Map each byte through the table → a string of "visible" unicode chars.
3. Run BPE over that char string against `vocab.json`/`merges.txt`.

### 3.5 Pre-tokenization regex (GPT-2 / ByteLevel `use_regex:true`)

Before byte-mapping, ByteLevel splits each input chunk with the GPT-2 regex (requires a
PCRE/ICU engine with Unicode properties — `\p{L}`, `\p{N}`):

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
Each match is a pre-token; a **leading space is kept as part of the match** and (after byte
mapping) becomes a `Ġ` prefix on the first BPE symbol (this is how word boundaries are
encoded — there is no `##`). `add_prefix_space:false` means the **very first** token of the
string does **not** get an artificial leading space (only spaces actually present in the text
become `Ġ`). **Verified pieces** of `" <w>коса</w> у воды 123"` (ignoring added-token
extraction): `[' <', 'w', '>', 'коса', '</', 'w', '>', ' у', ' воды', ' 123']`.

> Order of operations in the fast tokenizer: **(1)** split out added/special tokens
> (`<w>`,`</w>`,`<s>`,`</s>`, and any hypothesis variant present in added set) → **(2)** for
> each residual text span, apply the GPT-2 regex → **(3)** byte-map each piece → **(4)** BPE.

### 3.6 BPE merge algorithm

Standard GPT-2/Roberta BPE on the byte-mapped char string of each regex piece:

1. Start with the sequence of single byte-mapped chars (each is a symbol).
2. Build the rank table from `merges.txt` order: rank(`A B`) = its 0-based line index among
   the **49995** rules (ignore the `#version` header). Lower rank = higher priority.
3. Repeatedly find the adjacent symbol pair `(a,b)` with the **lowest** rank that exists in
   the table; merge it into `a+b`; repeat until no adjacent pair is in the table.
4. Map each final symbol to its id via `vocab.json` (`continuing_subword_prefix:""`,
   `end_of_word_suffix:""`, `fuse_unk:false`, `byte_fallback:false`, `unk_token:null`).
   `unk_token` is `null`, so every byte maps to *some* symbol in the base vocab (all 256 byte
   chars are present as singletons) — there is effectively **no `<unk>` path for normal text**.

### 3.7 Post-processor (`RobertaProcessing`) — single & pair layout (verified)

```jsonc
post_processor: { "type":"RobertaProcessing",
                  "sep":["</s>", 2], "cls":["<s>", 0],
                  "trim_offsets":true, "add_prefix_space":false }
```
- **Single sequence (A):** `<s> A </s>` → `[0] + A + [2]`.
- **Pair (A, B):** `<s> A </s> </s> B </s>` → `[0] + A + [2, 2] + B + [2]`. (Roberta inserts a
  **double `</s>`** between A and B; there is no segment-1 type id — `token_type_ids` is not
  produced/needed.) cls/bos = `<s>` = **0**; sep/eos = `</s>` = **2**.

**Tensors:** `input_ids` per above; `attention_mask` = all 1s for a single example; with
batch `padding=True` (even path, §3.9), pad to the batch max with `<pad>`=1 and set those
mask positions to 0. **No `token_type_ids`** is emitted or fed (verified: `model_input_names
= ['input_ids', 'attention_mask']`; ONNX has only those two inputs).

**Verified pair encode** (A = `"...к+оса <w>коса</w> у воды"`, B = `"к+оса"`):
```
ids: [0, 590, 11294, 225, 53686, 15829, 718, 225, 53101, 225, 60256, 296, 6907, 60257, 494, 3967, 2, 2, 53101, 2]
tok: <s> Ð£ ĠÐ±ÐµÑĢÐµÐ³Ð° Ġ росл+а ... Ġ к+оса Ġ <w> Ðº Ð¾ÑģÐ° </w> ĠÑĥ ĠÐ²Ð¾Ð´Ñĭ </s> </s> к+оса </s>
                                                       ^60256 ^split bytes ^60257            ^^ double </s>  ^53101 added-token hypothesis
```
Note `<w>`=60256, `</w>`=60257, and the hypothesis `к+оса`=53101 each encode as one token;
the unmarked `коса` inside `<w>...</w>` byte-BPEs to `Ðº`(296)+`Ð¾ÑģÐ°`(6907).

### 3.8 What the model sees — sequence A and sequence B

From `_process_omographs` + `OmographModel.classify`:
- **Sequence A** (`text`/`t[0]` in the code, also called `preprocessed_texts`): the full
  sentence, with the target word wrapped `' <w>'+word+'</w> '`, **space-before-punctuation
  removed** twice: once by `delete_spaces_before_punc` when building `texts_batch`, and again
  inside `classify` via `re.sub(r'\s+(?=(?:[,.?!:;…]))', '', text)` (strip whitespace
  immediately before `, . ? ! : ; …`).
- **Sequence B** (`hypotheses`/`hp`): a **single stress-marked variant string**, e.g.
  `к+оса` or `кос+а` (these usually exist as atomic added tokens).
- One model call scores one `(A, B)` pair → `logits[2]` = `[p_false_logit, p_true_logit]`.

### 3.9 `classify()` selection logic (reimplement the SELECTION, verified)

`classify(texts, hypotheses, num_hypotheses)` — `texts` are A-strings (one repeated per
variant of a given omograph), `hypotheses` is the **flattened** list of all variant strings
across all found omographs, `num_hypotheses` = per-omograph variant counts.

```python
preprocessed_texts = [re.sub(r'\s+(?=(?:[,.?!:;…]))', '', t) for t in texts]   # strip space-before-punct
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))   # NOTE: GLOBAL over the whole array
```

**EVEN/batch path** — taken iff **every** omograph has an **even** variant count
(`all(n % 2 == 0 for n in num_hypotheses)`):
1. Tokenize the whole batch as pairs: `tokenizer(preprocessed_texts, hypotheses,
   padding=True, truncation=True, max_length=512)` → batch of `(A_i, B_i)` pairs.
2. `outputs = model(...)[0]` shape `[N, 2]`; `outputs = softmax(outputs)` (global).
3. `prob_label_is_true[i] = outputs[i][1]` (the **second** logit/prob = "this variant is the
   correct stress").
4. Re-pair consecutive entries: hypotheses and probs are grouped in **2s**
   `(0,1),(2,3),...`; for each pair pick the hypothesis with the larger prob:
   `outs.append(pair[argmax(prob_pair)])`.
   - This path **only handles variant-count == 2** correctly (it hard-pairs by 2). Omographs
     with 2 variants are the common case (e.g. `коса` → `["к+оса","кос+а"]`).

**ODD/no-batch path** — taken iff **any** omograph has an **odd** variant count:
1. `grouped_h = group_words(hypotheses)`: re-chunk the flat hypothesis list back into
   per-word groups using `replace('+','')` equality to detect word boundaries, then:
   - if base word ∈ `special_words` (a hardcoded list, see §3.10) **and** group size > 3 →
     split into chunks of **3**;
   - elif group size > 3 **and** even → split into chunks of **2**;
   - else keep the whole group.
2. `grouped_t = transfer_grouping(grouped_h, preprocessed_texts)`: re-chunk the A-strings to
   the **same shape** as `grouped_h` (positional, by group lengths).
3. For each `(h_group, t_group)`: for each variant `hp` in `h_group`, run **one pair**
   `tokenizer(t_group[0], hp, max_length=512, truncation=True)` → `model` → `softmax` (global
   over the 2-vector) → `prob_label_is_true = out[0][1]`. Pick `h_group[argmax(probs)]`.

**Output:** `classify` returns one chosen stress-marked string per found omograph, assigned
back: `splitted_text[omograph.position] = cls_batch[cls_index]`.

> **SELECTION SIMPLIFICATION (verified, safe for Swift):** the upstream `softmax` subtracts a
> **global** max and divides by a **global** sum — both constants across the array. Therefore
> comparing `out[i][1]` across the variants of one omograph is **monotonically equivalent** to
> comparing the **raw true-class logit** `logits[i][1]`. A Swift port may either replicate the
> exact global softmax, or simply **pick the variant with the maximum `logits[h][1]`** (the
> second output value), per omograph — the argmax is identical (verified numerically). You do
> **not** need to reproduce the even/odd batching to get correct selection; batching is purely
> a performance grouping. The semantically required behavior is: **for each omograph, score
> each of its variants as a separate `(A_with_<w>marker, variant)` pair and choose the variant
> with the largest `out[1]`.** (The odd-path `group_words` 3-chunking for `special_words` and
> 2-chunking only change *batching*, not which variant wins, since each variant is still scored
> independently within its sub-batch and the per-variant `out[1]` values are compared.)

### 3.10 `special_words` list (for exact odd-path batching parity, if reproducing batching)

Hardcoded in `OmographModel.__init__` (`omograph_model.py`). Only relevant if you reproduce
the **batching** (not needed for selection correctness, per §3.9). The 80-entry list:

```
балчуга, вертела, волоки, волоку, воронью, выбродите, вывозите, выносите, выноситесь,
выходите, железы, начала, округа, перепела, развитая, развитого, развитое, развитой,
развитом, развитому, развитою, развитую, развитые, развитым, развитыми, развитых, сторожа,
сторожи, сторожу, удало, начался, началась, началось, бутиках, ожила, создало, коротки,
проклята, роженица, роженицы, рожениц, роженице, роженицам, роженицу, роженицей, роженицею,
роженицами, роженицах, пристава, приставов, приставам, приставами, приставах, пережитое,
пережитого, пережитые, пережитых, пережитому, пережитым, пережитыми, пережитом, нипоняла
```

---

## 4. Cross-model id/size quick-reference (all verified from files)

| | M1 char | M2 byte-BPE | M3 wordpiece | M4 wordpiece |
|---|---|---|---|---|
| tokenizer class | CharTokenizer | RobertaTokenizer | BertTokenizer | DistilBertTokenizer |
| vocab size | 45 | 60258 (50256 base + 10002 added) | 83828 | 5031 |
| pad | `[pad]`=0 | `<pad>`=1 | `[PAD]`=0 | `[PAD]`=0 |
| unk | `[unk]`=1 | `<unk>`=3 | `[UNK]`=1 | `[UNK]`=1 |
| start | `[bos]`=2 | `<s>`=0 | `[CLS]`=2 | `[CLS]`=2 |
| end | `[eos]`=3 | `</s>`=2 | `[SEP]`=3 | `[SEP]`=3 |
| mask | — | `<mask>`=4 | `[MASK]`=4 | `[MASK]`=4 |
| lowercase input | yes (in tokenizer) | no | no | yes (by caller) |
| token_type_ids fed | yes (zeros) | **no** | yes (zeros) | **no** |
| logits shape | [B,S,3] | [B,2] | [B,S,3] | [B,S,3] |
| id2label | NO/STRESS_PRIMARY/STRESS_SECONDARY | (binary, no labels) | NO_STRESS/PUNCT/STRESS | NO_YO/PUNCT/YO |

---

## 5. Open questions / caveats for the Swift author

1. **Fast-tokenizer offset semantics under normalization.** For RUAccent inputs (Cyrillic +
   ASCII) `BertNormalizer` is length-preserving, so `offset_mapping` indexes the original
   sentence and the `is_subword = len(word) != len(word_ref)` test holds. If the upstream
   `normalize_regex` ever leaves a normalizer-altering char (e.g. a full-width space) the
   offsets index the **normalized** string; the brief's pipeline applies `re.sub(normalize,...)`
   *before* tokenization, which strips most such chars. **Recommended:** drive `is_subword`
   directly off the `##` prefix (token string starts with `##`) rather than offset-length
   arithmetic — it is equivalent for `##`-prefixed WordPiece and avoids any offset edge cases.
   (This matches `len(word) != len(word_ref)` exactly whenever offsets are length-preserving.)
2. **`convert_tokens_to_string` for the aggregated `word`.** Only used as a debug label; the
   consumer reads `entity` only. A Swift port can skip building the joined word string.
3. **M2 even-path correctness for >2 variants.** The upstream EVEN branch hard-pairs by 2,
   so it is only correct when every omograph has exactly 2 variants. If a custom dict adds an
   omograph with 4+ even variants, upstream's even-path pairing is arguably buggy. The §3.9
   per-variant-argmax reimplementation sidesteps this and is strictly more correct; confirm
   with the parity oracle whether to mirror the upstream quirk or fix it. (RUAccent's shipped
   `omographs.json` variants are predominantly 2-way.)
4. **`<mask>` / `[MASK]` and `<pad>` never appear at inference** for these flows; included for
   completeness.
5. **Tokenizer `truncation`/`padding` in tokenizer.json are `null`** for M2/M3/M4 — truncation
   to 512 is applied imperatively in the Python callers (`max_length=512`), not by the saved
   tokenizer config. Sentences in RUAccent are short; truncation rarely triggers, but enforce
   `max_length=512` (M2) / `model_max_length=2048` (M3) / large (M4) to match.
6. **M2 base vocab byte-fallback:** `unk_token` is `null` and all 256 byte symbols exist as
   single-char vocab entries, so arbitrary UTF-8 always encodes without `<unk>`. No special
   unknown handling is needed for sequence A/B text.
