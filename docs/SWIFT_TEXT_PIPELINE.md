# SWIFT_TEXT_PIPELINE — RUAccent non-neural text handling, port spec

Authoritative Swift port spec for **all non-neural text handling** in the RUAccent
pipeline: input normalization, sentence segmentation (a faithful re-implementation of
`razdel.sentenize`), word splitting, post-processing, sentence reassembly, and the
M3/M4 positional-alignment contract. A Swift developer should be able to reimplement
everything here with **no further Python reading**.

Source of truth (read order):

- `ruaccent-src/ruaccent/ruaccent.py` — `process_all_internal` (the orchestrator), `normalize`,
  `count_vowels`, `has_punctuation`, `delete_spaces_before_punc`, `_process_accent`, `_process_yo`.
- `ruaccent-src/ruaccent/text_preprocessor.py` — `split_by_words`, `split_by_sentences` (the one **actually used**).
- `ruaccent-src/ruaccent/text_postprocessor.py` — `fix_capital`.
- `ruaccent-src/ruaccent/text_split.py` — a SECOND `split_by_sentences`; **dead code, NOT used** (see §2.0).
- `.venv/lib/python3.13/site-packages/razdel/` — `segmenters/{sentenize,base,punct,sokr}.py`,
  `{rule,split,substring,record}.py`. The sentenizer port is reverse-engineered from these.

All codepoints, char sets, and divergence examples below were produced by running the actual
installed code (Python 3.13, transformers 5.12, razdel as installed). Where a value was *verified
by execution*, it is marked **[verified]**.

> Unicode note that pervades this doc: Python `re` is run here in **Unicode mode** (default for
> `str` patterns; razdel additionally passes `re.U` explicitly). So `\w` = `[\p{L}\p{N}_]` plus
> the Mn/Mc marks Python's `\w` includes, `\s` = Unicode whitespace, `\d` = Unicode decimal digit,
> `[^\W\d]` = "letter or underscore but not digit". Swift `Character`/`Unicode.Scalar` properties
> must be used, **not** ASCII-only checks. Concretely, Cyrillic а-я/А-Я, `ё`/`Ё`, Latin, and digits
> are all `\w`; `+` and `~` are **not** `\w` (this matters in §3). See §7 for exact predicates.

---

## 0. Pipeline overview (where this spec sits)

`process_all_internal(text)` (ruaccent.py:234) is the entry the consumer hits via `process_all`
(when no `skip_regex`). Order:

```
text = re.sub(self.normalize, "", text)                       # §1  normalize: DELETE disallowed chars
sentences = TextPreprocessor.split_by_sentences(text)         # §2  razdel sentenize + stitch
for sentence in sentences:
    words, remaining_text = TextPreprocessor.split_by_words(sentence)   # §3
    if len(words) == 0:                                        # §3 empty-words edge case
        outputs.append("".join(remaining_text)); continue
    stress_usages = M3.predict_stress_usage(sentence)         # NEURAL (M3) — §6 alignment contract
    processed_words = _process_yo(words, sentence)            # NEURAL (M4) + dicts
    processed_words = _process_omographs(processed_words)     # NEURAL (M2) + dict — §5.4
    processed_words = _process_accent(processed_words, stress_usages)   # dict / M1 — uses §4 helpers, §6
    processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)]
                                 + [remaining_text[-1]])       # §5 reassembly (off-by-one)
    processed_sentence = self.delete_spaces_before_punc(processed_sentence)  # §4.1
    outputs.append(processed_sentence)
return "".join(outputs)
```

This spec covers everything except the four neural models themselves. The neural models'
*tokenization and word-grouping* is covered ONLY to the extent the §6 alignment contract requires.

Output marking convention (not produced by this layer but stated for completeness): the dict/M1
accenting produces `+` **before** the stressed vowel; the package's public boundary
(`RussianStressing`) re-renders to `U+0301` **after** the vowel. That re-rendering is out of scope
for this doc.

---

## 1. Normalize regex (ruaccent.py:24, applied at process_all_internal:235)

### 1.1 What the code *looks* like vs. what Python *compiles*

Source line 24, byte-for-byte:

```python
self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;""''(){}\[\]«»„“”-]")
```

**This is a trap.** Python tokenizes the argument as **two adjacent string literals** that are
implicitly concatenated (verified via `tokenize`):

1. `r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;"` — a raw string that **ends at the `"` right after `;`**.
2. `"''(){}\[\]«»„""”-]"` — a *non-raw* string that supplies the rest.

The consequence: the `""` pair you *think* adds ASCII double-quote actually just closes literal #1
and opens literal #2, so **U+0022 (ASCII `"`) is NOT in the allowed set**. The `''` *does* survive
as two literal apostrophes. The final compiled pattern (obtained via AST `literal_eval` of the
actual node — **[verified]**) is:

```
[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;''(){}\[\]«»„“”-]
```

45 chars; full codepoint dump **[verified]**:

```
[  ^  a  -  z  A  -  Z  0  -  9  \  s  а  -  я  А  -  Я  ё  Ё  —  .  ,  !  ?  :  ;  '  '  (  )  {  }  \  [  \  ]  «  »  „  “  ”  -
```

Note indices 28–29 are two `'` (U+0027), and there is **no** U+0022 anywhere.

> Port directive: **hardcode the 45-char pattern above** in Swift; do not re-derive it from the
> source string with adjacent-literal semantics. Equivalently, build the allowed set explicitly
> (§1.3). Inside `[...]` the duplicate `''`, the multiple `-` ranges, and the `\[ \]` escapes are
> all harmless; the only behaviorally meaningful facts are the membership of the allowed set.

### 1.2 Semantics: `re.sub(pattern, "", text)`

The class is **negated** (`[^...]`). `re.sub` replaces every char matching the class (i.e. every
char **NOT** in the allowed set) with the empty string. Net effect: **DELETE all disallowed
characters**, keep order, no substitution, no normalization of survivors. Apply once to the whole
input before sentence splitting.

### 1.3 Exact allowed set (survivors). Everything else is deleted.

- ASCII letters `a-z A-Z`, ASCII digits `0-9`.
- Cyrillic core ranges `а-я` (U+0430..U+044F) and `А-Я` (U+0410..U+042F), **plus** `ё` (U+0451)
  and `Ё` (U+0401) explicitly (they sit *outside* the а-я/А-Я ranges, hence listed separately).
- `\s` = **Unicode whitespace** (space, tab `\t`, newline `\n`, CR `\r`, form-feed, vertical tab,
  NBSP U+00A0, and other Unicode space separators). **[verified]** NBSP survives.
- Punctuation/symbols, exactly: `—` (EM DASH U+2014), `.`, `,`, `!`, `?`, `:`, `;`,
  `'` (U+0027, apostrophe — survives), `(`, `)`, `{`, `}`, `[`, `]`,
  `«` (U+00AB), `»` (U+00BB), `„` (U+201E), `“` (U+201C), `”` (U+201D),
  `-` (U+002D, HYPHEN-MINUS).

### 1.4 Unusual / surprising membership (call these out in tests) — all **[verified]**

| Char | Codepoint | In set? | Note |
|------|-----------|---------|------|
| `"` ASCII double-quote | U+0022 | **DELETED** | the `""` collapsed to empty string (the trap in §1.1) |
| `'` ASCII apostrophe | U+0027 | **survives** | the `''` are two real members |
| `—` EM DASH | U+2014 | **survives** | the ONLY dash besides ASCII `-` |
| `-` HYPHEN-MINUS | U+002D | **survives** | |
| `–` EN DASH | U+2013 | **DELETED** | not in set |
| `−` MINUS SIGN | U+2212 | **DELETED** | |
| `―` HORIZONTAL BAR | U+2015 | **DELETED** | |
| `‒` FIGURE DASH | U+2012 | **DELETED** | |
| `“` `”` `„` curly double-quotes | U+201C/D, U+201E | **survive** | |
| `‘` `’` curly single-quotes | U+2018/9 | **DELETED** | (only ASCII `'` survives) |
| `«` `»` guillemets | U+00AB/BB | **survive** | |
| `…` HORIZONTAL ELLIPSIS | U+2026 | **DELETED** | a literal `…` is removed; only `...` (three ASCII dots) survives |
| `+` PLUS SIGN | U+002B | **DELETED** | manual-stress marker in INPUT is wiped here (see §1.5) |
| `~` TILDE | U+007E | **DELETED** | irrelevant: `~` is introduced *later* by split_by_words, after normalize |
| `_` `/` `@` `%` `#` `&` `\` | — | **DELETED** | |
| combining acute U+0301 | U+0301 | **DELETED** | a pre-combined stress mark in input is removed |
| `é` `ü` (Latin w/ diacritics) | U+00E9/FC | **DELETED** | |
| `ї` `і` `ў` (Ukr/Bel Cyrillic) | U+0457/0456/045E | **DELETED** | outside а-я; **silently dropped** |

### 1.5 Port consequences worth a unit test

- **Manual stress with `+` is wiped by normalize.** If a caller passes `"к+оса"`, normalize deletes
  the `+`, yielding `"коса"` before any word logic runs. So "manual stress in the input always wins"
  (CLAUDE.md) must be enforced by the **public package boundary BEFORE** `process_all_internal`,
  not by relying on `+` to survive normalize. (`split_by_words`'s `\w*(?:\+\w+)*` *can* carry a `+`,
  but only if a `+` is still present — which it won't be after normalize.) **Flag for integration:
  the package must intercept/protect caller-supplied stresses upstream of normalize.**
- **A literal ellipsis `…` is dropped but `...` is kept.** Sentence boundaries depend on this
  (razdel treats `…` as an ENDING), so after normalize there are no `…` chars for razdel to see —
  only `.` runs. Port must apply normalize **before** the sentenizer, exactly as upstream does.
- **Cyrillic-extended letters are silently removed**, which can glue two words together
  (`"мати́р"`→`"матир"` if the acute were combining; `"наїв"`→`"нав"`). Acceptable for parity
  (replicate upstream), but document it.

---

## 2. Sentence segmentation

### 2.0 WHICH `split_by_sentences` is used (resolve the ambiguity)

There are **two** functions named `split_by_sentences`:

- `ruaccent/text_split.py:97` — a hand-rolled regex sentenizer (`SENTENCE_SPLITTER`, `is_sentence_end`,
  its own `SHORTENINGS`/`JOINING_SHORTENINGS`/`PAIRED_SHORTENINGS`). **NOT CALLED by the pipeline.**
  Grep of `ruaccent.py` shows it imports `from .text_preprocessor import TextPreprocessor` and calls
  `TextPreprocessor.split_by_sentences` (lines 223, 236). `text_split` is imported nowhere in
  `ruaccent.py`. **Ignore `text_split.py` entirely for the port.** (Do not transcribe its lists; they
  differ from razdel's and would mislead.)
- `ruaccent/text_preprocessor.py:22` — the **USED** one. It delegates to **`razdel.sentenize`** and
  then stitches gap text back. This is what §2.1–§2.3 specify.

### 2.1 `TextPreprocessor.split_by_sentences` (the stitching wrapper)

```python
@staticmethod
def split_by_sentences(string):
    sentences = list(sentenize(string))          # list of razdel Substring(start, stop, text)
    if len(sentences) == 0:
        return []
    result = [string[l.stop:r.start] + r.text if l.stop != r.start else r.text
              for l, r in zip([Substring(0,0,"")] + sentences, sentences)]
    result[-1] = result[-1] + string[sentences[-1].stop:]
    return result
```

`sentenize` returns spans with `.start`/`.stop` char offsets into `string` and `.text` = the
**stripped** sentence body (razdel strips leading/trailing whitespace, see §2.2.6). The wrapper's
job is to make the returned list **lossless**: `"".join(result) == string` exactly **[verified]**.

**Algorithm (port this literally):**

1. Run the sentenizer → ordered list `spans` of `(start, stop, text)`.
2. If empty, return `[]`.
3. Build `result` by `zip` of `[sentinel] + spans` with `spans`, where `sentinel = (0,0,"")`:
   - For each pair `(prev, cur)`: the **inter-sentence gap** is `string[prev.stop : cur.start]`
     (the whitespace/leading text razdel stripped off the front of `cur` and the back of `prev`).
     If `prev.stop != cur.start`, prepend that gap to `cur.text`: `gap + cur.text`. Else just `cur.text`.
   - For the FIRST element, `prev` is the sentinel `(0,0,"")`, so the gap is `string[0 : spans[0].start]`
     = any leading whitespace before the first sentence. It is attached to the **first** sentence.
   - **Net rule: the gap between two sentences is reattached to the FOLLOWING (right) sentence.**
4. Append the **final tail** `string[spans[-1].stop:]` (trailing whitespace after the last sentence)
   to the LAST element: `result[-1] += tail`.

**Worked examples [verified]:**

```
"Привет мир. Как дела?"
  razdel spans: [0,11)"Привет мир."  [12,21)"Как дела?"
  result: ["Привет мир.", " Как дела?"]      # the gap " " (offset 11..12) went to sentence #2
  "".join == input ✓

"Что?! Невероятно... Да!"
  spans: [0,5)"Что?!"  [6,19)"Невероятно..."  [20,23)"Да!"
  result: ["Что?!", " Невероятно...", " Да!"]

"  Пробелы  в  начале.  Конец.  "      # note leading + internal + trailing whitespace
  spans: [2,21)"Пробелы  в  начале."  [23,29)"Конец."
  result: ["  Пробелы  в  начале.", "  Конец.  "]   # leading 2 spaces on #1, trailing 2 on #2

"Один.Два.Три."                          # no spaces -> single sentence (no boundary; see §2.2)
  result: ["Один.Два.Три."]
```

> Implementation tip: `Substring` here only needs `start`, `stop`, `text`. Use Swift `String.Index`
> or integer **character** offsets — but be careful: razdel offsets are **Python str indices =
> Unicode scalar / code-point counts**, NOT UTF-16 and NOT grapheme clusters. See §2.4 indexing note.

### 2.2 `razdel.sentenize` — full algorithm

`sentenize` is a `SentSegmenter` (segmenters/sentenize.py:350). Calling it (`base.py:44 __call__`)
does three stages: **split → segment(join/merge) → find_substrings**, then `post` strips each chunk.

```
def __call__(text):
    parts  = self.split(text)        # SentSplitter: yields alternating [str_chunk, SentSplit, str_chunk, ...]
    chunks = self.segment(parts)     # Segmenter.segment: greedily JOIN/SPLIT around each delimiter
    chunks = self.post(chunks)       # strip() each chunk
    return find_substrings(chunks, text)   # locate each chunk back in text -> Substring(start,stop,chunk)
```

#### 2.2.1 The splitter (`SentSplitter`, sentenize.py:301)

A regex finds **candidate boundary delimiters**; the text between consecutive delimiters becomes
"atoms", and each delimiter becomes a `SentSplit` carrying left/right context windows.

Delimiter regex (built at sentenize.py:295, **[verified]** final value):

```
([=:;]-?[)(]{1,3}|[\.\?!…;"„'»”’\)\]\}])
```

i.e. **either** a smiley `SMILES = [=:;]-?[)(]{1,3}`  (e.g. `:)`, `;-)`, `=(((`, `:(`)
**or** a single char from `DELIMITERS`. `DELIMITERS` (sentenize.py:55) =
`ENDINGS + ';' + GENERIC_QUOTES + CLOSE_QUOTES + CLOSE_BRACKETS`:

| Group | Source const | Chars |
|-------|--------------|-------|
| ENDINGS | punct.py:3 | `.` `?` `!` `…` (U+2026) |
| (literal) | — | `;` |
| GENERIC_QUOTES | punct.py:8 | `"` (U+0022) `„` (U+201E) `'` (U+0027) |
| CLOSE_QUOTES | punct.py:7 | `»` (U+00BB) `”` (U+201D) `’` (U+2019) |
| CLOSE_BRACKETS | punct.py:12 | `)` `]` `}` |

So `DELIMITERS = ".?!…;"„'»”’)]}"` (14 chars). NOTE: after **normalize (§1)** runs upstream,
`…` `"` `’` are already gone, so in this pipeline the live delimiters are effectively
`. ? ! ; „ ' » ” ) ] }` plus smileys. The Swift port should still implement the FULL delimiter set
(sentenize is a standalone component; future callers may bypass normalize).

`SentSplitter.__call__(text)` (sentenize.py:309), `window = 10`:

```
previous = 0
for each regex match m in text:          # finditer, left-to-right, non-overlapping
    start, stop = m.start(), m.end()
    delimiter = m.group(1)               # the matched delimiter or smiley string
    yield text[previous:start]           # ATOM (the chunk before this delimiter)
    left  = text[max(0, start-10) : start]   # up to 10 chars of left context
    right = text[stop : stop+10]             # up to 10 chars of right context
    yield SentSplit(left, delimiter, right)
    previous = stop
yield text[previous:]                    # trailing ATOM
```

So `parts` is a generator alternating **atom, split, atom, split, …, atom** (always starts and ends
with an atom; atoms may be empty strings).

> `left`/`right` are **10-char windows**, not full neighbors. The rules below only ever inspect the
> nearest token(s); 10 chars is enough. The port must use the same window or the token-extraction
> regexes (§2.2.4) could see different context near very long tokens. **Use window = 10.**

#### 2.2.2 The merge loop (`Segmenter.segment`, base.py:27)

Consumes `parts` and decides, at each delimiter, whether to **JOIN** (delimiter is *inside* a
sentence, keep buffering) or **SPLIT** (delimiter ends a sentence, emit buffer):

```
buffer = first_atom
for split in parts:            # 'split' is a SentSplit
    right = next(parts)        # the atom AFTER this split (consumes it)
    split.buffer = buffer      # full accumulated left side so far (used by some rules)
    if join(split):            # run RULES in order; first rule returning JOIN/SPLIT decides
        buffer = buffer + split.delimiter + right       # JOIN: absorb delimiter + right atom
    else:
        yield buffer + split.delimiter                  # SPLIT: emit sentence ending with delimiter
        buffer = right                                  # start next sentence with right atom
yield buffer                   # emit the final buffer
```

`join(split)` (base.py:21) runs the ordered `RULES`; the **first** rule that returns a truthy action
decides: returns `True` if that action == `JOIN` (`'join'`), `False` if `SPLIT` (`'split'`). If a
rule returns `None`, the next rule is tried. **If NO rule fires, `join` returns `None` → falsy →
the loop SPLITS.** So the default for an un-handled delimiter is **SPLIT**.

> Implementation note: the rules return one of `{None, 'join', 'split'}`. The "MAYBE" concept from
> `text_split.py` does **not** exist in razdel — razdel rules are binary (join/split) or abstain.

#### 2.2.3 `SentSplit` cached fields (sentenize.py:245)

Each `SentSplit(left, delimiter, right, buffer)` exposes (lazily computed) — port as plain computed
properties:

| Field | How computed (regex on the split) | Meaning |
|-------|-----------------------------------|---------|
| `left`, `delimiter`, `right`, `buffer` | given | windows + full left buffer |
| `right_space_prefix` | `bool(SPACE_PREFIX.match(right))`, `SPACE_PREFIX=^\s` | does `right` start with whitespace |
| `left_space_suffix` | `bool(SPACE_SUFFIX.search(left))`, `SPACE_SUFFIX=\s$` | does `left` end with whitespace |
| `right_token` | `FIRST_TOKEN.match(right).group(1)` | first token of `right` (after leading spaces) or `None` |
| `left_token` | `LAST_TOKEN.search(left).group(1)` | last token of `left` (before trailing spaces) or `None` |
| `left_pair_sokr` | `PAIR_SOKR.search(left).groups()` | `(a,b)` for `…a . b` shape (abbrev like `т.е`) or `None` |
| `left_int_sokr` | `INT_SOKR.search(left).group(1)` | word after a number, e.g. `5 руб` → `руб` (defined, **unused** by rules) |
| `right_word` | `WORD.search(right).group(1)` | first *word* (letters or digits) anywhere in `right` |
| `buffer_tokens` | `TOKEN.findall(buffer)` | all tokens in the accumulated buffer |
| `buffer_first_token` | `FIRST_TOKEN.match(buffer).group(1)` | (defined, **unused** by rules) |

#### 2.2.4 Token regexes (all `re.U`)

```
TOKEN       = ([^\W\d]+|\d+|[^\w\s])      # a run of letters, OR a run of digits, OR ONE non-word non-space char
FIRST_TOKEN = ^\s*([^\W\d]+|\d+|[^\w\s])  # first TOKEN, skipping leading whitespace
LAST_TOKEN  = ([^\W\d]+|\d+|[^\w\s])\s*$  # last TOKEN, allowing trailing whitespace
WORD        = ([^\W\d]+|\d+)              # letters-run or digits-run (no punctuation)
PAIR_SOKR   = (\w)\s*\.\s*(\w)\s*$        # "<wordchar> . <wordchar>" at end of left (e.g. "т.е")
INT_SOKR    = \d+\s*-?\s*(\w+)\s*$        # digits then a word at end of left
ROMAN       = ^[IVXML]+$                  # roman numeral (no D/C — note!)
```

Semantics to replicate exactly:

- `[^\W\d]` = "word char that is not a digit" = **letter or underscore** (Unicode letters incl.
  Cyrillic). A `TOKEN` is therefore: a maximal run of letters/underscore, **or** a maximal run of
  digits, **or** exactly one char that is neither word nor space (a single punctuation/symbol).
  Punctuation does **not** group: each punctuation char is its own token. (This is why
  `LAST_TOKEN` on `"...вот?!"`-style left windows returns just `!`.)
- `\s` is Unicode whitespace. `\w` is Unicode word char (`[\p{L}\p{N}_]`-ish).
- `ROMAN` notably excludes `C`/`D` (only `I V X M L`); replicate as-is.

#### 2.2.5 The RULES (order matters — first hit wins) — sentenize.py:331

Evaluated top-to-bottom; the first returning `JOIN`/`SPLIT` decides; `None` falls through. Constants:
`GENERIC_QUOTES = "„'`, `CLOSE_QUOTES = »”’`, `QUOTES = «“‘»”’"„'`, `CLOSE_BRACKETS = )]}`,
`ENDINGS = .?!…`, `DASHES = ‑–—−-` (U+2011, U+2013, U+2014, U+2212, U+002D — FIVE dashes),
`DELIMITERS` as §2.2.1, `SMILE_PREFIX = ^\s*[=:;]-?[)(]{1,3}`, `BULLET_CHARS = §абвгдеabcdef`,
`BULLET_BOUNDS = .)`, `BULLET_SIZE = 20`.

Helper: `is_lower_alpha(t)` = `t.isalpha() and t.islower()`. `is_sokr(t)` = `True` if `t.isdigit()`,
`True` if not `t.isalpha()` (punct), else `t.islower()`.

1. **`empty_side`** — if `left_token` is `None` OR `right_token` is `None` → **JOIN**.
   (A delimiter with no token on one side is never a sentence boundary, e.g. leading/trailing `.`.)
2. **`no_space_prefix`** — if `right` does NOT start with whitespace (`not right_space_prefix`)
   → **JOIN**. (No space after the delimiter ⇒ same sentence; this is why `"Точка.Без пробела."`
   stays one sentence — **[verified]** `"Один.Два.Три."` → 1 sentence.)
3. **`lower_right`** — if `right_token` is lowercase alpha (`is_lower_alpha`) → **JOIN**.
   (Sentence-initial lowercase word ⇒ not a real boundary.)
4. **`delimiter_right`** — (only reached when `right` DID start with whitespace, since rule 2
   `no_space_prefix` already JOINed the no-space case; so this rule handles `delimiter <space>
   delimiter/smiley`):
   - if `right_token` ∈ `GENERIC_QUOTES` (`"„'`) → return `None` (abstain, fall through);
   - elif `right_token` ∈ `DELIMITERS` → **JOIN**;
   - elif `right` starts with a smiley (`SMILE_PREFIX`) → **JOIN**.
   - **[verified]** `?!` is actually JOINed by rule 2 `no_space_prefix` (no space between `?` and `!`),
     NOT by this rule. `delimiter_right` fires for cases like `"Текст . ! Конец"` (`.` then space then
     `!`) and `"Конец. :) Дальше"` (`.` then space then smiley `:)`).
5. **`sokr_left`** (abbreviation on the left) — only if `delimiter == '.'`:
   - `match = left_pair_sokr` (the `(a,b)` from `PAIR_SOKR` = trailing `a . b`):
     if present, `left=(a.lower(), b.lower())`:
       - if `left ∈ HEAD_PAIR_SOKRS` → **JOIN**;
       - elif `left ∈ PAIR_SOKRS`: if `is_sokr(right_token)` → **JOIN**, else return `None`.
   - then `left = left_token.lower()`:
       - if `left ∈ HEAD_SOKRS` → **JOIN**;
       - elif `left ∈ SOKRS and is_sokr(right_token)` → **JOIN**.
6. **`inside_pair_sokr`** — if `delimiter == '.'` and `(left_token.lower(), right_token.lower()) ∈
   PAIR_SOKRS` → **JOIN**. (Handles `т.е` where the `.` is between the two letters.)
7. **`initials_left`** — if `delimiter == '.'`:
   - if `left_token.isupper() and len(left_token)==1` → **JOIN** (single capital = initial, e.g. `А.`);
   - if `left_token.lower() ∈ INITIALS` (`дж`,`ed`,`вс`) → **JOIN**.
8. **`list_item`** — if `delimiter ∈ BULLET_BOUNDS` (`.` or `)`):
   - if `len(buffer) > BULLET_SIZE` (20) → return `None`;
   - if **all** `buffer_tokens` satisfy `is_bullet` → **JOIN**.
     `is_bullet(t)`: `True` if `t.isdigit()`, if `t ∈ ".)"`, if `t.lower() ∈ BULLET_CHARS`
     (`§абвгдеabcdef`), or if `ROMAN.match(t)`. (Handles list markers `1.`, `а)`, `IV.`.)
9. **`close_quote`** — `delimiter = split.delimiter`; if `delimiter ∉ QUOTES` return `None`. Else:
   - if `delimiter ∈ CLOSE_QUOTES` (`»”’`) → `close_bound(split)`;
   - if `delimiter ∈ GENERIC_QUOTES` (`"„'`):
       - if NOT `left_space_suffix` → `close_bound(split)`; else → **JOIN**.
   `close_bound(split)`: if `left_token ∈ ENDINGS` (`.?!…`) return `None` (let later/earlier logic
   handle — here, abstain so default SPLIT applies via the *quote* being the boundary); else **JOIN**.
   (Effect: a closing quote ends a sentence only when it immediately follows sentence-ending
   punctuation, e.g. `…Привет.»` splits after `»`; otherwise the quote joins.)
10. **`close_bracket`** — if `delimiter ∈ CLOSE_BRACKETS` (`)]}`) → `close_bound(split)` (same logic
    as above: split after the bracket only if the preceding token is an ENDING).
11. **`dash_right`** — if `right_token ∉ DASHES` return `None`. Else `right = right_word`
    (first word in the right window); if `right and is_lower_alpha(right)` → **JOIN**.
    (A dash followed by a lowercase word ⇒ same sentence, e.g. dialogue dashes.)

If none of 1–11 fires → `join` returns `None` → loop **SPLITS**.

#### 2.2.6 `post` and `find_substrings`

- `post` (sentenize.py:358) yields `chunk.strip()` for each emitted chunk — Python `str.strip()`
  removes **leading/trailing Unicode whitespace** (this is why `.text` is the trimmed body and the
  wrapper in §2.1 must reattach gaps). Empty-after-strip chunks are still yielded as `""` (but see
  find_substrings below).
- `find_substrings(chunks, text)` (substring.py:14): re-locates each chunk in `text`:
  ```
  offset = 0
  for chunk in chunks:
      start = text.find(chunk, offset)   # first occurrence at/after offset
      stop  = start + len(chunk)
      yield Substring(start, stop, chunk)
      offset = stop
  ```
  Uses a moving `offset` so repeated identical sentences map to successive occurrences
  (**[verified]** `"Да. Да. Да."` → `[0,3) [4,7) [8,11)`). **Port caveat:** if a stripped `chunk`
  is `""`, `text.find("", offset) == offset` and `stop == offset` → a zero-width Substring. The
  wrapper §2.1 tolerates this (gap math still works) but you generally won't see empty chunks for
  normal input. `len(chunk)` is a **code-point** length (see §2.4).

### 2.3 The abbreviation/initial lists — TRANSCRIBED VERBATIM (sokr.py)

These are the ONLY data the rules consult. Transcribe **exactly** (lowercase ASCII/Cyrillic as
written). Build them in Swift as `Set<String>` / `Set<[String]>` (pairs) constants.

`parse_sokrs(lines)`: splits each line on whitespace, yields each whitespace-separated word.
`parse_pair_sokrs(lines)`: each line → one tuple of its whitespace-separated words.

**`TAIL_SOKRS`** (sokr.py:14):
```
дес тыс млн млрд дол долл коп руб р проц га барр куб кв км см
час мин сек в вв г гг с стр co corp inc изд ed др al
```

**`HEAD_SOKRS`** (sokr.py:40):
```
букв ст трад
лат венг исп кат укр нем англ фр итал греч
евр араб яп слав кит рус русск латв словацк хорв
mr mrs ms dr vs св арх зав зам проф акад кн корр ред гр ср
чл корр им тов нач пол chap
п пп ст ч чч гл стр абз пт no
просп пр ул ш г гор д стр к корп пер корп обл эт пом ауд оф ком комн каб
домовлад лит т рп пос с х пл bd о оз р а
обр ум ок откр пс ps upd см напр доп юр физ тел сб внутр дифф гос отм
```

**`OTHER_SOKRS`** (sokr.py:97):
```
сокр рис искл прим яз устар шутл
```

`SOKRS = TAIL_SOKRS | HEAD_SOKRS | OTHER_SOKRS` (sokr.py:105). Deduplicated union; the
single-word `SOKRS` membership test (rule 5) uses this union, while rule 5's *first* check uses
`HEAD_SOKRS` alone. Build both sets.

> Dedup note: several entries repeat across groups (`ст`, `стр`, `корп`, `см`, `г`, `р`, `о`, `с`,
> `co corp inc`, `ed`, `др`/`al`). Sets dedup; no behavioral effect. Just don't drop any.

**Pair sokrs** — built by `parse_pair_sokrs` (one tuple per line):

`TAIL_PAIR_SOKRS` (sokr.py:107):
```
('т','п') ('т','д') ('у','е') ('н','э') ('p','m') ('a','m') ('с','г') ('р','х')
('с','г') ('с','ш') ('з','д') ('л','с') ('ч','т') ('т','д')
```
(Note `('с','г')` and `('т','д')` appear twice — harmless dups.)

`HEAD_PAIR_SOKRS` (sokr.py:123):
```
('т','е') ('т','к') ('т','н') ('и','о') ('к','н') ('к','п') ('п','н') ('к','т') ('т','н') ('л','д')
```

`OTHER_PAIR_SOKRS` (sokr.py:134) — **transcribe with the upstream BUG preserved for parity**:
```python
'ед ч',
'мн ч',
'повел накл',
'жен р'        # <-- NO trailing comma! Python concatenates this with the next line:
'муж р',       #     -> the literal becomes 'жен р' + 'муж р' = 'жен рмуж р'
```
So `parse_pair_sokrs` sees lines `['ед ч', 'мн ч', 'повел накл', 'жен рмуж р']` and produces tuples:
```
('ед','ч') ('мн','ч') ('повел','накл') ('жен','рмуж','р')   # last is a 3-tuple, never matches a 2-tuple lookup
```
**[verified]** `('жен','р')` and `('муж','р')` are therefore **NOT** in `OTHER_PAIR_SOKRS` (the
missing comma fused them into a useless 3-element tuple). **Replicate this:** do NOT add `('жен','р')`
or `('муж','р')`. The 3-tuple is dead (lookups are always 2-tuples), so you may simply omit it.

`PAIR_SOKRS = TAIL_PAIR_SOKRS | HEAD_PAIR_SOKRS | OTHER_PAIR_SOKRS` (sokr.py:142). Effective
2-tuple members (union, the dead 3-tuple dropped):
```
(т,п)(т,д)(у,е)(н,э)(p,m)(a,m)(с,г)(р,х)(с,ш)(з,д)(л,с)(ч,т)
(т,е)(т,к)(т,н)(и,о)(к,н)(к,п)(п,н)(к,т)(л,д)
(ед,ч)(мн,ч)(повел,накл)
```

`HEAD_PAIR_SOKRS` (used directly by rule 5) = `{(т,е),(т,к),(т,н),(и,о),(к,н),(к,п),(п,н),(к,т),(л,д)}`.

`INITIALS` (sokr.py:144): `{'дж', 'ed', 'вс'}`.

### 2.4 Indexing / string-length parity note (CRITICAL for Swift)

razdel and the wrapper compute offsets and lengths in **Python `str` units = Unicode code points
(scalars)**. `text.find`, `len(chunk)`, `[start:stop]` slices, the 10-char windows, and regex
positions are all code-point based.

- Swift `String.count` and `String.Index` are **grapheme-cluster** based; `String.utf16` is UTF-16.
  **Neither matches Python directly** for non-BMP or combining-mark text.
- **Use `String.unicodeScalars` (`[Unicode.Scalar]`) as the working unit** for all offset math,
  window slicing, `find`, and length, to mirror Python `str`. After normalize (§1) most input is
  BMP Cyrillic/Latin (1 scalar each, no combining marks survive — U+0301 is deleted), so in practice
  scalars == code points == what Python sees. But implement on scalars to be safe; do **not** index
  by grapheme or UTF-16.
- The `«»„“”—` chars are all single BMP scalars; `…` is deleted by normalize; smileys are ASCII.
  So post-normalize there are no surrogate pairs in normal Russian text — but keep scalar indexing
  as the contract.

### 2.5 `sentenize` behavior corpus (regression fixtures) — all **[verified]**

| Input | Sentence spans (`razdel`) | Note |
|-------|---------------------------|------|
| `Привет мир. Как дела?` | `[0,11)"Привет мир."`, `[12,21)"Как дела?"` | normal split |
| `Это т.е. пример.` | `[0,16)"Это т.е. пример."` | `т.е.` joined via pair-sokr + sokr rules |
| `В 1996 г. он родился. Потом уехал.` | `…"В 1996 г. он родился."`, `…"Потом уехал."` | `г.` (HEAD_SOKRS) joins, real `.` splits |
| `А.С. Пушкин писал стихи.` | `[0,24)` single | `А.` `С.` initials JOIN |
| `Цена 5 руб. за штуку. Дорого!` | `…"Цена 5 руб. за штуку."`, `…"Дорого!"` | `руб.` (TAIL_SOKRS) joins |
| `Он сказал: «Привет.» И ушёл.` | `[0,20)"Он сказал: «Привет.»"`, `[21,28)"И ушёл."` | `.` before `»` joins via `no_space_prefix`; `»` joins via `no_space_prefix` (right=` И…`)... wait: split is AFTER `»`. Trace: the `.`→`»` join (no space), `»`→` И` then the boundary; the final `.` joins via `empty_side`. Net: split after `»` |
| `Список: 1. Первое. 2. Второе.` | `[0,10)"Список: 1."`, `[11,18)"Первое."`, `[19,29)"2. Второе."` | `:` not a delimiter here; `1.` is its own item; bullet logic on `2.` |
| `Что?! Невероятно... Да!` | `"Что?!"`, `"Невероятно..."`, `"Да!"` | `?!` joins via **`no_space_prefix`** (NOT delimiter_right — no space after `?`); `...` internal `.`s join via `no_space_prefix` too |
| `Сравни ст. 5 ч. 2 закона.` | single | `ст.` `ч.` are sokrs → join |
| `Дж. Р. Р. Толкин — автор.` | single | `Дж` ∈ INITIALS; `Р.` single-capital initials |
| `Смайл :) был тут. И тут.` | `"Смайл :) был тут."`, `"И тут."` | `:)` smiley joined (followed by space+lowercase) |
| `Точка.Без пробела.После.` | single | `no_space_prefix` → JOIN every `.` |
| `  Пробелы  в  начале.  Конец.  ` | `[2,21)…`, `[23,29)…` | leading/internal/trailing spaces preserved by wrapper |
| `Да. Да. Да.` | `[0,3) [4,7) [8,11)` | find_substrings moving offset |

---

## 3. `split_by_words` (text_preprocessor.py:6)

```python
@staticmethod
def split_by_words(string):
    string = string.replace(" - ", " ~ ")
    match = list(re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower()))
    remaining_text = [string[l.end():r.start()] for l, r in zip(match, match[1:])]
    words = [string[x.start():x.end()] for x in match]
    words_mask = [i for i, w in enumerate(words) if w]
    valid_words = [words[i] for i in words_mask]
    if len(words_mask) == 0:
        return valid_words, ["", ""]
    remaining_text_res = (["".join(remaining_text[:words_mask[0]])]
        + ["".join(remaining_text[l+1:r]) for l, r in zip(words_mask, words_mask[1:])])
    remaining_text_res.append("".join(remaining_text[words_mask[-1]+1:]))
    return valid_words, remaining_text_res
```

### 3.1 Step-by-step

1. **`" - " → " ~ "`** — replace the exact 3-char sequence space-hyphen-space with
   space-tilde-space (Python `str.replace`, ALL non-overlapping occurrences, left to right).
   Rationale: a spaced hyphen is a dash/punctuation, not a word-internal hyphen; `~` is a sentinel
   restored to `-` later by `delete_spaces_before_punc` (§4.1). **The `~` becomes its own word token.**
   (Note: overlapping case `" - - "` → Python replaces non-overlapping: `" - - "` →first match at
   idx0 consumes `" - "`, leaving `"- "`, no further `" - "` → result `" ~ - "`. Replicate Python
   `replace` semantics exactly: scan left→right, non-overlapping.)

2. **Tokenize on `string.lower()`** with `re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", ...)` (Unicode).
   The regex alternation, tried left-to-right at each position:
   - **`\w*(?:\+\w+)*`** — a (possibly empty) run of word chars, optionally followed by groups of
     `+` then word chars. This captures **words including embedded manual-stress `+`**
     (`к+оса`, `сло+во`). NOTE: `\w*` can match the **empty string**, so `finditer` produces
     **zero-width matches** at non-word, non-`+` boundaries (these become `""` entries, filtered by
     `words_mask`). The `(?:\+\w+)*` requires `+` to be followed by ≥1 word char, so a trailing bare
     `+` (`уже+`) is NOT absorbed — it falls to the next branch.
   - **`[^\w\s]+`** — a maximal run of chars that are neither word nor whitespace
     (**punctuation/symbols grouped together**: `?!`, `...`, `».`).
   - Whitespace matches **neither** branch → it is the gap captured in `remaining_text`.

   > **Lowercasing for span-finding only.** `finditer` runs on `string.lower()`, but `words` are
   > sliced from the **original-case `string`** (`string[x.start():x.end()]`). So **word case is
   > preserved** (`"Привет"` stays `"Привет"`). The downstream `_process_yo` re-applies case via
   > `fix_capital` (§4.3). Caveat: Python `str.lower()` is locale-independent and can change string
   > **length** for a few chars (e.g. `İ`→`i̇`), which would desync `x.start()/x.end()` from the
   > original. For post-normalize Russian/Latin text this does not occur (all survivors lower 1:1).
   > **Port directive:** lowercase per Unicode default case-folding-lite (Python `str.lower`), but
   > slice spans from the ORIGINAL string; verify with the §3.4 fixtures.

3. **`remaining_text`** = inter-match gaps: for consecutive matches `l, r`, the substring
   `string[l.end():r.start()]` (from ORIGINAL `string`). `len(remaining_text) == len(match) - 1`.
   These hold whitespace and (because zero-width word matches interleave with punctuation runs)
   often empty strings.

4. **`words`** = `[string[x.start():x.end()] for x in match]` — includes the empty-string matches.

5. **`words_mask`** = indices `i` where `words[i]` is non-empty (truthy). Empty matches dropped.

6. **`valid_words`** = `words` at those indices = the returned word list (no empties).

7. **Empty edge case:** if `words_mask == []` (input had no word/punctuation tokens, e.g. all
   whitespace or empty) → return `([], ["", ""])`. **[verified]** `"   "` → `([], ["", ""])`.
   Note the returned `remaining_text` is **always length-2** in this case, and the caller's
   reassembly (§5) is bypassed because `process_all_internal` short-circuits on `len(words)==0`
   (line 240) and just emits `"".join(remaining_text)` = `""` (both elements empty). For
   `split_by_words` of an all-whitespace sentence, the whitespace is **lost** here — but in practice
   sentences come from the sentenizer with content, and the gap whitespace lives in the *sentence
   wrapper's* remaining text, so the all-whitespace sentence path is rare/degenerate.

8. **`remaining_text_res`** = re-bucketed gaps so that `len(remaining_text_res) == len(valid_words)+1`
   (the **off-by-one**, see §5):
   - element 0 = join of all gaps **before** the first real word (`remaining_text[:words_mask[0]]`)
     — leading punctuation/whitespace.
   - element k (1..n-1) = join of gaps strictly **between** real word k-1 and word k
     (`remaining_text[l+1:r]` for consecutive mask indices `l,r`).
   - last element = join of gaps **after** the last real word (`remaining_text[words_mask[-1]+1:]`).

### 3.2 Returned contract

`valid_words` (length n) and `remaining_text_res` (length n+1) satisfy, for any non-empty-words
sentence: **interleaving `remaining_text_res[i] + valid_words[i]` then the final
`remaining_text_res[n]` reconstructs the (`" - "`→`" ~ "`-substituted) input exactly.** The reassembly
in §5 relies on this. **[verified]** for all §3.4 examples.

> The reconstruction is over the **`~`-substituted** string, not the literal original — the `~` is
> restored to `-` only at the very end by `delete_spaces_before_punc` (§4.1, §5).

### 3.3 Unicode `\w` / `\s` semantics to replicate (Swift §7)

- `\w` (Python, Unicode) ⊇ `{Cyrillic letters а-я А-Я ё Ё, Latin a-z A-Z, digits 0-9, underscore _,
  and other Unicode letters/marks}`. Post-normalize, the live `\w` set is Cyrillic core + ё/Ё +
  Latin + digits + `_` (though `_` is deleted by normalize, so it won't appear).
- `+` and `~` are **NOT** `\w` and **NOT** `\s` → they fall in `[^\w\s]+` **unless** consumed by the
  `(?:\+\w+)*` word-with-stress branch (only `+` between word chars). A standalone `~` (from the
  `" - "` swap) is therefore a punctuation token `[^\w\s]+` of length 1.
- `\s` = Unicode whitespace (same as §1.3 `\s`).

### 3.4 Worked examples — all **[verified]**

```
"Привет мир."            -> words=['Привет','мир','.']          remaining=['',' ','','']
"  Как дела?"            -> words=['Как','дела','?']             remaining=['  ',' ','','']   # leading spaces in rem[0]
"что-то и кто-то"        -> words=['что','-','то','и','кто','-','то']  remaining=['','','',' ',' ','','','']
                            # NOTE: word-internal '-' (not " - ") is NOT swapped; it is a [^\w\s] token
"слово - тире"           -> words=['слово','~','тире']          remaining=['',' ',' ','']    # " - " -> " ~ "
"к+оса уже+"             -> words=['к+оса','уже','+']           remaining=['',' ','','']     # 'к+оса' kept whole; trailing '+' separate
"Цена 5 руб."            -> words=['Цена','5','руб','.']        remaining=['',' ',' ','','']
"Привет, как дела?"      -> words=['Привет',',','как','дела','?'] remaining=['','',' ',' ','','']
"Что?! Невероятно..."    -> words=['Что','?!','Невероятно','...'] remaining=['','',' ','','']  # '?!' and '...' grouped
"..."                    -> words=['...']                        remaining=['','']
"   "                    -> words=[]                             remaining=['','']            # empty edge case
"a.b.c"                  -> words=['a','.','b','.','c']          remaining=['','','','','','']
```

Reassembly check (§5) e.g. `"Привет мир."`: `''+'Привет' + ' '+'мир' + ''+'.' + ''` = `"Привет мир."` ✓.

---

## 4. Post-processing helpers (ruaccent.py / text_postprocessor.py)

### 4.1 `delete_spaces_before_punc` (ruaccent.py:137)

```python
def delete_spaces_before_punc(self, text):
    punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"   # 30 chars; NOTE: NO '~' here
    for char in punc:
        if char == '-':
            text = text.replace(" " + char, char).replace(char + " ", char)
        text = text.replace(" " + char, char)
    return text.replace('~', '-')
```

Exact `punc` set (30 chars, **[verified]** by reading the literal; note the backslash-escaping):
`! " # $ % & ' ( ) * , . / : ; < = > ? @ [ \ ] ^ _ \` { | } -`
i.e. ASCII `!"#$%&'()*,./:;<=>?@[\]^_`{|}-` — this is the ASCII punctuation block **minus** `+` and
**minus** `~` (and `~`/`+` are intentionally excluded; `~` is handled by the final `.replace`).

Behavior, per char (loop order = order of chars in `punc` string above):
- For every `char`: remove a **single** preceding space: `text.replace(" "+char, char)`. (Python
  `str.replace` replaces ALL non-overlapping occurrences in one pass; only ONE space is removed per
  occurrence — `"  ,"` → `" ,"` after this single pass, NOT `","`.)
- **Special-case `'-'`:** ADDITIONALLY run `.replace(" -", "-")` then `.replace("- ", "-")` BEFORE
  the generic `.replace(" -","-")`. Net for `-`: removes a space **before** AND a space **after**
  the hyphen (so `" - "`-derived content collapses), then the generic pass runs again (idempotent).
  Order: `if char=='-': text = text.replace(" -","-").replace("- ","-")` then unconditionally
  `text = text.replace(" -","-")`.
- **Final:** `text.replace('~', '-')` — restore every `~` sentinel (from §3.1) back to `-`.
  **IMPORTANT [verified]:** this restore runs **AFTER** the per-char loop, so a `~` that still has
  surrounding spaces (e.g. `" ~ "`) becomes `" - "` with the spaces **INTACT** — the `-` space-removal
  pass already ran on the original `-` chars, NOT on the freshly-restored ones. So
  `delete_spaces_before_punc("a ~ b")` → `"a - b"` (spaces kept), NOT `"a-b"`. By contrast a literal
  `" - "` already present in `text` (not via `~`) → `"-"` (spaces removed). Do not conflate the two.

  Verified behaviors (port must match exactly):
  - `"слово ,"` → `"слово,"`   (one preceding space removed)
  - `"слово  ,"` → `"слово ,"` (only ONE space removed — single pass, run NOT collapsed)
  - `"a - b"` (literal hyphen) → `"a-b"` (before+after space removed)
  - `"a ~ b"` (tilde sentinel) → `"a - b"` (spaces KEPT; `~`→`-` happens last)
  - `"x  -  y"` → `"x- y"` (asymmetric: `" -"`→`"-"` removes one leading space twice via the two
    `" -"` passes leaving `"x-  y"`... actually: `.replace(" -","-")` turns `"x  -  y"`→`"x -  y"`→ via
    second generic pass `"x-  y"`; `.replace("- ","-")` turns the `"-  y"`→`"- y"` ⇒ net `"x- y"`)

> Port directive: implement as the **literal loop in the given char order**. The single-pass
> `replace` (one space removed per occurrence) is load-bearing — do **not** collapse runs of spaces.
> Only `-` gets the extra after-space removal. Do the `~`→`-` substitution **last**, after the loop,
> and do **not** re-run space removal on the restored `-` (so `" ~ "` → `" - "`, spaces kept).

### 4.2 `count_vowels` (ruaccent.py:127)

```python
vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
return sum(1 for char in text if char in vowels)
```

Vowel set (20 chars, **[verified]**), both cases: `а е ё и о у ы э ю я` + `А Е Ё И О У Ы Э Ю Я`
(U+0430,0435,0451,0438,043E,0443,044B,044D,044E,044F and uppercase counterparts incl. `Ё`/`ё`).
Counts Cyrillic vowels only; Latin/digits don't count. Used in `_process_accent` gate
(`count_vowels(lower_word) > 1` ⇒ eligible for the M1 char model).

### 4.3 `has_punctuation` (ruaccent.py:131)

```python
for char in text:
    if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        return True
return False
```

Punctuation set (32 chars, **[verified]**): the **full ASCII punctuation block** `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`
— i.e. `delete_spaces_before_punc`'s set **plus `+` and `~`**. Returns `True` if ANY char of `text`
is in this set. Used in `_process_accent`: a word containing punctuation (e.g. a `+` already there,
or stray punctuation) is **not** sent to the M1 char model.

### 4.4 `fix_capital` (text_postprocessor.py:1)

```python
def fix_capital(source, target):
    if len(source) != len(target):
        return target
    mask = [x.isupper() for x in source]
    return "".join([tgt.upper() if m else tgt.lower() for m, tgt in zip(mask, target)])
```

- If `len(source) != len(target)` (Python `len` = code-point count) → return `target` unchanged.
- Else, position-by-position: where `source[i].isupper()`, force `target[i].upper()`, else
  `target[i].lower()`. I.e. **transfer source's per-character capitalization onto target.**
- `str.isupper()`/`.upper()`/`.lower()` are Unicode-aware. For Russian letters this is 1:1.
- **Length parity caveat:** `len` is code points. With Swift, compare `source.unicodeScalars.count`
  vs `target.unicodeScalars.count`, and map per **scalar** (not grapheme). Post-normalize text is
  combining-mark-free, so scalar==char; still, use scalars to match Python.
- Used in `_process_yo` to re-apply the original word's casing onto the ё-restored dictionary form
  (which is lowercase in the dict). E.g. `source="Все"`, `target="всё"` → `"Всё"`.

---

## 5. Sentence reassembly (process_all_internal:247) and the off-by-one

```python
processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)]
                             + [remaining_text[-1]])
processed_sentence = self.delete_spaces_before_punc(processed_sentence)
```

- `remaining_text` and `processed_words` come from §3 with `len(remaining_text) == len(processed_words)+1`
  (the off-by-one: there is a gap-bucket **before** word 0, between every adjacent pair, and **after**
  the last word ⇒ n words ⇒ n+1 gaps).
- `zip(remaining_text, processed_words)` pairs gap[i] with word[i] for i in 0..n-1 (zip stops at the
  shorter list = `processed_words`, length n). Each pair contributes `gap[i] + word[i]`.
- The list then has `+ [remaining_text[-1]]` appended = the **final trailing gap** (`remaining_text[n]`),
  which `zip` skipped. This is why the explicit append is needed: **`zip` would otherwise drop the
  last gap bucket**, truncating trailing whitespace/punctuation.
- Result = `gap0 + word0 + gap1 + word1 + … + gap_{n-1} + word_{n-1} + gap_n` = the sentence
  reconstructed with `processed_words` substituted for the originals (and `~` still present), then
  `delete_spaces_before_punc` (§4.1) cleans spacing and restores `~`→`-`.

> The `process_yo` variant (ruaccent.py:228) uses `"".join([l+r for l,r in zip(remaining_text, processed_words)])`
> **WITHOUT** the `+ [remaining_text[-1]]` append — so `process_yo` (the ё-only public method) drops the
> trailing gap. The main `process_all_internal` (line 247) DOES append it. **Port both correctly per
> method; the accenting path uses the append.**

### 5.1 Whole-sentence assembly across the output list

`process_all_internal` collects per-sentence strings into `outputs` and returns `"".join(outputs)`
(no separator). Because §2.1 already attached inter-sentence gaps (including the leading/trailing
whitespace) to the sentences, `"".join(outputs)` reproduces the original spacing between sentences.
(`process_yo` uses `" ".join(outputs)` — a different join; again, port per method.)

### 5.2 Empty-words sentence in the loop (process_all_internal:240)

If `split_by_words(sentence)` yields `len(words)==0`, the loop appends `"".join(remaining_text)` and
`continue`s (skipping all neural + reassembly). With the §3.7 empty edge case `remaining_text==["",""]`,
this contributes `""`. So an all-whitespace sentence body collapses to nothing in output (the
whitespace was already accounted for in neighboring sentences by §2.1's stitching in normal inputs).

---

## 6. ALIGNMENT CONTRACT & PARITY RISK — M3 stress_usages vs `split_by_words`

### 6.1 The contract as written

`_process_accent(text, stress_usages)` (ruaccent.py:203) indexes **positionally**:

```python
for i, word in enumerate(splitted_text):     # splitted_text == split_by_words words (length n)
    ...
    if stress_usages[i] == "STRESS":          # stress_usages from M3 (length m)
        ...
```

`stress_usages = extract_entities(M3.predict_stress_usage(sentence))` — a list of label strings
(`"NO_STRESS"`/`"PUNCT"`/`"STRESS"`, per M3 `id2label` **[verified]** =
`{0:"NO_STRESS", 1:"PUNCT", 2:"STRESS"}`). `predict_stress_usage` groups WordPiece tokens into
"words" via `aggregate_words` with **AVERAGE** aggregation (stress_usage_model.py:84/109): a new word
starts at each non-subword token; subword tokens (`##...`, `is_subword` true) are absorbed; the
group's label = argmax of the **mean softmax** over its member tokens. Special tokens
(`[CLS]/[SEP]/[PAD]`) are skipped (`special_tokens_mask`).

**The assumption (unstated in code):** `stress_usages[i]` describes `words[i]` — i.e. the M3
WordPiece-word grouping yields the **same number, in the same order**, as `split_by_words`. The two
come from **different segmenters** (BERT WordPiece vs the §3 regex), so this is an *assumption*, not a
guarantee. M3 tokenizer is BERT WordPiece (`continuing_subword_prefix="##"`, `unk_token_id=1`,
`[UNK]` for OOV like `—`) **[verified]**.

The **same positional contract** governs M4 in `_process_yo` (ruaccent.py:159): `yo_predictions[i]`
vs `words[i]`, where `yo_predictions` comes from `M4.predict_yo_homographs(sentence)` using the
**identical** `aggregate_words`/AVERAGE machinery (yo_homograph_model.py). So **the same risk applies
to M4.** (M2 omograph does NOT use positional alignment against M3 — it operates per-word by dict
lookup, see §5.4-equivalent in ruaccent.py:167.)

### 6.2 When it HOLDS (the common case) — **[verified]**

For typical sentences where every `split_by_words` token corresponds 1:1 to a WordPiece word group,
the lengths and order match. Verified equal for:
`"Привет мир."`, `"Привет, как дела?"`, `"Цена 5 руб."`, `"что-то и кто-то"`, `"Это т.е. пример."`,
`"Слово — тире здесь."` (even with `—`→`[UNK]` and `тире`→`ти##ре`), `"В 1996 г. родился."`.
The agreement holds because **single** punctuation chars and whole words split the same way in both,
and WordPiece subwords get re-merged by `aggregate_words`.

### 6.3 When it DIVERGES (the parity risk) — **[verified], reproducible**

**`split_by_words` groups CONSECUTIVE punctuation `[^\w\s]+` into ONE token, but the BERT WordPiece
tokenizer emits each punctuation char as a SEPARATE token (each its own "word" group).** Whenever a
sentence contains a punctuation **run of length ≥2**, `len(stress_usages) > len(words)` and **all
indices after the run shift**. Also `+`/`=` (if present) and any char WordPiece splits differently
contribute. Verified divergences:

| Sentence | `split_by_words` (n) | M3 groups (m) |
|----------|----------------------|---------------|
| `Что?! Невероятно...` | `['Что','?!','Невероятно','...']` (4) | `['Что','?','!','Невероят##но','.','.','.']` (7) |
| `Да!!! Нет???` | `['Да','!!!','Нет','???']` (4) | `['Да','!','!','!','Нет','?','?','?']` (8) |
| `Текст (в скобках).` | `['Текст','(','в','скобках',').']` (5) | `['Текст','(','в','скобках',')','.']` (6) |
| `«Цитата».` | `['«','Цитата','».']` (3) | `['«','Цита##та','»','.']` (4) |
| `Привет... мир` | `['Привет','...','мир']` (3) | `['Привет','.','.','.','мир']` (5) |

**Concrete downstream corruption** (**[verified]**), `"Ого?! замок открыт."`:

```
words      : [0]'Ого'  [1]'?!'   [2]'замок'  [3]'открыт'  [4]'.'
M3 entities: [0]'Ого'  [1]'?'    [2]'!'      [3]'замок'   [4]'открыт'  [5]'.'   (len 6 > 5)

positional pairing stress_usages[i] vs words[i]:
  [2] word='замок'  gets stress_usages[2] = label of M3 '!'  (PUNCT, not "STRESS")
  [3] word='открыт' gets stress_usages[3] = label of M3 'замок'
```

So `замок` (an omograph requiring stress) is paired with a PUNCT label ⇒ the
`if stress_usages[i]=="STRESS"` gate is **False** ⇒ `замок` is **silently left unstressed**. Every
real word after a multi-char punctuation run is mis-gated this way.

### 6.4 Why it does NOT usually crash

`_process_accent` only indexes `stress_usages[i]` for `i in range(len(words))`. The divergence above
makes `len(stress_usages) >= len(words)` in the punctuation-run case (WordPiece *adds* tokens), so the
read stays in-bounds — it just reads the **wrong** (shifted) label. **However**, the opposite is
possible: if WordPiece ever produces **fewer** word-groups than `split_by_words` (e.g. a sequence the
WordPiece tokenizer merges that the regex splits), `len(stress_usages) < len(words)` and
`stress_usages[i]` would **IndexError**. We did not find such a case in the verified corpus, but it is
not provably impossible — **mark as an open risk and add a guard** (see §6.6).

### 6.5 Parity directive for the Swift port

To match upstream **exactly** (parity-first, per CLAUDE.md), the Swift port must:

1. Reproduce M3/M4's `aggregate_words` grouping (new word at each non-`##` token, AVERAGE over the
   group's softmax) so that `stress_usages`/`yo_predictions` are produced with the **same length and
   order** as upstream — i.e. **do NOT "fix" the alignment to match `split_by_words`**. Replicate the
   buggy positional pairing so on-device output equals upstream output, including the `замок`-style
   miss.
2. Equivalently: index `stress_usages[i]` against `words[i]` positionally, **as-is**. Do not attempt
   to re-align by word text. (A "smarter" alignment would change outputs and break parity with the
   oracle.)
3. This is a **must-test divergence**: include the §6.3 sentences (punctuation runs, leading
   guillemets, bracketed clauses) in the parity fixtures, asserting on-device output == upstream
   `process_all_internal` output **including the mis-stress**. This is the cheapest way to catch a
   port that "accidentally fixed" the alignment.

### 6.6 Safety guard (parity-preserving)

Because §6.4 leaves an IndexError possibility, mirror upstream's *effective* behavior: if a future
input makes `i >= len(stress_usages)`, upstream would raise. For robustness the Swift port MAY clamp
(treat missing as non-`"STRESS"` ⇒ skip accenting) but should **log/flag** it, since upstream crashes
there and the parity oracle would too. Keep the guard behind a flag and default to parity (let the
fixture corpus reveal any real occurrence). **Open question (§ risks): does any natural post-normalize
sentence yield `len(stress_usages) < len(words)`?** Not observed; needs a fuzz pass.

### 6.7 Note on `sentence` vs `words` source

M3/M4 receive the **raw `sentence` string** (with original case, the `" - "`→`" ~ "` swap **not**
applied — that swap is internal to `split_by_words`), while `words` come from `split_by_words(sentence)`
which **does** apply the `~` swap. So a sentence containing `" - "` yields a `~` word that has **no**
counterpart token from M3 (M3 sees `-` as a separate `[UNK]`/punct token). This is another source of
index drift and is covered by the same §6.5 parity directive (replicate, don't fix). Add a `" - "`
fixture (e.g. `"кто - то идёт"`).

---

## 7. Swift Unicode predicates (exact equivalents)

Implement these on `Unicode.Scalar` (preferred) to mirror Python `re` Unicode classes. Use
`String.unicodeScalars` as the iteration/index unit throughout (§2.4).

| Python | Meaning (this pipeline) | Swift equivalent |
|--------|-------------------------|------------------|
| `\s` | Unicode whitespace | `scalar.properties.isWhitespace` (covers space, `\t\n\r`, NBSP, FF, VT, Unicode spaces). Verify NBSP true. |
| `\w` | letter/digit/underscore (Unicode) | `scalar.properties.isAlphabetic \|\| (scalar.properties.numericType == .decimal) \|\| scalar == "_"` — but to match Python `\w` precisely, prefer: `CharacterSet.alphanumerics.contains(scalar) \|\| scalar == "_"`. Note Python `\w` also includes some Mn/Mc marks; post-normalize none survive, so this is moot. |
| `[^\W\d]` | letter or `_`, not digit | `(scalar.properties.isAlphabetic \|\| scalar == "_")` and NOT a decimal digit |
| `\d` | Unicode decimal digit | `scalar.properties.numericType == .decimal` (or `CharacterSet.decimalDigits`) |
| `str.lower()` | locale-independent lowercase | `String.lowercased()` (root locale). For span-finding, lowercase a COPY but slice from original (§3.1). |
| `str.isupper()` (per char) | uppercase letter | `scalar.properties.isUppercase` (or `Character.isUppercase`); for `fix_capital` use per-scalar. |
| `len(s)` | code-point count | `s.unicodeScalars.count` (NOT `.count`, NOT `.utf16.count`). |
| `s.find(sub, off)` | first code-point index ≥ off | scalar-index search over `s.unicodeScalars`. |

Concrete fixed sets to hardcode (do not derive from properties — they're tiny and exact):

- **Normalize allowed set** — see §1.3 (45-char class; the exact membership table §1.4).
- **Sentenize delimiters** — `.?!…;"„'»”’)]}` + `SMILES` regex `[=:;]-?[)(]{1,3}` (§2.2.1).
- **DASHES** (rule 11) — `‑–—−-` = U+2011, U+2013, U+2014, U+2212, U+002D.
- **ENDINGS** — `.?!…`. **CLOSE_QUOTES** `»”’`. **GENERIC_QUOTES** `"„'`. **QUOTES** `«“‘»”’"„'`.
  **CLOSE_BRACKETS** `)]}`. **BULLET_CHARS** `§абвгдеabcdef`. **BULLET_BOUNDS** `.)`.
- **Vowels** (`count_vowels`) — `аеёиоуыэюяАЕЁИОУЫЭЮЯ` (§4.2).
- **`has_punctuation` set** — full ASCII punct `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~` (§4.3).
- **`delete_spaces_before_punc` set** — same minus `+` and `~` (§4.1), `-` special-cased, final `~`→`-`.
- **sokr / pair-sokr / INITIALS** — §2.3 (transcribed; preserve the `жен р`/`муж р` dedup bug by
  omission).

---

## 8. Test checklist for the Swift port (acceptance fixtures)

1. **Normalize** — assert the §1.4 membership table char-by-char (esp. `"` U+0022 DELETED, `'` U+0027
   kept, `…` deleted vs `...` kept, `+` deleted, EN/MINUS dashes deleted, em-dash kept, Ukr Cyrillic
   deleted).
2. **Sentenize** — the §2.5 corpus (14 inputs) → exact spans. Plus dialogue dashes, smileys, list
   markers, initials, sokr/pair-sokr joins.
3. **split_by_sentences wrapper** — `"".join(result) == input` (losslessness) for all §2.5 inputs;
   gap-attaches-to-following-sentence; leading/trailing whitespace preserved.
4. **split_by_words** — the §3.4 examples (incl. `" - "`→`" ~ "`, `к+оса`, empty `"   "` →
   `([],["",""])`, punctuation grouping `?!`/`...`); `len(rem)==len(words)+1`; reconstruction equals
   `~`-substituted input.
5. **delete_spaces_before_punc** — single-space removal (not run-collapse), `-` before+after removal,
   `~`→`-` restore. count_vowels / has_punctuation char sets.
6. **fix_capital** — case transfer; length-mismatch passthrough.
7. **Reassembly** — off-by-one (trailing gap appended); `process_all` vs `process_yo` join/append
   differences.
8. **Alignment (§6)** — the §6.3 divergence sentences asserting upstream-parity **including** the
   mis-stress (`замок` left unstressed after `?!`); the `" - "` drift; and a fuzz pass hunting any
   `len(stress_usages) < len(words)` (IndexError) case.

> The golden oracle is the upstream `process_all_internal` output (run via `.venv` per CLAUDE.md
> "parity oracle"). Match it byte-for-byte (including bugs) for parity; deviations are a port defect.
