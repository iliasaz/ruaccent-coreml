# SWIFT_DICT_PACK — on-device dictionary pack format (`.rapack` v2)

Phase 3 deliverable (Agent D). Compact, **mmap-able, dependency-free** binary packs for the
4 RUAccent dictionaries, plus a Swift reader spec. Built by
[`converter/pack_dicts.py`](../converter/pack_dicts.py); outputs go to
`converter/_work/dictpack/*.rapack` (**gitignored — never commit the binaries**).

**Decision (locked):** ship the FULL `accents.json.gz` (3,194,879 entries) for exact parity,
not the small `accents_nn.json.gz` variant.

---

## 0. TL;DR for the Swift author

- Four files: `accents.rapack`, `omographs.rapack`, `yo_words.rapack`, `yo_homographs.rapack`.
- Each is a flat little-endian blob you `mmap` read-only and **binary-search** — no decompression,
  no third-party FST, no allocation at load time.
- Keys are stored once, sorted by UTF-8 bytes, with **bucketed front-coding** (shared-prefix
  compression). Lookup = binary search over bucket anchors → linear scan of ≤16 keys in one bucket.
- Values are **not** stored as strings; you reconstruct them from the key + a tiny payload:
  - `accents`: 1 byte = codepoint index where `+` goes (`0xFF` = no stress).
  - `yo_words` / `yo_homographs`: list of codepoint indices to flip `е`→`ё`.
  - `omographs`: the full variant strings (verbatim).
- **All lookups use a lowercased key** (upstream lowercases every word before dict lookup).

### Measured sizes (uncompressed = on-device footprint)

| pack | entries | bytes | MiB | gz(6) MiB |
|---|---:|---:|---:|---:|
| `accents.rapack` | 3,194,879 | 22,801,810 | **21.75** | 4.56 |
| `omographs.rapack` | 19,740 | 978,030 | 0.93 | 0.22 |
| `yo_words.rapack` | 85,568 | 1,050,163 | 1.00 | 0.22 |
| `yo_homographs.rapack` | 637 | 8,942 | 0.01 | 0.00 |
| **TOTAL** | | **24,838,945** | **23.69** | ~5.0 |

The packs ship uncompressed in the bundle so they can be `mmap`'d directly. The `gz` column is
informational: the App Store/IPA compresses on-the-wire, so download impact ≈ 5 MiB.
(For reference, raw `accents.json.gz` source is 20.95 MiB and must be fully gunzipped + JSON-parsed
into RAM — ~150 MB+ resident; the pack is mmap'd and touches only the pages a search walks.)

---

## 1. Data facts that justify the encoding

Measured from the source gz dicts (re-derivable by re-running the packer; the asserts in
`pack_dicts.py` enforce every one of these at build time, so a regression fails the build):

### accents (3,194,879 entries)
- Keys are **100% already lowercase** and **unique**; none contain `+`.
- `value.replace('+','') == key` for **100.0000%** of entries — the value is the key with a
  stress mark inserted and **nothing else changed**.
- `+` count per value: **3,194,816 have exactly 1**, **63 have 0** (value == key, e.g. `брр`,
  `аав`, `омск`, acronyms/interjections). **Never ≥2.**
- The `+` always immediately precedes a vowel (`аеёиоуыэюяАЕЁИОУЫЭЮЯ`) — 0 exceptions.
- Max `+` codepoint position = **54**; max key length = **56 codepoints / 112 UTF-8 bytes**.
  Max codepoint value = `U+0451` (ё) → all keys are BMP Cyrillic/Latin.
  → A stress position fits in **one byte**; `0xFF` is free as the "no stress" sentinel.

### yo_words (85,568) and yo_homographs (637)
- Keys 100% lowercase + unique. `len(key) == len(value)` always.
- `value == key` with the only differing positions being `е`→`ё`: **100.0000%**.
- Substitutions per entry: yo_words `{1: 85337, 2: 231}` (max 2); yo_homographs all exactly 1.
  → Store the codepoint indices of the substituted chars; one byte each.

### omographs (19,741 source → 19,740 packed)
- Value is `list[str]`, 1–4 variants/key (`{1: 3, 2: 19675, 3: 62, 4: 1}`); 39,543 variants total.
- Every variant is `key` with `+`(s) inserted: **100.0000%** strip-equal. `+` per variant
  `{1: 39539, 2: 4}`. **Some variants are irregular** (trailing `+` like `свеклУ+`, or mixed-case
  dead keys whose `+` does not precede a normal lowercase vowel). → Store **full variant strings
  verbatim** (cheap: ~706 KB) instead of positions, to guarantee byte-exact fidelity.
- **Runtime mutation replicated:** upstream `ruaccent.py:92` does
  `self.omographs.update({"коса": ["к+оса", "кос+а"]})` *after* loading the gz (the gz itself has
  `"коса": ["кос+а"]`). The packer applies the same override, so the pack ships `коса →
  ["к+оса","кос+а"]`. `custom_homographs` / `custom_dict` are NOT baked in (caller-supplied at
  runtime; handle those in the Swift layer above the pack).
- **5 keys in the source are mixed-case** (`икрИщ`, `сорвалИсь`, …). `split_by_words` lowercases
  every word before lookup, so a mixed-case key can NEVER match → they are dead. The packer
  lowercases all keys; the one lowercasing collision (`сорвалИсь` vs `сорвались`) is resolved in
  favor of the already-lowercase `сорвались` (the only reachable one), dropping the dead entry →
  19,740 packed.

> `accents_nn.json.gz` (110,826) and `accents.json.gz` (full) share the same value invariant; the
> same `KIND_STRESS` format packs either. We ship the full one.

---

## 2. On-disk format (`.rapack` v2)

Everything is **little-endian**. Offsets are absolute byte offsets from the start of the file
unless stated "relative". A pack is:

```
+-----------------------------+  0
| Header (40 bytes)           |
+-----------------------------+  bucketDir_off (= 40)
| bucketDir : u32[nBuckets]   |  each = offset of a bucket's anchor, RELATIVE to keyRegion_off
+-----------------------------+  keyRegion_off
| keyRegion : front-coded keys|
+-----------------------------+  payload_off
| payload region (by kind)    |
+-----------------------------+  fileSize (== file length)
```

### 2.1 Header (40 bytes)

| off | type | field | notes |
|---:|---|---|---|
| 0  | `u8[4]` | magic | ASCII `"RAPK"` (0x52 0x41 0x50 0x4B) |
| 4  | `u8` | version | `2` |
| 5  | `u8` | kind | `1`=STRESS(accents), `2`=OMOGRAPH, `3`=YO(yo_words & yo_homographs) |
| 6  | `u8` | bucket | entries per front-coding bucket = **16** |
| 7  | `u8` | reserved | 0 |
| 8  | `u32` | count | number of entries |
| 12 | `u32` | nBuckets | `ceil(count / bucket)` |
| 16 | `u32` | bucketDir_off | always 40 |
| 20 | `u32` | keyRegion_off | `40 + 4*nBuckets` |
| 24 | `u32` | payload_off | start of the payload region |
| 28 | `u32` | payloadBlob_off | STRESS: `== payload_off`; YO/OMOGRAPH: start of the var-length blob (after the offset table) |
| 32 | `u32` | fileSize | total file length (sanity check vs mmap length) |
| 36 | `u32` | reserved2 | 0 |

Swift sanity checks at open: `magic=="RAPK" && version==2 && fileSize==mappedLength && kind==expectedKind`.

### 2.2 bucketDir — `u32[nBuckets]`

`bucketDir[b]` = byte offset **relative to `keyRegion_off`** of bucket `b`'s anchor record. Bucket
`b` holds the entries with global index `b*16 .. min(b*16+16, count)-1`. The anchors are strictly
increasing in UTF-8 order, so you binary-search this directory.

### 2.3 keyRegion — bucketed front-coding

Keys are sorted by **UTF-8 byte order** (this is exactly `lhs.utf8` lexicographically compared to
`rhs.utf8` in Swift — a plain `[UInt8]` comparison; do **not** use `String <` which is Unicode-
canonical and would diverge). Within `keyRegion`, for each entry in order:

- **Anchor** (global index `i` with `i % 16 == 0`):
  `[uvarint keyLen][keyLen bytes of UTF-8]` — the full key.
- **Member** (any other entry):
  `[uvarint sharedPrefixLen][uvarint sufLen][sufLen bytes]`
  Reconstruct: `key = previousKeyInThisBucket[0 ..< sharedPrefixLen] + suffix`.
  `sharedPrefixLen` is measured in **bytes** against the immediately preceding key in the same
  bucket (the chain restarts at every anchor, which is why random access only needs to replay
  ≤15 members).

`uvarint` = unsigned LEB128: 7 bits/byte, low byte first, high bit set = "more bytes follow". In
these dicts all these lengths are < 128 so each is a single byte, but the reader must handle the
general case.

### 2.4 payload region

#### kind = STRESS (accents)
`payloadBlob_off == payload_off`. The region is a **flat array of `count` bytes**, indexed by
global entry index `i`:
- `b = blob[payload_off + i]`
- if `b == 0xFF` → value = key (no stress).
- else → value = key with a stress mark inserted **before the codepoint at codepoint-index `b`**,
  i.e. `value = String(key.unicodeScalars[..<b]) + MARK + String(key.unicodeScalars[b...])`.
  `b` is a **codepoint (Unicode scalar) index**, not a byte index.

> The upstream mark is the literal `'+'` placed *before* the stressed vowel. The integration
> contract emits `U+0301` *after* the vowel. Both are trivially derived from the same position `b`:
> the stressed vowel is the scalar at index `b`; insert `U+0301` right after it. Choose at the call
> site — the pack stores only the position.

#### kind = YO (yo_words, yo_homographs)
Region = `[payloadOffsets : u32[count+1]][payloadBlob]`. `payloadBlob_off = payload_off +
4*(count+1)`. Entry `i`'s payload = `payloadBlob[ off[i] ..< off[i+1] ]` where
`off = payloadOffsets`. Payload bytes: `[u8 nSubs][u8 pos]*nSubs`. Reconstruct: copy the key's
scalars, and for each `pos` replace scalar `pos` (`е`, `U+0435`) with `ё` (`U+0451`). (`nSubs` is
0..2 here; an empty payload `[0]` would mean "value == key", though no such YO entry exists.)

#### kind = OMOGRAPH
Same `[payloadOffsets : u32[count+1]][payloadBlob]` framing. Entry payload:
`[uvarint nVariants]` then per variant `[uvarint byteLen][byteLen UTF-8 bytes]`. The variant
strings are the final replacement strings (they already contain `+`); return them as `[String]`.

---

## 3. Lookup algorithm & cost

```
func index(of target: [UInt8]) -> Int?     // target = word.lowercased().utf8 as bytes
  # 1. binary search bucketDir for the rightmost bucket whose anchor <= target
  lo=0; hi=nBuckets-1; cand = -1
  while lo<=hi:
     mid=(lo+hi)/2
     anchor = readAnchorKey(mid)            # uvarint len + bytes at keyRegion_off+bucketDir[mid]
     if anchor <= target { cand=mid; lo=mid+1 } else { hi=mid-1 }
  if cand<0 { return nil }
  # 2. replay the bucket, comparing each reconstructed key
  base = cand*16
  prev = anchor(cand)
  if prev == target { return base }
  for j in 1..<min(16, count-base):
     (shared,suf) = nextMember()
     cur = prev[0..<shared] + suf
     if cur == target { return base+j }
     if cur >  target  { return nil }       # sorted ⇒ early out
     prev = cur
  return nil
```

- **Comparisons:** binary search is `O(log2(nBuckets))`; accents `nBuckets=199,680` ⇒ ~18 anchor
  reads, each a uvarint + a byte-slice compare. Then ≤16 member reconstructions in one bucket.
  Total per lookup: well under ~35 small reads, no allocation beyond a transient ≤112-byte buffer.
- **Anchor read for binary search is cheap** — you only need to *compare* bytes, you do not
  reconstruct anything until step 2.
- **mmap locality:** the bucketDir is contiguous (one region) and the bucket you land in is a
  contiguous ≤~600-byte run, so a hot lookup touches ~2–3 pages.

`get(word) -> Value?`: `index(of:)`, then read the payload for that index and reconstruct per §2.4.

---

## 4. Recommended Swift loader shape

```swift
struct RAPack {                       // one per .rapack, value type over an mmap
    let data: UnsafeRawBufferPointer  // from mmap(MAP_PRIVATE|read-only) of the bundle resource
    let count, nBuckets: Int
    let bucket: Int                   // 16
    let kind: UInt8
    let bucketDirOff, keyRegionOff, payloadOff, payloadBlobOff: Int

    init?(mmapping url: URL, expectedKind: UInt8)   // validate header, retain the mapping

    // returns the global entry index for a *lowercased* word, else nil
    func index(ofUTF8 target: UnsafeBufferPointer<UInt8>) -> Int?
}

// thin typed facades:
struct AccentsPack  { func stressPosition(of word: String) -> Int?  /* scalar idx, nil if absent */ }
struct YoPack       { func resolve(_ word: String) -> String?       /* nil if absent */ }
struct OmographPack { func variants(of word: String) -> [String]?   /* nil if absent */ }
```

Notes:
- Open with `mmap` (or `Data(contentsOf:options:.mappedIfSafe)` and keep the `Data` alive — it
  maps lazily). Do all reads through `withUnsafeBytes` / `loadUnaligned(fromByteOffset:as:)`.
  The format is **unaligned-safe**: never assume natural alignment for the u32 tables; use
  `loadUnaligned`.
- The comparison key is `word.lowercased()` then its `.utf8`. Lowercase with the same semantics as
  Python `str.lower()` for Cyrillic (Swift `String.lowercased()` matches for the Russian alphabet,
  including `Ё`→`ё`). Keep an apostrophe/edge-case test against the golden fixtures.
- For STRESS, return the **scalar index**; the caller decides `+`-before vs `U+0301`-after and
  whether to apply (manual stress in the input always wins — handle above the pack).
- The packs are immutable; `custom_dict`/`custom_homographs` overrides live in a small in-memory
  dictionary consulted *before* the pack (mirrors `accents.update(custom_dict)` ordering).
- Endianness: format is LE; all current Apple targets are LE so reads are direct. If you ever need
  BE, byte-swap the u32s on read (documented here for completeness).

---

## 5. Validation (what the packer proves)

`pack_dicts.py` reloads each pack through a pure-Python mirror of the Swift reader and checks
`key → value` fidelity against the source gz:

- `omographs`, `yo_words`, `yo_homographs`: **every** entry, 100.000000% match.
- `accents`: a 500,000-key random sample **plus all 63 non-`1-plus` entries**, 100.000000% match.
  A separate full pass over all **3,194,879** entries also reports **0 mismatches** (run:
  the one-off in the agent log / re-derivable). Absent-key probes return `None`.

Re-run any time:
```bash
export HF_HOME=/Users/ilia/Downloads/HF_HOME
/Users/ilia/Developer/ruaccent-coreml/.venv/bin/python converter/pack_dicts.py
```

---

## 6. Open items / notes for integration

- **Sort key parity is load-bearing.** The pack is ordered by raw UTF-8 bytes. The Swift reader
  MUST compare `[UInt8]` (utf8), not `String`. A golden test should assert that for a handful of
  near-collision Cyrillic keys the Swift binary search finds the same entry as a linear scan.
- **`str.lower()` vs `String.lowercased()`** for exotic inputs (Latin look-alikes such as the key
  `лoшаp`, which mixes Cyrillic and Latin `o`/`p`): such keys exist verbatim in `accents`. The
  Swift side must lowercase identically to Python for the lookup to hit. Add a fixture.
- The `letters_accent` map (`{'о':'+о','О':'+О'}`) and `custom_dict` are applied *after* the pack
  in upstream `load()`; replicate that override order in the Swift `RUAccent` layer, not in the
  pack.
- Versioned by the header `version` byte; bump it if the layout changes so a stale bundled pack
  fails fast instead of mis-decoding.
