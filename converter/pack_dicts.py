#!/usr/bin/env python3
"""WS-D (Agent D, Phase 3): pack the 4 RUAccent dictionaries into compact, mmap-able,
dependency-free binary packs for the Swift reader, and round-trip validate them.

Decision (locked): ship the FULL `accents.json.gz` (3.19M entries) for exact parity, NOT
`accents_nn.json.gz`.

Source dicts (gzipped JSON) in converter/_work/dictionary/:
  - accents.json.gz       str(lowercased key) -> str (key with exactly 0 or 1 '+' before a vowel)
  - omographs.json.gz     str -> list[str] variants (full strings, each = key with '+' inserted)
  - yo_words.json.gz      str -> str (key with some е->ё)
  - yo_homographs.json.gz str -> str (key with some е->ё)

Verified facts that drive the encoding (numbers in docs/SWIFT_DICT_PACK.md):
  - accents: value.replace('+','') == key for 100% of 3,194,879 entries; <=1 '+'; '+' always
    precedes a vowel; max '+' codepoint position = 54 (< 255). 63 entries have 0 '+' (value==key).
    => store ONLY the stress position as 1 flat byte/entry (codepoint index of '+'; 0xFF = none).
  - yo_words (85,568) / yo_homographs (637): value == key with only е->ё substitutions, length
    preserved; <=2 subs. => store ONLY the codepoint indices of substituted chars.
  - omographs (19,740 after runtime override): 1..4 variants/key, each variant = key with '+'
    inserted; a few irregular (trailing '+', mixed-case). => store FULL variant strings verbatim.

KEYS for all four dicts are stored with BUCKETED FRONT-CODING (shared-prefix compression that
still supports binary search): sorted UTF-8 keys are grouped into buckets of B; the first key of
each bucket ("anchor") is stored in full, the rest store [sharedPrefixLen][suffix] vs the previous
key in the bucket. Binary-search the anchors, then linearly reconstruct within one bucket.
This shrinks the accents key region from 71.8 MB (raw) to ~17 MB.

Output (gitignored): converter/_work/dictpack/{accents,omographs,yo_words,yo_homographs}.rapack

Run:
  export HF_HOME=/Users/ilia/Downloads/HF_HOME
  /Users/ilia/Developer/ruaccent-coreml/.venv/bin/python converter/pack_dicts.py
"""
import gzip
import json
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_DIR = os.path.join(ROOT, "converter", "_work", "dictionary")
OUT_DIR = os.path.join(ROOT, "converter", "_work", "dictpack")

MAGIC = b"RAPK"          # RuAccent PacK
VERSION = 2              # v2 = bucketed front-coded keys

# Pack "kind" tags (1 byte).
KIND_STRESS = 1          # accents: payload = 1 flat byte stress position (0xFF = none)
KIND_OMOGRAPH = 2        # omographs: payload = list of full variant strings (var-length blob)
KIND_YO = 3              # yo_*: payload = list of substitution positions (var-length blob)

NO_STRESS = 0xFF
BUCKET = 16              # entries per front-coding bucket (worst-case linear scan length)

VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_gz(name):
    with gzip.open(os.path.join(DICT_DIR, name)) as fh:
        return json.load(fh)


def sort_keys_by_utf8(keys):
    """Sort keys by their UTF-8 byte sequence — the exact order the Swift reader
    binary-searches (Swift `String.utf8` is compared lexicographically)."""
    return sorted(keys, key=lambda s: s.encode("utf-8"))


def lower_dedup(items):
    """Lowercase every key. On a lowercasing collision prefer the entry whose
    ORIGINAL key was already all-lowercase (the only one upstream can match, since
    split_by_words lowercases the whole string). Returns dict(lower_key -> value)."""
    out = {}
    for k, v in items:
        lk = k.lower()
        if lk in out:
            if k == lk:
                out[lk] = v
            # else drop the mixed-case dead entry
        else:
            out[lk] = v
    return out


# ---------------------------------------------------------------------------
# uvarint (LEB128 unsigned)
# ---------------------------------------------------------------------------
def write_uvarint(buf, n):
    assert n >= 0
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def read_uvarint(mv, pos):
    shift = 0
    result = 0
    while True:
        b = mv[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


# ---------------------------------------------------------------------------
# pack layout (little-endian throughout)
# ---------------------------------------------------------------------------
# A .rapack file is:
#   [Header 40 bytes]
#   [bucketDir : nBuckets * u32 LE]   # byte offset (rel. to keyRegion start) of each bucket anchor
#   [keyRegion : front-coded keys]    # see below
#   [payload region: kind-specific, see below]
#
# keyRegion, entries in sorted order, grouped in buckets of BUCKET:
#   anchor entry (i % BUCKET == 0):  [uvarint keyLen][keyLen UTF-8 bytes]        (full key)
#   member entry:                    [uvarint sharedPrefixLen][uvarint sufLen][sufLen bytes]
#     reconstruct: key = prevKeyInBucket[:sharedPrefixLen] + suffix   (prefixLen in BYTES)
#
# payload region:
#   KIND_STRESS : flat `count` bytes — payload[i] = stress codepoint pos (0xFF = none). O(1) index.
#   KIND_YO / KIND_OMOGRAPH : [payloadOffsets : (count+1) * u32 LE][payloadBlob]
#     payloadBlob entry i = payloadBlob[off[i]:off[i+1]]
#       KIND_YO       : [u8 nSubs][u8 codepointPos]*nSubs
#       KIND_OMOGRAPH : [uvarint nVariants] then per variant [uvarint byteLen][UTF-8 bytes]
#
# Header:
#   0  : 4s  magic "RAPK"
#   4  : u8  version (2)
#   5  : u8  kind
#   6  : u8  bucket (BUCKET)
#   7  : u8  reserved (0)
#   8  : u32 count
#   12 : u32 nBuckets
#   16 : u32 bucketDir_off   (= 40)
#   20 : u32 keyRegion_off
#   24 : u32 payload_off      # start of payload region (flat bytes OR offset table)
#   28 : u32 payloadBlob_off  # KIND_STRESS: == payload_off; YO/OMOGRAPH: start of blob (after offsets)
#   32 : u32 file_size
#   36 : u32 reserved2 (0)
HEADER_FMT = "<4sBBBBIIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 40, HEADER_SIZE


def build_key_region(sorted_keys, bucket):
    key_bytes = [k.encode("utf-8") for k in sorted_keys]
    region = bytearray()
    bucket_dir = []
    prev = b""
    for i, kb in enumerate(key_bytes):
        if i % bucket == 0:
            bucket_dir.append(len(region))
            write_uvarint(region, len(kb))
            region.extend(kb)
            prev = kb
        else:
            m = 0
            mx = min(len(kb), len(prev))
            while m < mx and kb[m] == prev[m]:
                m += 1
            suf = kb[m:]
            write_uvarint(region, m)
            write_uvarint(region, len(suf))
            region.extend(suf)
            prev = kb
    return bytes(region), bucket_dir


def assemble(kind, sorted_keys, payload_region, payload_blob_rel):
    """payload_region: bytes for the whole payload region.
    payload_blob_rel: offset WITHIN payload_region where the variable blob starts
                      (0 for KIND_STRESS; (count+1)*4 for YO/OMOGRAPH)."""
    count = len(sorted_keys)
    key_region, bucket_dir = build_key_region(sorted_keys, BUCKET)
    n_buckets = len(bucket_dir)

    bucket_dir_off = HEADER_SIZE
    key_region_off = bucket_dir_off + 4 * n_buckets
    payload_off = key_region_off + len(key_region)
    payload_blob_off = payload_off + payload_blob_rel
    file_size = payload_off + len(payload_region)

    out = bytearray(file_size)
    struct.pack_into(
        HEADER_FMT, out, 0,
        MAGIC, VERSION, kind, BUCKET, 0, count, n_buckets,
        bucket_dir_off, key_region_off, payload_off, payload_blob_off, file_size, 0,
    )
    struct.pack_into(f"<{n_buckets}I", out, bucket_dir_off, *bucket_dir)
    out[key_region_off:key_region_off + len(key_region)] = key_region
    out[payload_off:payload_off + len(payload_region)] = payload_region
    return bytes(out)


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------
def stress_pos(key, value):
    if "+" not in value:
        return NO_STRESS
    pos = value.index("+")
    assert pos < NO_STRESS, (key, value, pos)
    return pos


def yo_subs(key, value):
    assert len(key) == len(value), (key, value)
    diffs = [i for i in range(len(key)) if key[i] != value[i]]
    for i in diffs:
        assert key[i] == "е" and value[i] == "ё", (key, value, i)
    assert not diffs or max(diffs) < 256
    return diffs


def build_stress_payload(sorted_keys, src):
    return bytes(stress_pos(k, src[k]) for k in sorted_keys), 0


def build_yo_payload(sorted_keys, src):
    blob = bytearray()
    offs = [0]
    for k in sorted_keys:
        diffs = yo_subs(k, src[k])
        blob.append(len(diffs))
        blob.extend(diffs)
        offs.append(len(blob))
    count = len(sorted_keys)
    region = bytearray()
    region.extend(struct.pack(f"<{count + 1}I", *offs))
    blob_rel = len(region)
    region.extend(blob)
    return bytes(region), blob_rel


def build_omograph_payload(sorted_keys, src):
    blob = bytearray()
    offs = [0]
    for k in sorted_keys:
        variants = src[k]
        write_uvarint(blob, len(variants))
        for var in variants:
            vb = var.encode("utf-8")
            write_uvarint(blob, len(vb))
            blob.extend(vb)
        offs.append(len(blob))
    count = len(sorted_keys)
    region = bytearray()
    region.extend(struct.pack(f"<{count + 1}I", *offs))
    blob_rel = len(region)
    region.extend(blob)
    return bytes(region), blob_rel


# ---------------------------------------------------------------------------
# pure-python pack reader (mirror of the Swift mmap reader) — used to validate
# ---------------------------------------------------------------------------
class Pack:
    def __init__(self, blob):
        self.b = blob
        (magic, version, self.kind, self.bucket, _r, self.count, self.n_buckets,
         self.bucket_dir_off, self.key_region_off, self.payload_off,
         self.payload_blob_off, self.file_size, _r2) = struct.unpack_from(HEADER_FMT, blob, 0)
        assert magic == MAGIC and version == VERSION, (magic, version)
        assert self.file_size == len(blob), (self.file_size, len(blob))
        self.bucket_dir = struct.unpack_from(f"<{self.n_buckets}I", blob, self.bucket_dir_off)
        if self.kind in (KIND_YO, KIND_OMOGRAPH):
            self.pl_off = struct.unpack_from(
                f"<{self.count + 1}I", blob, self.payload_off)

    # ---- key reconstruction within a bucket ----
    def _anchor_key(self, bi):
        pos = self.key_region_off + self.bucket_dir[bi]
        klen, pos = read_uvarint(self.b, pos)
        return bytes(self.b[pos:pos + klen]), pos + klen

    def _bucket_keys(self, bi):
        """Reconstruct all (UTF-8 bytes) keys in bucket bi, plus their global indices."""
        start = bi * self.bucket
        end = min(start + self.bucket, self.count)
        pos = self.key_region_off + self.bucket_dir[bi]
        klen, pos = read_uvarint(self.b, pos)
        prev = bytes(self.b[pos:pos + klen]); pos += klen
        keys = [prev]
        for _ in range(start + 1, end):
            shared, pos = read_uvarint(self.b, pos)
            suflen, pos = read_uvarint(self.b, pos)
            suf = bytes(self.b[pos:pos + suflen]); pos += suflen
            cur = prev[:shared] + suf
            keys.append(cur)
            prev = cur
        return keys, start

    def index_of(self, key_str):
        target = key_str.encode("utf-8")
        # binary search over bucket anchors
        lo, hi = 0, self.n_buckets - 1
        # find rightmost bucket whose anchor <= target
        cand = -1
        while lo <= hi:
            mid = (lo + hi) >> 1
            anchor, _ = self._anchor_key(mid)
            if anchor <= target:
                cand = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if cand < 0:
            return -1
        keys, base = self._bucket_keys(cand)
        for j, kb in enumerate(keys):
            if kb == target:
                return base + j
            if kb > target:
                break
        return -1

    def get_value(self, key_str):
        i = self.index_of(key_str)
        if i < 0:
            return None
        bi = i // self.bucket
        keys, base = self._bucket_keys(bi)
        key = keys[i - base].decode("utf-8")
        if self.kind == KIND_STRESS:
            pos = self.b[self.payload_off + i]
            if pos == NO_STRESS:
                return key
            return key[:pos] + "+" + key[pos:]
        if self.kind == KIND_YO:
            a = self.payload_blob_off + self.pl_off[i]
            n = self.b[a]
            chars = list(key)
            for t in range(n):
                p = self.b[a + 1 + t]
                assert chars[p] == "е"
                chars[p] = "ё"
            return "".join(chars)
        if self.kind == KIND_OMOGRAPH:
            a = self.payload_blob_off + self.pl_off[i]
            nvar, a = read_uvarint(self.b, a)
            out = []
            for _ in range(nvar):
                blen, a = read_uvarint(self.b, a)
                out.append(bytes(self.b[a:a + blen]).decode("utf-8"))
                a += blen
            return out
        raise ValueError(self.kind)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_stress_pack(src):
    keys = sort_keys_by_utf8(src.keys())
    region, blob_rel = build_stress_payload(keys, src)
    return assemble(KIND_STRESS, keys, region, blob_rel)


def build_yo_pack(src):
    keys = sort_keys_by_utf8(src.keys())
    region, blob_rel = build_yo_payload(keys, src)
    return assemble(KIND_YO, keys, region, blob_rel)


def build_omograph_pack(src):
    keys = sort_keys_by_utf8(src.keys())
    region, blob_rel = build_omograph_payload(keys, src)
    return assemble(KIND_OMOGRAPH, keys, region, blob_rel)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_full(name, src, pack, eq):
    pk = Pack(pack)
    assert pk.count == len(src), (pk.count, len(src))
    mism = 0
    examples = []
    for k, v in src.items():
        got = pk.get_value(k)
        if not eq(v, got):
            mism += 1
            if len(examples) < 5:
                examples.append((k, v, got))
    assert pk.get_value("zzz_not_a_key_zzz") is None
    rate = 100.0 * (len(src) - mism) / len(src)
    print(f"  [{name}] entries={len(src)} mismatches={mism} match_rate={rate:.6f}%")
    for e in examples:
        print(f"      MISMATCH key={e[0]!r} expected={e[1]!r} got={e[2]!r}")
    return mism == 0


def validate_sample(name, src, pack, eq, sample_n, force_keys):
    pk = Pack(pack)
    assert pk.count == len(src), (pk.count, len(src))
    keys = list(src.keys())
    rnd = random.Random(0xACCE)
    sample = set(rnd.sample(keys, min(sample_n, len(keys))))
    sample |= set(force_keys)
    mism = 0
    examples = []
    for k in sample:
        got = pk.get_value(k)
        if not eq(src[k], got):
            mism += 1
            if len(examples) < 5:
                examples.append((k, src[k], got))
    assert pk.get_value("zzz_not_a_key_zzz") is None
    rate = 100.0 * (len(sample) - mism) / len(sample)
    print(f"  [{name}] sampled={len(sample)} (incl {len(force_keys)} forced) "
          f"mismatches={mism} match_rate={rate:.6f}%")
    for e in examples:
        print(f"      MISMATCH key={e[0]!r} expected={e[1]!r} got={e[2]!r}")
    return mism == 0


def fmt_bytes(n):
    return f"{n:,} B  ({n / 1024:.1f} KiB / {n / 1024 / 1024:.2f} MiB)"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== loading source dicts ==")
    accents = load_gz("accents.json.gz")
    omographs_raw = load_gz("omographs.json.gz")
    yo_words = load_gz("yo_words.json.gz")
    yo_homographs = load_gz("yo_homographs.json.gz")
    print(f"  accents={len(accents)} omographs={len(omographs_raw)} "
          f"yo_words={len(yo_words)} yo_homographs={len(yo_homographs)}")

    # replicate ruaccent runtime mutation: omographs.update({"коса":["к+оса","кос+а"]})
    # then lowercase-dedup (drops the 1 mixed-case dead key 'сорвалИсь').
    omographs = lower_dedup(omographs_raw.items())
    omographs["коса"] = ["к+оса", "кос+а"]
    print(f"  omographs after runtime override + lower-dedup: {len(omographs)} "
          f"(коса -> {omographs['коса']})")

    print(f"\n== building packs (bucketed front-coding, B={BUCKET}) ==")
    pk_acc = build_stress_pack(accents)
    pk_omo = build_omograph_pack(omographs)
    pk_yow = build_yo_pack(yo_words)
    pk_yoh = build_yo_pack(yo_homographs)

    out = {
        "accents.rapack": pk_acc,
        "omographs.rapack": pk_omo,
        "yo_words.rapack": pk_yow,
        "yo_homographs.rapack": pk_yoh,
    }
    for fn, blob in out.items():
        with open(os.path.join(OUT_DIR, fn), "wb") as fh:
            fh.write(blob)

    print("\n== round-trip validation ==")
    ok = True
    ok &= validate_full("omographs", omographs, pk_omo, lambda a, b: a == b)
    ok &= validate_full("yo_words", yo_words, pk_yow, lambda a, b: a == b)
    ok &= validate_full("yo_homographs", yo_homographs, pk_yoh, lambda a, b: a == b)
    force = [k for k, v in accents.items() if v.count("+") != 1]   # all 63 no-stress entries
    ok &= validate_sample("accents", accents, pk_acc, lambda a, b: a == b,
                           sample_n=500_000, force_keys=force)

    print("\n== pack sizes (uncompressed, on-disk == on-device footprint) ==")
    total = 0
    for fn, blob in out.items():
        total += len(blob)
        gz = len(gzip.compress(blob, 6))
        print(f"  {fn:22} {fmt_bytes(len(blob))}   [gz {gz / 1024 / 1024:.2f} MiB]")
    print(f"  {'TOTAL':22} {fmt_bytes(total)}")
    print(f"\n  packs written to: {OUT_DIR}")
    print("\nRESULT:", "ALL PACKS VALID (100% round-trip)" if ok else "VALIDATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
