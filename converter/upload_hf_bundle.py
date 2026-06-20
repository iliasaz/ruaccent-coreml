#!/usr/bin/env python3
"""Assemble + upload the RUAccentCoreML on-device resource bundle to a (private) HF repo.

Mirrors how chatterbox-coreml hosts its weights on `iliasaz/chatterbox-turbo-coreml`:
large CoreML artifacts + packed resources live in a private Hugging Face model repo and the
Swift package downloads them on-device via swift-transformers `HubApi.snapshot` (see
`Sources/RUAccentCoreML/ModelRepository.swift`). This keeps THIS git repo binary-free.

The bundle layout matches `RUAccent(modelDirectory:)` exactly, so the downloaded snapshot dir
can be handed straight to that initializer:

    coreml/M1_accent_fp16.mlpackage/...
    coreml/M2_omograph_fp16_pal8.mlpackage/...
    coreml/M3_stress_fp16_pal8.mlpackage/...
    coreml/M4_yo_fp16.mlpackage/...
    dictpack/{accents,omographs,yo_words,yo_homographs}.rapack
    nn/nn_accent/vocab.txt
    nn/nn_stress_usage_predictor/vocab.txt
    nn/nn_yo_homograph_resolver/vocab.txt
    nn/nn_omograph/turbo3.1/{vocab.json,merges.txt,added_tokens.json}

Only the tokenizer files the Swift tokenizers actually read are staged (not the raw onnx /
tokenizer.json). Run packers/converters first if `_work` is empty:
    export HF_HOME=/Users/ilia/Downloads/HF_HOME
    .venv/bin/python converter/pack_dicts.py        # -> _work/dictpack/*.rapack
    # (and the convert_*_coreml.py scripts -> _work/coreml/*.mlpackage)

Usage:
    export HF_HOME=/Users/ilia/Downloads/HF_HOME            # machine is logged in as iliasaz
    .venv/bin/python converter/upload_hf_bundle.py                 # stage + create + upload
    .venv/bin/python converter/upload_hf_bundle.py --stage-only    # stage only, no network
    .venv/bin/python converter/upload_hf_bundle.py --repo-id iliasaz/ruaccent-coreml
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK = REPO_ROOT / "converter" / "_work"
STAGE = WORK / "hf_bundle"

DEFAULT_REPO_ID = "iliasaz/ruaccent-coreml"

# (source under _work)  ->  (dest under the bundle, == RUAccent modelDirectory layout)
MODELS = {
    "coreml/M1_accent_fp16.mlpackage":         "coreml/M1_accent_fp16.mlpackage",
    "coreml/M2_omograph_fp16_pal8.mlpackage":  "coreml/M2_omograph_fp16_pal8.mlpackage",
    "coreml/M3_stress_fp16_pal8.mlpackage":    "coreml/M3_stress_fp16_pal8.mlpackage",
    "coreml/M4_yo_fp16.mlpackage":             "coreml/M4_yo_fp16.mlpackage",
}
DICTS = {
    "dictpack/accents.rapack":        "dictpack/accents.rapack",
    "dictpack/omographs.rapack":      "dictpack/omographs.rapack",
    "dictpack/yo_words.rapack":       "dictpack/yo_words.rapack",
    "dictpack/yo_homographs.rapack":  "dictpack/yo_homographs.rapack",
}
# Only the tokenizer files the Swift tokenizers read (CharTokenizer/WordPieceTokenizer/ByteLevelBPE).
TOKENIZERS = {
    "nn/nn_accent/vocab.txt":                       "nn/nn_accent/vocab.txt",
    "nn/nn_stress_usage_predictor/vocab.txt":       "nn/nn_stress_usage_predictor/vocab.txt",
    "nn/nn_yo_homograph_resolver/vocab.txt":        "nn/nn_yo_homograph_resolver/vocab.txt",
    "nn/nn_omograph/turbo3.1/vocab.json":           "nn/nn_omograph/turbo3.1/vocab.json",
    "nn/nn_omograph/turbo3.1/merges.txt":           "nn/nn_omograph/turbo3.1/merges.txt",
    "nn/nn_omograph/turbo3.1/added_tokens.json":    "nn/nn_omograph/turbo3.1/added_tokens.json",
}

README = """\
---
license: apache-2.0
tags:
  - coreml
  - russian
  - text-to-speech
  - accentuation
  - ruaccent
---

# RUAccentCoreML — on-device resource bundle

CoreML port of [RUAccent](https://github.com/Den4ikAI/ruaccent) (Russian lexical stress,
variant `turbo3.1`) for the Swift package
[`RUAccentCoreML`](https://github.com/iliasaz/ruaccent-coreml). Downloaded on-device by
`ModelRepository.download()` (swift-transformers `HubApi.snapshot`) and consumed by
`RUAccent(modelDirectory:)`.

**Layout**
- `coreml/` — the four CoreML models (fp16; M2/M3 8-bit palettized):
  M1 accent (RoFormer), M2 omograph (BERT cross-encoder), M3 stress-usage (BERT),
  M4 yo-homograph (DistilBERT).
- `dictpack/` — packed dictionaries (`.rapack` v2, mmap-able): full `accents` (3.19M entries),
  `omographs`, `yo_words`, `yo_homographs`.
- `nn/` — the minimal tokenizer files (vocab/merges) the Swift tokenizers read.

Weights/dicts are the apache-2.0 HF `ruaccent/accentuator` artifacts, re-packaged. The pipeline
code is an independent reimplementation (the upstream Python is CC BY-NC-ND and was not copied).
"""


def stage() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    missing = []
    total = 0
    for group in (MODELS, DICTS, TOKENIZERS):
        for src_rel, dst_rel in group.items():
            src = WORK / src_rel
            dst = STAGE / dst_rel
            if not src.exists():
                missing.append(src_rel)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            total += sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) if dst.is_dir() else dst.stat().st_size
    (STAGE / "README.md").write_text(README, encoding="utf-8")
    if missing:
        print("!! MISSING source artifacts (run pack_dicts.py / convert_*_coreml.py first):", flush=True)
        for m in missing:
            print(f"     {m}", flush=True)
        return -1
    n_files = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    print(f">> staged {n_files} files, {total/1e6:.1f} MB -> {STAGE}", flush=True)
    return total


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami().get("name")
    print(f">> authenticated as: {who}", flush=True)
    print(f">> create_repo {repo_id} (private, exist_ok)...", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f">> upload_folder {STAGE} -> {repo_id} ...", flush=True)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=repo_id,
        repo_type="model",
        commit_message="RUAccentCoreML on-device bundle: 4 CoreML models + .rapack dicts + tokenizers",
    )
    print(f">> DONE. https://huggingface.co/{repo_id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--stage-only", action="store_true", help="assemble the bundle but do not touch the network")
    args = ap.parse_args()

    total = stage()
    if total < 0:
        sys.exit(1)
    if args.stage_only:
        print(">> --stage-only: skipping create/upload.", flush=True)
        return
    upload(args.repo_id)


if __name__ == "__main__":
    main()
