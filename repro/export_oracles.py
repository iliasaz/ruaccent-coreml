#!/usr/bin/env python3
"""Export per-model CoreML probe oracles (ready-to-feed inputs + expected argmax) from the
onnxruntime logit refs. Consumed by the Swift RuaccentProbe to check parity across compute units.

  repro/oracles.json : { model: {mlpackage, seq_fixed|null, inputs[], cases:[{ids/mask/tt, expected_argmax, label}]} }

Run: python repro/export_oracles.py
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "converter", "fixtures", "refs")
MAN = {m["tag"]: m for m in json.load(open(os.path.join(REFS, "manifest.json")))}

SPEC = {
    "M1_accent":   dict(pkg="M1_accent_fp16.mlpackage",   inputs=["input_ids", "attention_mask", "token_type_ids"], seq_fixed=48),
    "M4_yo":       dict(pkg="M4_yo_fp16.mlpackage",       inputs=["input_ids", "attention_mask"],                   seq_fixed=None),
    "M3_stress":   dict(pkg="M3_stress_fp16.mlpackage",   inputs=["input_ids", "attention_mask", "token_type_ids"], seq_fixed=None),
    "M2_omograph": dict(pkg="M2_omograph_fp16.mlpackage", inputs=["input_ids", "attention_mask"],                   seq_fixed=None),
}
PREFIX = {"M1_accent": "M1_accent__", "M4_yo": "M4_yo__", "M3_stress": "M3_stress__", "M2_omograph": "M2_omograph__"}


def main():
    out = {}
    for model, spec in SPEC.items():
        cases = []
        for r in sorted(glob.glob(os.path.join(REFS, f"{PREFIX[model]}*.npz"))):
            z = np.load(r); ids = z["input_ids"].astype(int)[0]; L = len(ids)
            am = (z["attention_mask"][0].astype(int) if "attention_mask" in z else np.ones(L, int))
            tt = (z["token_type_ids"][0].astype(int) if "token_type_ids" in z else np.zeros(L, int))
            logits = z["logits"]                       # onnxruntime oracle
            tag = os.path.basename(r)[:-4]
            label = MAN.get(tag, {}).get("word") or MAN.get(tag, {}).get("hypothesis") or tag
            S = spec["seq_fixed"]
            if S:                                       # pad to fixed length (M1)
                pad = S - L
                ids = np.pad(ids, (0, pad)); am = np.pad(am, (0, pad)); tt = np.pad(tt, (0, pad))
            case = {"label": label, "real_len": int(L),
                    "input_ids": ids.tolist(), "attention_mask": am.tolist()}
            if "token_type_ids" in spec["inputs"]:
                case["token_type_ids"] = tt.tolist()
            if logits.ndim == 3:                        # token-classifier: per-token argmax over real tokens
                case["expected_argmax"] = logits[0][:L].argmax(-1).astype(int).tolist()
            else:                                       # M2 binary decision
                case["expected_argmax"] = [int(logits[0].argmax())]
            cases.append(case)
        out[model] = {"mlpackage": spec["pkg"], "inputs": spec["inputs"],
                      "seq_fixed": spec["seq_fixed"], "cases": cases}
        print(f"{model}: {len(cases)} cases")
    json.dump(out, open(os.path.join(ROOT, "repro", "oracles.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote repro/oracles.json")


if __name__ == "__main__":
    main()
