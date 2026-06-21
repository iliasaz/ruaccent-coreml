#!/usr/bin/env python3
"""WS-D (issue #11): convert M4 yo_homograph (DistilBertForTokenClassification) -> CoreML fp16.

2nd conversion (after M1) — locks the BERT-family recipe. Source = fp32 model.onnx (already fp32).
Runtime path feeds ONE razdel sentence with NO padding => all-ones mask => the additive attention
mask is all-zeros => fp16-safe with NO mask patch needed. We convert with a FLEXIBLE seq dim
(ct.RangeDim) and validate at each oracle's exact length (all-ones mask).

Gates (Mac CPU):
  1. torch(rebuilt) vs onnxruntime(model.onnx)   expect ~1e-4
  2. coreml fp16 vs onnxruntime                   expect logit |Δ|<~5e-2 AND per-token argmax identical

Run: /Users/ilia/Developer/chatterbox-coreml/.venv/bin/python converter/convert_yo_coreml.py
"""
import os, json, glob
import numpy as np
import onnxruntime as ort
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_extract import extract_state_dict, load_into

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M4 = os.path.join(ROOT, "converter", "_work", "nn", "nn_yo_homograph_resolver")
OUTDIR = os.path.join(ROOT, "converter", "_work", "coreml")
os.makedirs(OUTDIR, exist_ok=True)
REFS = sorted(glob.glob(os.path.join(ROOT, "converter", "fixtures", "refs", "M4_yo__*.npz")))


def main():
    import torch
    from transformers import DistilBertConfig, DistilBertForTokenClassification
    import coremltools as ct

    cfg = json.load(open(os.path.join(M4, "config.json")))
    config = DistilBertConfig(**cfg)
    config._attn_implementation = "eager"          # traceable / ANE-friendly (no sdpa/flash)
    print(f">> DistilBert: dim={config.dim} layers={config.n_layers} heads={config.n_heads} "
          f"vocab={config.vocab_size} labels={config.id2label}")

    sd_np, wmap = extract_state_dict(os.path.join(M4, "model.onnx"))
    model = DistilBertForTokenClassification(config).eval()
    rep = load_into(model, sd_np)
    print(f">> mapped {len(wmap)} Linear weights by bias-adjacency; "
          f"loaded {rep['loaded']}/{rep['total']} params; missing={rep['missing'][:6]} "
          f"skipped={rep['shape_skipped'][:3]}")

    sess = ort.InferenceSession(os.path.join(M4, "model.onnx"), providers=["CPUExecutionProvider"])

    # ---- GATE 1: torch vs onnxruntime ----
    print("\n>> GATE 1 (torch vs onnxruntime), exact length / all-ones mask:")
    max1 = 0.0
    for r in REFS:
        z = np.load(r); ids = z["input_ids"].astype(np.int64); am = np.ones_like(ids)
        on = sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
        with torch.no_grad():
            tl = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(am)).logits.numpy()
        d = float(np.abs(tl - on).max()); max1 = max(max1, d)
        print(f"   {os.path.basename(r):16} len={ids.shape[1]:3} |Δ|={d:.2e}")
    print(f">> GATE 1 max|Δ|={max1:.2e}  ({'PASS' if max1 < 1e-3 else 'CHECK'})")

    # ---- trace (flexible seq via RangeDim) + convert fp32/fp16 ----
    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits
    wrap = Wrap(model).eval()
    S0 = int(np.load(REFS[-1])["input_ids"].shape[1])         # representative length
    ex = (torch.zeros(1, S0, dtype=torch.long), torch.ones(1, S0, dtype=torch.long))
    # transformers 5.2's create_bidirectional_mask emits coremltools-unsupported ops (new_ones).
    # Runtime always uses all-ones masks (no padding) => no masking needed => return None
    # (numerically identical for the all-ones path). Patch AFTER GATE 1 so GATE 1 keeps real masking.
    import transformers.models.distilbert.modeling_distilbert as mdb
    mdb.create_bidirectional_mask = lambda *a, **k: None
    print(f"\n>> tracing at S0={S0}, converting with RangeDim(4,512) ...")
    traced = torch.jit.trace(wrap, ex)
    seq = ct.RangeDim(lower_bound=4, upper_bound=512, default=S0)

    def convert(precision, label):
        m = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32)],
            outputs=[ct.TensorType(name="logits")],
            compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18)
        p = os.path.join(OUTDIR, f"M4_yo_{label}.mlpackage"); m.save(p); return m, p

    def gate2(m, label):
        print(f"\n>> GATE 2 (coreml {label} vs onnxruntime):")
        mx, amok = 0.0, True
        for r in REFS:
            z = np.load(r); ids = z["input_ids"].astype(np.int64); L = ids.shape[1]; am = np.ones_like(ids)
            on = sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
            cm = m.predict({"input_ids": ids.astype(np.int32), "attention_mask": am.astype(np.int32)})["logits"]
            d = float(np.abs(cm - on).max()); mx = max(mx, d)
            a = bool((cm.argmax(-1) == on.argmax(-1)).all()); amok &= a
            print(f"   {os.path.basename(r):16} len={L:3} |Δ|={d:.2e} argmax{'=OK' if a else '!!MISMATCH'}")
        ok = mx < 5e-2 and amok
        print(f">> GATE 2 [{label}] max|Δ|={mx:.2e} argmax_all={amok}  ({'PASS' if ok else 'CHECK'})")
        return ok

    m32, _ = convert(ct.precision.FLOAT32, "fp32"); gate2(m32, "fp32")
    m16, p16 = convert(ct.precision.FLOAT16, "fp16"); gate2(m16, "fp16")
    print(f"\n>> M4 conversion done. fp16 package: {p16}")


if __name__ == "__main__":
    main()
