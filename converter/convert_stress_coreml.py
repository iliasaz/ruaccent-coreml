#!/usr/bin/env python3
"""WS-C (issue #10): convert M3 stress_usage (BertForTokenClassification) -> CoreML fp16.

3rd conversion. Vanilla BERT (absolute pos, exact-erf GELU, eps 1e-12). Inputs include
token_type_ids (all-zeros, single segment). Runtime feeds ONE razdel sentence, no padding =>
all-ones mask => fp16-safe; neutralize transformers-5.2 create_bidirectional_mask (new_ones) like M4.

Gates (Mac CPU): 1) torch vs onnxruntime ~1e-4; 2) coreml fp16 vs onnxruntime: |Δ|<~5e-2 + argmax identical.
Run: /Users/ilia/Developer/chatterbox-coreml/.venv/bin/python converter/convert_stress_coreml.py
"""
import os, json, glob, sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_extract import extract_state_dict, load_into

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3 = os.path.join(ROOT, "converter", "_work", "nn", "nn_stress_usage_predictor")
OUTDIR = os.path.join(ROOT, "converter", "_work", "coreml"); os.makedirs(OUTDIR, exist_ok=True)
REFS = sorted(glob.glob(os.path.join(ROOT, "converter", "fixtures", "refs", "M3_stress__*.npz")))


def feed(z):
    ids = z["input_ids"].astype(np.int64)
    return {"input_ids": ids,
            "attention_mask": (z["attention_mask"] if "attention_mask" in z else np.ones_like(ids)).astype(np.int64),
            "token_type_ids": (z["token_type_ids"] if "token_type_ids" in z else np.zeros_like(ids)).astype(np.int64)}


def main():
    import torch
    from transformers import BertConfig, BertForTokenClassification
    import coremltools as ct

    cfg = json.load(open(os.path.join(M3, "config.json")))
    config = BertConfig(**cfg); config._attn_implementation = "eager"
    print(f">> BERT: {config.num_hidden_layers}L/{config.hidden_size}h vocab={config.vocab_size} "
          f"max_pos={config.max_position_embeddings} labels={config.id2label}")

    sd_np, wmap = extract_state_dict(os.path.join(M3, "model.onnx"))
    model = BertForTokenClassification(config).eval()
    rep = load_into(model, sd_np)
    print(f">> mapped {len(wmap)} Linear weights; loaded {rep['loaded']}/{rep['total']} "
          f"params; missing={rep['missing'][:6]} skipped={rep['shape_skipped'][:3]}")

    sess = ort.InferenceSession(os.path.join(M3, "model.onnx"), providers=["CPUExecutionProvider"])
    print("\n>> GATE 1 (torch vs onnxruntime):")
    max1 = 0.0
    for r in REFS:
        f = feed(np.load(r)); on = sess.run(None, f)[0]
        with torch.no_grad():
            tl = model(input_ids=torch.tensor(f["input_ids"]), attention_mask=torch.tensor(f["attention_mask"]),
                       token_type_ids=torch.tensor(f["token_type_ids"])).logits.numpy()
        d = float(np.abs(tl - on).max()); max1 = max(max1, d)
        print(f"   {os.path.basename(r):16} len={f['input_ids'].shape[1]:3} |Δ|={d:.2e}")
    print(f">> GATE 1 max|Δ|={max1:.2e}  ({'PASS' if max1 < 1e-3 else 'CHECK'})")

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
    wrap = Wrap(model).eval()
    S0 = int(np.load(REFS[-1])["input_ids"].shape[1])
    ex = (torch.zeros(1, S0, dtype=torch.long), torch.ones(1, S0, dtype=torch.long), torch.zeros(1, S0, dtype=torch.long))
    import transformers.models.bert.modeling_bert as mbert
    if hasattr(mbert, "create_bidirectional_mask"):
        mbert.create_bidirectional_mask = lambda *a, **k: None
    print(f"\n>> tracing at S0={S0}, RangeDim(4,512) ...")
    traced = torch.jit.trace(wrap, ex)
    seq = ct.RangeDim(lower_bound=4, upper_bound=512, default=S0)

    def convert(precision, label):
        m = ct.convert(traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="token_type_ids", shape=(1, seq), dtype=np.int32)],
            outputs=[ct.TensorType(name="logits")], compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.iOS18)
        p = os.path.join(OUTDIR, f"M3_stress_{label}.mlpackage"); m.save(p); return m, p

    def gate2(m, label):
        print(f"\n>> GATE 2 (coreml {label} vs onnxruntime):")
        mx, amok = 0.0, True
        for r in REFS:
            f = feed(np.load(r)); on = sess.run(None, f)[0]
            cm = m.predict({k: v.astype(np.int32) for k, v in f.items()})["logits"]
            d = float(np.abs(cm - on).max()); mx = max(mx, d)
            a = bool((cm.argmax(-1) == on.argmax(-1)).all()); amok &= a
            print(f"   {os.path.basename(r):16} len={f['input_ids'].shape[1]:3} |Δ|={d:.2e} argmax{'=OK' if a else '!!MISMATCH'}")
        ok = mx < 5e-2 and amok
        print(f">> GATE 2 [{label}] max|Δ|={mx:.2e} argmax_all={amok}  ({'PASS' if ok else 'CHECK'})")
    m32, _ = convert(ct.precision.FLOAT32, "fp32"); gate2(m32, "fp32")
    m16, p16 = convert(ct.precision.FLOAT16, "fp16"); gate2(m16, "fp16")
    print(f"\n>> M3 conversion done. fp16 package: {p16}  ({os.path.getsize(os.path.join(p16,'Data/com.apple.CoreML/weights/weight.bin'))/1e6 if os.path.exists(os.path.join(p16,'Data/com.apple.CoreML/weights/weight.bin')) else '?':.0f} MB weights)")


if __name__ == "__main__":
    main()
