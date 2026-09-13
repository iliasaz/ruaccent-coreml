#!/usr/bin/env python3
"""WS-B (issue #4): convert M2 omograph (DebertaForSequenceClassification) -> CoreML fp16.

The "hard one" that isn't: relative_attention=false / pos_att_type=null => disentangled attention
is DEAD CODE; it runs as a vanilla BERT absolute-position encoder. The only DeBERTa-v1 quirk is the
FUSED in_proj (768->2304, no bias) split Q/K/V with SEPARATE q_bias/v_bias (no k_bias). The generic
bias-adjacency extractor maps everything except in_proj/q_bias/v_bias; we add those by shape+order
(validated by GATE 1 — wrong q/v or layer order makes torch vs onnx diverge loudly).

Cross-encoder: input_ids+attention_mask [1,S] (NO token_type_ids) -> logits [1,2]; index 1 = P(hyp correct).
Run: python converter/convert_omograph_coreml.py
"""
import os, json, glob, sys, re
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_extract import extract_state_dict, load_into

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M2 = os.path.join(ROOT, "converter", "_work", "nn", "nn_omograph", "turbo3.1")
OUTDIR = os.path.join(ROOT, "converter", "_work", "coreml"); os.makedirs(OUTDIR, exist_ok=True)
REFS = sorted(glob.glob(os.path.join(ROOT, "converter", "fixtures", "refs", "M2_omograph__*.npz")))


def suffix(n): return int(re.findall(r"\d+", n)[0])


def extract_deberta(onnx_path, n_layers, hidden, heads):
    sd, wmap = extract_state_dict(onnx_path)             # named + adjacent-bias Linear weights
    inits = {t.name: numpy_helper.to_array(t) for t in onnx.load(onnx_path).graph.initializer}
    anon = {n: a for n, a in inits.items() if n.startswith("onnx::")}
    head_dim = hidden // heads
    inproj = sorted([n for n, a in anon.items() if a.shape == (hidden, 3 * hidden)], key=suffix)
    biases = sorted([n for n, a in anon.items() if a.shape == (1, heads, 1, head_dim)], key=suffix)
    assert len(inproj) == n_layers, f"expected {n_layers} in_proj, got {len(inproj)}"
    assert len(biases) == 2 * n_layers, f"expected {2*n_layers} q/v biases, got {len(biases)}"
    for L in range(n_layers):
        base = f"deberta.encoder.layer.{L}.attention.self"
        sd[f"{base}.in_proj.weight"] = inits[inproj[L]].T.copy()        # onnx [in,3h] -> HF [3h,in]
        sd[f"{base}.q_bias"] = inits[biases[2 * L]].reshape(hidden)
        sd[f"{base}.v_bias"] = inits[biases[2 * L + 1]].reshape(hidden)
    return sd, wmap, len(inproj)


def main():
    import torch
    from transformers import DebertaConfig, DebertaForSequenceClassification
    import coremltools as ct

    cfg = json.load(open(os.path.join(M2, "config.json")))
    config = DebertaConfig(**cfg)
    print(f">> DeBERTa: {config.num_hidden_layers}L/{config.hidden_size}h heads={config.num_attention_heads} "
          f"vocab={config.vocab_size} rel_attn={getattr(config,'relative_attention',None)} labels={config.num_labels}")

    model = DebertaForSequenceClassification(config).eval()
    tgt = model.state_dict()
    print(">> sanity: HF attention param names present?",
          all(any(f".layer.0.attention.self.{k}" in t for t in tgt) for k in ("in_proj.weight", "q_bias", "v_bias")))

    sd_np, wmap, ninp = extract_deberta(os.path.join(M2, "model.onnx"),
                                        config.num_hidden_layers, config.hidden_size, config.num_attention_heads)
    rep = load_into(model, sd_np)
    print(f">> mapped {len(wmap)} adjacent-bias + {ninp} fused-in_proj; loaded {rep['loaded']}/{rep['total']} "
          f"params; missing={rep['missing'][:8]}")

    sess = ort.InferenceSession(os.path.join(M2, "model.onnx"), providers=["CPUExecutionProvider"])
    print("\n>> GATE 1 (torch vs onnxruntime), [1,2] logits + P(hyp):")
    max1 = 0.0
    for r in REFS:
        z = np.load(r); ids = z["input_ids"].astype(np.int64); am = z["attention_mask"].astype(np.int64)
        on = sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
        with torch.no_grad():
            tl = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(am)).logits.numpy()
        d = float(np.abs(tl - on).max()); max1 = max(max1, d)
        sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        print(f"   {os.path.basename(r):20} len={ids.shape[1]:3} |Δ|={d:.2e}  P_on={sm(on[0])[1]:.4f} P_torch={sm(tl[0])[1]:.4f}")
    print(f">> GATE 1 max|Δ|={max1:.2e}  ({'PASS' if max1 < 1e-3 else 'CHECK — q/v or layer order?'})")
    if max1 >= 1e-3:
        print(">> GATE 1 failed; not converting. Fix the fused-QKV mapping first."); return

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits
    wrap = Wrap(model).eval()
    S0 = int(np.load(REFS[-1])["input_ids"].shape[1])
    ex = (torch.zeros(1, S0, dtype=torch.long), torch.ones(1, S0, dtype=torch.long))
    # carried learning: DeBERTa's scaled_size_sqrt does sqrt(query.size(-1)*factor); the size-derived
    # value traces as int32 and coremltools' sqrt rejects int. head_dim is static (64) -> bake a const.
    import math
    import transformers.models.deberta.modeling_deberta as mdeb
    mdeb.scaled_size_sqrt = lambda q, sf: torch.tensor(math.sqrt(int(q.shape[-1]) * sf), dtype=torch.float32)
    print(f"\n>> tracing at S0={S0}, RangeDim(8,256) ...")
    traced = torch.jit.trace(wrap, ex)
    seq = ct.RangeDim(lower_bound=8, upper_bound=256, default=S0)

    def convert(precision, label):
        m = ct.convert(traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32)],
            outputs=[ct.TensorType(name="logits")], compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.iOS18)
        p = os.path.join(OUTDIR, f"M2_omograph_{label}.mlpackage"); m.save(p); return m, p

    def gate2(m, label):
        print(f"\n>> GATE 2 (coreml {label} vs onnxruntime):")
        mx, dok = 0.0, True
        for r in REFS:
            z = np.load(r); ids = z["input_ids"].astype(np.int64); am = z["attention_mask"].astype(np.int64)
            on = sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
            cm = m.predict({"input_ids": ids.astype(np.int32), "attention_mask": am.astype(np.int32)})["logits"]
            d = float(np.abs(cm - on).max()); mx = max(mx, d)
            dec = bool((cm[0].argmax() == on[0].argmax())); dok &= dec
            print(f"   {os.path.basename(r):20} len={ids.shape[1]:3} |Δ|={d:.2e} decision{'=OK' if dec else '!!FLIP'}")
        ok = mx < 5e-2 and dok
        print(f">> GATE 2 [{label}] max|Δ|={mx:.2e} decision_all={dok}  ({'PASS' if ok else 'CHECK'})")
    m16, p16 = convert(ct.precision.FLOAT16, "fp16"); gate2(m16, "fp16")
    print(f"\n>> M2 conversion done. fp16 package: {p16}")


if __name__ == "__main__":
    main()
