#!/usr/bin/env python3
"""Phase 1 (WS-A): convert M1 accent (RoFormerForTokenClassification) -> CoreML fp16.

Source = the **fp32** `big.onnx` (producer=pytorch), NOT the int8 `model.onnx` upstream runs.
We rebuild the HF RoFormer module from config.json, load weights extracted from big.onnx
(clean initializers + anonymous onnx::MatMul_* mapped by bias-adjacency graph walk), trace
with a STATIC sequence length, and ct.convert to fp16 for ANE.

Parity gates:
  (1) torch(rebuilt) vs onnxruntime(big.onnx)   -> validates weight loading (expect ~1e-4)
  (2) coreml(fp16)   vs onnxruntime(big.onnx)   -> conversion fidelity (expect <2e-3 fp16)
  (3) coreml rendered stress == upstream model.onnx (int8) rendered stress, on the M1 oracle
      words + OOV fixtures  -> FUNCTIONAL equivalence with the actual upstream pipeline.

Run with the conversion venv (torch 2.8 + coremltools 9 + transformers):
  python converter/convert_accent_coreml.py
"""
import os, json, glob
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "converter", "_work", "nn", "nn_accent")
OUTDIR = os.path.join(ROOT, "converter", "_work", "coreml")
os.makedirs(OUTDIR, exist_ok=True)
SEQ = 48                      # static traced length (max_pos=60; longest fixture word ~21 chars)

# ----------------------------------------------------------------------------
# 1. extract a torch state_dict from big.onnx (fp32)
# ----------------------------------------------------------------------------
def extract_state_dict(onnx_path):
    m = onnx.load(onnx_path)
    g = m.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    # producer of each tensor
    produced_by = {}
    for n in g.node:
        for o in n.output:
            produced_by[o] = n
    # map: bias-name -> the MatMul weight initializer feeding the same Add
    # walk Add nodes; if one input is a named '*.bias' initializer, the other input is a
    # MatMul output whose weight initializer is the Linear weight (onnx layout [in,out]).
    weight_for_bias = {}
    for n in g.node:
        if n.op_type != "Add":
            continue
        bias_in = [x for x in n.input if x in inits and x.endswith(".bias")]
        if not bias_in:
            continue
        bias = bias_in[0]
        other = [x for x in n.input if x != bias]
        for o in other:
            prod = produced_by.get(o)
            if prod is not None and prod.op_type in ("MatMul", "Gemm"):
                w = [x for x in prod.input if x in inits]
                if w:
                    weight_for_bias[bias] = w[0]
    sd = {}
    # clean fp32 params we keep verbatim
    for name, arr in inits.items():
        if name.startswith("onnx::"):
            continue
        sd[name] = arr
    # Linear weights: HF Linear.weight is [out,in]; onnx MatMul init is [in,out] -> transpose
    for bias, wname in weight_for_bias.items():
        linear = bias[:-len(".bias")] + ".weight"
        sd[linear] = inits[wname].T.copy()
    return sd, weight_for_bias


def main():
    import torch
    from transformers import RoFormerConfig, RoFormerForTokenClassification

    cfg = json.load(open(os.path.join(ACC, "config.json")))
    config = RoFormerConfig(**cfg)
    print(f">> RoFormer config: {config.num_hidden_layers}L {config.hidden_size}h "
          f"vocab={config.vocab_size} labels={config.id2label}")

    sd_np, wmap = extract_state_dict(os.path.join(ACC, "big.onnx"))
    print(f">> extracted {len(sd_np)} tensors ({len(wmap)} Linear weights mapped by bias-adjacency)")

    model = RoFormerForTokenClassification(config).eval()
    target = model.state_dict()
    sd = {}
    for k, v in sd_np.items():
        if k in target:
            t = torch.tensor(v)
            if tuple(t.shape) == tuple(target[k].shape):
                sd[k] = t
            else:
                print(f"   !! shape mismatch {k}: onnx{tuple(t.shape)} vs hf{tuple(target[k].shape)}")
    missing = [k for k in target if k not in sd and "embed_positions" not in k
               and "position_ids" not in k]
    res = model.load_state_dict(sd, strict=False)
    print(f">> loaded {len(sd)}/{len(target)} params; truly-missing (non-buffer): {missing[:8]}{' ...' if len(missing)>8 else ''}")

    # fp16 hardening (carried learning): HF builds the additive attention mask as
    # (1-mask)*finfo(fp32).min = -3.4e38, which casts to -inf in fp16 -> nan. Use a finite
    # large-negative value; softmax over real tokens is unchanged (exp(-1e4) ~ 0).
    import types
    def finite_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        em = attention_mask[:, None, None, :].to(dtype=torch.float32)
        return (1.0 - em) * -1e4
    # NOTE: the call site is RoFormerModel.forward -> self.get_extended_attention_mask, where
    # self is model.roformer (the inner encoder), NOT the ForTokenClassification wrapper.
    model.roformer.get_extended_attention_mask = types.MethodType(finite_mask, model.roformer)

    # ----- gate (1): torch vs onnxruntime(big.onnx) -----
    ort_big = ort.InferenceSession(os.path.join(ACC, "big.onnx"), providers=["CPUExecutionProvider"])
    ort_int8 = ort.InferenceSession(os.path.join(ACC, "model.onnx"), providers=["CPUExecutionProvider"])
    refs = sorted(glob.glob(os.path.join(ROOT, "converter", "fixtures", "refs", "M1_accent__*.npz")))
    print("\n>> GATE 1 (torch vs big.onnx) + GATE int8-vs-fp32 source, on oracle words:")
    max_t = 0.0
    samples = []
    for r in refs:
        z = np.load(r)
        ids = z["input_ids"].astype(np.int64)
        am = z["attention_mask"].astype(np.int64) if "attention_mask" in z else np.ones_like(ids)
        tt = np.zeros_like(ids)
        feed = {"input_ids": ids, "attention_mask": am, "token_type_ids": tt}
        big_logits = ort_big.run(None, feed)[0]
        int8_logits = ort_int8.run(None, feed)[0]
        with torch.no_grad():
            tl = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(am),
                       token_type_ids=torch.tensor(tt)).logits.numpy()
        d_torch = np.abs(tl - big_logits).max()
        d_q = np.abs(big_logits - int8_logits).max()
        max_t = max(max_t, d_torch)
        samples.append((os.path.basename(r), ids.shape[1], d_torch, d_q))
    for nm, L, dt, dq in samples:
        print(f"   {nm:34} len={L:3}  |torch-big|={dt:.2e}  |big-int8|={dq:.2e}")
    print(f">> GATE 1 max |torch-big| = {max_t:.2e}  ({'PASS' if max_t < 1e-3 else 'CHECK'})")

    # ----- trace with static SEQ + convert -----
    import coremltools as ct

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits

    wrap = Wrap(model).eval()
    ex = (torch.zeros(1, SEQ, dtype=torch.long), torch.ones(1, SEQ, dtype=torch.long),
          torch.zeros(1, SEQ, dtype=torch.long))
    print(f"\n>> tracing at static seq={SEQ} ...")
    traced = torch.jit.trace(wrap, ex)

    def convert(precision, label):
        m = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=(1, SEQ), dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=(1, SEQ), dtype=np.int32),
                    ct.TensorType(name="token_type_ids", shape=(1, SEQ), dtype=np.int32)],
            outputs=[ct.TensorType(name="logits")],
            compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )
        p = os.path.join(OUTDIR, f"M1_accent_{label}.mlpackage")
        m.save(p)
        return m, p

    def gate2(mlmodel, label):
        print(f"\n>> GATE 2 (coreml {label} vs big.onnx), padded to SEQ, real-token diff + argmax:")
        max_c, argmax_ok = 0.0, True
        for r in refs:
            z = np.load(r); ids = z["input_ids"].astype(np.int64); L = ids.shape[1]
            if L > SEQ:
                continue
            am = np.ones_like(ids); pad = SEQ - L
            ids_p = np.pad(ids, ((0, 0), (0, pad))); am_p = np.pad(am, ((0, 0), (0, pad)))
            tt_p = np.zeros_like(ids_p)
            big = ort_big.run(None, {"input_ids": ids_p, "attention_mask": am_p, "token_type_ids": tt_p})[0]
            cm = mlmodel.predict({"input_ids": ids_p.astype(np.int32),
                                  "attention_mask": am_p.astype(np.int32),
                                  "token_type_ids": tt_p.astype(np.int32)})["logits"]
            d = float(np.abs(cm[:, :L] - big[:, :L]).max())
            am_match = bool((cm[:, :L].argmax(-1) == big[:, :L].argmax(-1)).all())
            argmax_ok &= am_match
            max_c = max(max_c, d)
            print(f"   {os.path.basename(r):34} len={L:3} |Δ|={d:.2e} argmax{'=OK' if am_match else '!!MISMATCH'}")
        ok = max_c < 5e-3 and argmax_ok
        print(f">> GATE 2 [{label}] max|Δ|={max_c:.2e} argmax_all_match={argmax_ok}  ({'PASS' if ok else 'CHECK'})")
        return max_c, argmax_ok

    # GATE 3: FUNCTIONAL parity — does the rendered '+' stress match the int8 upstream?
    man = {m["tag"]: m for m in json.load(open(os.path.join(ROOT, "converter", "fixtures", "refs", "manifest.json")))}
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    def softmax(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    def render(word, logits):                          # upstream AccentModel.render_stress
        probs = softmax(logits); lab = probs.argmax(-1); sc = probs.max(-1)
        text = list(word)
        for i in range(len(lab)):
            if id2label[lab[i]] not in ("NO", "STRESS_SECONDARY") and sc[i] >= 0.55 and 0 <= i - 1 < len(text):
                text[i - 1] = "+" + text[i - 1]
        return "".join(text)
    def gate3(mlmodel, label):
        print(f"\n>> GATE 3 (rendered stress: coreml {label} vs int8 upstream):")
        allok = True
        for r in refs:
            z = np.load(r); ids = z["input_ids"].astype(np.int64); L = ids.shape[1]
            if L > SEQ: continue
            word = man[os.path.basename(r)[:-4]]["word"]
            up = man[os.path.basename(r)[:-4]]["rendered"]
            ids_p = np.pad(ids, ((0, 0), (0, SEQ - L)))
            cm = mlmodel.predict({"input_ids": ids_p.astype(np.int32),
                                  "attention_mask": np.pad(np.ones_like(ids), ((0, 0), (0, SEQ - L))).astype(np.int32),
                                  "token_type_ids": np.zeros_like(ids_p).astype(np.int32)})["logits"][0][:L]
            rc = render(word, cm); ok = rc == up; allok &= ok
            print(f"   {word:22} upstream={up:22} coreml={rc:22} {'OK' if ok else 'DIFF'}")
        print(f">> GATE 3 [{label}] rendered==int8-upstream on all oracle words: {allok}  ({'PASS' if allok else 'CHECK'})")
        return allok

    # isolate precision: fp32 sanity first, then fp16 (the deployment target)
    m32, _ = convert(ct.precision.FLOAT32, "fp32")
    gate2(m32, "fp32")
    m16, p16 = convert(ct.precision.FLOAT16, "fp16")
    gate2(m16, "fp16")
    gate3(m16, "fp16")
    print(f"\n>> M1 conversion done. fp16 package: {p16}")


if __name__ == "__main__":
    main()
