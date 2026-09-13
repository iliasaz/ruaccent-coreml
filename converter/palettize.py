#!/usr/bin/env python3
"""WS / Phase 6 (issue #8): palettize the large fp16 CoreML models for ship size, re-validate parity.

8-bit k-means palettization (deterministic) of M2/M3 (embedding-dominated). Re-checks decision/argmax
parity vs the onnxruntime oracles so size optimization doesn't break numeric equivalency.
Run: python converter/palettize.py
"""
import os, glob, sys
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COREML = os.path.join(ROOT, "converter", "_work", "coreml")
NN = os.path.join(ROOT, "converter", "_work", "nn")
REFS = os.path.join(ROOT, "converter", "fixtures", "refs")


def feed_from(z):
    ids = z["input_ids"].astype(np.int64)
    f = {"input_ids": ids, "attention_mask": (z["attention_mask"] if "attention_mask" in z else np.ones_like(ids)).astype(np.int64)}
    if "token_type_ids" in z:
        f["token_type_ids"] = z["token_type_ids"].astype(np.int64)
    return f


MODELS = {
    "M2_omograph":   dict(onnx=os.path.join(NN, "nn_omograph/turbo3.1/model.onnx"),
                          refs="M2_omograph__*.npz", tt=False, decision="binary"),
    "M3_stress":     dict(onnx=os.path.join(NN, "nn_stress_usage_predictor/model.onnx"),
                          refs="M3_stress__*.npz", tt=True, decision="token"),
}


def main():
    import coremltools as ct
    from coremltools.optimize.coreml import palettize_weights, OpPalettizerConfig, OptimizationConfig

    for name, spec in MODELS.items():
        src = os.path.join(COREML, f"{name}_fp16.mlpackage")
        if not os.path.exists(src):
            print(f"!! {src} missing, skip"); continue
        ml = ct.models.MLModel(src)
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(nbits=8, mode="kmeans"))
        pal = palettize_weights(ml, cfg)
        out = os.path.join(COREML, f"{name}_fp16_pal8.mlpackage")
        pal.save(out)

        sess = ort.InferenceSession(spec["onnx"], providers=["CPUExecutionProvider"])
        max_d, ok = 0.0, True
        for r in sorted(glob.glob(os.path.join(REFS, spec["refs"]))):
            z = np.load(r); f = feed_from(z)
            if spec["tt"] and "token_type_ids" not in f:
                f["token_type_ids"] = np.zeros_like(f["input_ids"])
            on = sess.run(None, f)[0]
            pin = {k: v.astype(np.int32) for k, v in f.items()}
            cm = pal.predict(pin)["logits"]
            d = float(np.abs(cm - on).max()); max_d = max(max_d, d)
            if spec["decision"] == "binary":
                ok &= bool(cm[0].argmax() == on[0].argmax())
            else:
                ok &= bool((cm.argmax(-1) == on.argmax(-1)).all())
        def sz(p):
            return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e6
        print(f"{name}: fp16 {sz(src):6.1f} MB -> pal8 {sz(out):6.1f} MB | "
              f"max|Δ|={max_d:.2e} decision/argmax_all={ok}  ({'PASS' if ok and max_d < 1e-1 else 'CHECK'})")


if __name__ == "__main__":
    main()
