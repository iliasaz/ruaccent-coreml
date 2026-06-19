"""Shared: extract an HF state_dict from a PyTorch-exported ONNX graph.

The ruaccent ONNX exports name biases/LayerNorms/embeddings cleanly but decompose each
nn.Linear into MatMul(W_transposed)+Add(bias) with the WEIGHT as an anonymous `onnx::MatMul_*`
initializer. We recover Linear.weight = onnx_init.T by **bias-adjacency**: the Add that consumes
a named `*.bias` also consumes the MatMul output whose weight initializer is that Linear's weight.
Used by M1 (RoFormer), M3 (BERT), M4 (DistilBERT). (M2 DeBERTa adds a fused-QKV special case.)
"""
import numpy as np
import onnx
from onnx import numpy_helper


def extract_state_dict(onnx_path):
    """Return (state_dict: {hf_name: np.ndarray}, weight_for_bias: {bias_name: onnx_weight_name})."""
    g = onnx.load(onnx_path).graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    produced_by = {o: n for n in g.node for o in n.output}

    weight_for_bias = {}
    for n in g.node:
        if n.op_type != "Add":
            continue
        bias_in = [x for x in n.input if x in inits and x.endswith(".bias")]
        if not bias_in:
            continue
        bias = bias_in[0]
        for o in (x for x in n.input if x != bias):
            prod = produced_by.get(o)
            if prod is not None and prod.op_type in ("MatMul", "Gemm"):
                w = [x for x in prod.input if x in inits]
                if w:
                    weight_for_bias[bias] = w[0]

    sd = {name: arr for name, arr in inits.items() if not name.startswith("onnx::")}
    for bias, wname in weight_for_bias.items():
        sd[bias[:-len(".bias")] + ".weight"] = inits[wname].T.copy()  # onnx [in,out] -> HF [out,in]
    return sd, weight_for_bias


def load_into(model, sd_np):
    """Load a numpy state_dict into an HF module by name+shape (strict=False); print a report."""
    import torch
    target = model.state_dict()
    sd, skipped = {}, []
    for k, v in sd_np.items():
        if k in target and tuple(v.shape) == tuple(target[k].shape):
            sd[k] = torch.tensor(v)
        elif k in target:
            skipped.append(f"{k}: onnx{tuple(v.shape)} vs hf{tuple(target[k].shape)}")
    model.load_state_dict(sd, strict=False)
    missing = [k for k in target if k not in sd
               and not any(s in k for s in ("position_ids", "embed_positions", "token_type_ids"))]
    return {"loaded": len(sd), "total": len(target), "missing": missing, "shape_skipped": skipped}
