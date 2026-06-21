#!/usr/bin/env python3
"""Phase 0 ground-truth dumper for the RUAccent -> CoreML port.

Runs upstream RUAccent (variant ``turbo3.1``, ``use_dictionary=True``,
``tiny_mode=False`` -- the canonical full pipeline) over a curated Russian
fixture set and emits the artifacts every later phase validates against:

  fixtures/fixtures.json   - [{input, output(+notation)}] end-to-end ground truth
  fixtures/model_io.json   - exact ONNX input/output tensor specs for all 4 models
  fixtures/refs/*.npz      - per-model (tokenized input -> logits) reference vectors
  fixtures/refs/manifest.json

The four neural models (confirmed in Phase 0):
  M1 accent       RoFormerForTokenClassification  (char tok, vocab 45)   OOV accentor
  M2 omograph     DebertaForSequenceClassification(RoBERTa-BPE, 60258)   homograph disambig
  M3 stress_usage BertForTokenClassification      (wordpiece, 83828)     whether-to-stress
  M4 yo_homograph DistilBertForTokenClassification(wordpiece, 5031)      e->yo disambig

The upstream ``RuleEngine`` (koziev CRF POS tagger) is *loaded but never called*
in ``process_all`` -- we neutralize its load so Phase 0 doesn't depend on
python-crfsuite / koziev. This is output-neutral (verified: only ``rule_accent.load``
is referenced in ruaccent.py; ``rule_accent.accentuate`` is never invoked).
"""
import os, sys, json, gzip
os.environ.setdefault("HF_HOME", "/Users/ilia/Downloads/HF_HOME")
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.path.join(REPO_ROOT, "converter", "_work")           # gitignored (downloaded weights)
OUT = os.path.join(REPO_ROOT, "converter", "fixtures")
REFS = os.path.join(OUT, "refs")
os.makedirs(WORKDIR, exist_ok=True); os.makedirs(REFS, exist_ok=True)

# ---- output-neutral neutralization of the unused RuleEngine (koziev/crfsuite) ----
import types
_stub = types.ModuleType("ruaccent.rule_accent_engine")
class _NoopRuleEngine:
    def load(self, *a, **k):
        return None
_stub.RuleEngine = _NoopRuleEngine
sys.modules["ruaccent.rule_accent_engine"] = _stub

from ruaccent import RUAccent

# ----------------------------- curated fixtures -----------------------------
# Grouped by what they exercise. Upstream output marks stress with '+' BEFORE the
# stressed vowel and restores 'ё'. These pairs are THE ground truth for the port.
FIXTURES = {
    "homograph_context": [
        "на двери висит замок",
        "на горе стоит старинный замок",
        "я мою посуду",
        "это мой дом",
        "мужики косили траву косой",
        "у неё была русая коса",
        "уже прошёл целый час",
        "дорога стала уже",
        "большие белки прыгали по веткам",
        "в яйце много белка",
        "мы поехали домой",
        "стрелки часов показывают полдень",
    ],
    "plan_named_cases": [
        "мой", "русский", "тёплый", "ёжик", "йога", "мой тёплый дом",
    ],
    "yo_restoration": [
        "ежик нашел гриб",
        "теплый летний вечер",
        "она все еще здесь",
        "все люди разные",
        "он надел черный пиджак",
        "это была теплая встреча",
    ],
    "oov_neural": [
        "квазиблагоприятный",
        "гиперинтенсификация",
        "монокристаллический",
        "флибустьерствующий",
    ],
    "stress_usage_function_words": [
        "я и ты пошли в кино",
        "он не был там вчера",
        "что бы это значило",
        "кот сидит на столе у окна",
    ],
    "multi_sentence_punct": [
        "Привет! Как дела? Всё хорошо.",
        "Мама мыла раму, а папа читал газету.",
        "В 2024 году, в марте, выпал снег.",
        "Доктор И. И. Иванов пришёл на работу.",
    ],
    "edge": [
        "",
        "...",
        "123",
        "abc",
        "кот",          # monosyllable
        "Россия",
    ],
}

# representative inputs for per-model reference activations (numeric oracle)
ACCENT_WORDS = ["квазиблагоприятный", "монокристаллический", "переподготовка", "молоко"]
SENTENCES_FOR_TOKENCLS = [
    "на двери висит замок",
    "она все еще здесь",
    "мама мыла раму а папа читал газету",
]
OMOGRAPH_CASES = [  # (sentence_with_marked_word, hypothesis)
    ("на двери висит <w>замок</w>", "зам+ок"),
    ("на двери висит <w>замок</w>", "з+амок"),
    ("на горе стоит старинный <w>замок</w>", "з+амок"),
    ("на горе стоит старинный <w>замок</w>", "зам+ок"),
]


def onnx_io_spec(session):
    def spec(io):
        return [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in io]
    return {"inputs": spec(session.get_inputs()), "outputs": spec(session.get_outputs())}


def install_safe_run(session):
    """Auto-fill any required ONNX input the (transformers 5.x) tokenizer omits with zeros.

    The M1/M3 graphs require ``token_type_ids``; newer transformers tokenizers don't emit it
    for single-segment inputs. For single-segment text token_type_ids is all-zeros, so this is
    numerically IDENTICAL to the author's contemporaneous (4.x) behavior -- it just makes the
    upstream run under our 5.x venv. Applied uniformly to all 4 sessions."""
    orig = session.run
    required = [i.name for i in session.get_inputs()]
    def run(output_names, feed, *a, **k):
        if "input_ids" in feed:
            for name in required:
                if name not in feed:
                    feed[name] = np.zeros_like(feed["input_ids"])
        return orig(output_names, feed, *a, **k)
    session.run = run


def save_ref(tag, inputs: dict, logits: np.ndarray, extra: dict | None = None):
    path = os.path.join(REFS, f"{tag}.npz")
    np.savez_compressed(path, logits=logits, **{k: np.asarray(v) for k, v in inputs.items()})
    rec = {"tag": tag, "file": os.path.relpath(path, OUT),
           "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype),
           "inputs": {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
                      for k, v in inputs.items()}}
    if extra:
        rec.update(extra)
    return rec


def main():
    print(">> loading RUAccent turbo3.1 (use_dictionary=True, tiny_mode=False)...", flush=True)
    acc = RUAccent()
    acc.koziev_paths = []            # output-neutral: rule engine unused in process_all
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
             device="CPU", workdir=WORKDIR)
    print(">> loaded. installing zero-fill for required-but-omitted token_type_ids...", flush=True)
    for m in (acc.accent_model, acc.omograph_model, acc.stress_usage_predictor, acc.yo_homograph_model):
        install_safe_run(m.session)
    print(">> dumping model I/O specs...", flush=True)

    model_io = {
        "M1_accent":       {"arch": "RoFormerForTokenClassification",  **onnx_io_spec(acc.accent_model.session)},
        "M2_omograph":     {"arch": "DebertaForSequenceClassification", **onnx_io_spec(acc.omograph_model.session)},
        "M3_stress_usage": {"arch": "BertForTokenClassification",       **onnx_io_spec(acc.stress_usage_predictor.session)},
        "M4_yo_homograph": {"arch": "DistilBertForTokenClassification", **onnx_io_spec(acc.yo_homograph_model.session)},
    }
    with open(os.path.join(OUT, "model_io.json"), "w", encoding="utf-8") as f:
        json.dump(model_io, f, ensure_ascii=False, indent=2)
    print(json.dumps(model_io, ensure_ascii=False, indent=2), flush=True)

    # ----------------- end-to-end fixtures (ground truth) -----------------
    print(">> running end-to-end fixtures...", flush=True)
    fixtures = {"meta": {"variant": "turbo3.1", "use_dictionary": True, "tiny_mode": False,
                         "notation": "plus_before_vowel", "ruaccent_version": __import__("ruaccent").__version__},
                "groups": {}}
    for group, items in FIXTURES.items():
        rows = []
        for s in items:
            try:
                out = acc.process_all(s)
            except Exception as e:
                out = f"<<ERROR: {type(e).__name__}: {e}>>"
            rows.append({"input": s, "output": out})
            print(f"   [{group}] {s!r} -> {out!r}", flush=True)
        fixtures["groups"][group] = rows
    with open(os.path.join(OUT, "fixtures.json"), "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)

    # ----------------- per-model reference activations -----------------
    print(">> dumping per-model reference activations...", flush=True)
    manifest = []

    # M1 accent (char-level RoFormer token classifier)
    am = acc.accent_model
    out_names = {o.name: i for i, o in enumerate(am.session.get_outputs())}
    for w in ACCENT_WORDS:
        inp = am.tokenizer(w.lower(), return_tensors="np")
        inp = {k: v.astype(np.int64) for k, v in inp.items()}
        logits = am.session.run(None, inp)[out_names["logits"]]
        manifest.append(save_ref(f"M1_accent__{w}", inp, logits,
                                 {"model": "M1_accent", "word": w,
                                  "rendered": am.put_accent(w)}))

    # M3 stress_usage (BERT token classifier)
    sm = acc.stress_usage_predictor
    for i, s in enumerate(SENTENCES_FOR_TOKENCLS):
        inp = sm.tokenizer(s, return_tensors="np")
        inp = {k: v.astype(np.int64) for k, v in inp.items()}
        logits = sm.session.run(None, inp)[0]
        manifest.append(save_ref(f"M3_stress__{i}", inp, logits,
                                 {"model": "M3_stress_usage", "sentence": s}))

    # M4 yo_homograph (DistilBERT token classifier)
    ym = acc.yo_homograph_model
    for i, s in enumerate(SENTENCES_FOR_TOKENCLS):
        inp = ym.tokenizer(s, return_tensors="np")
        inp = {k: v.astype(np.int64) for k, v in inp.items()}
        logits = ym.session.run(None, inp)[0]
        manifest.append(save_ref(f"M4_yo__{i}", inp, logits,
                                 {"model": "M4_yo_homograph", "sentence": s}))

    # M2 omograph (DeBERTa cross-encoder, text + hypothesis pair)
    om = acc.omograph_model
    for i, (txt, hyp) in enumerate(OMOGRAPH_CASES):
        inp = om.tokenizer(txt, hyp, max_length=512, truncation=True, return_tensors="np")
        inp = {k: v.astype(np.int64) for k, v in inp.items()}
        logits = om.session.run(None, inp)[0]
        manifest.append(save_ref(f"M2_omograph__{i}", inp, logits,
                                 {"model": "M2_omograph", "text": txt, "hypothesis": hyp,
                                  "prob_true": float(om.softmax(logits)[0][1])}))

    with open(os.path.join(REFS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f">> DONE. {sum(len(v) for v in fixtures['groups'].values())} fixtures, "
          f"{len(manifest)} reference vectors.", flush=True)


if __name__ == "__main__":
    main()
