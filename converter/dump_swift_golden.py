#!/usr/bin/env python3
"""Swift golden-reference dumper for the RUAccent -> CoreML port (Agent A).

Produces ONE committable, byte-exact oracle file ``converter/fixtures/golden.json``
that the Swift unit tests validate against. It loads the SAME upstream RUAccent
(variant ``turbo3.1``, ``use_dictionary=True``, ``tiny_mode=False``) the way
``dump_ruaccent_fixtures.py`` does -- with the unused ``RuleEngine`` neutralized and
the ``install_safe_run`` zero-fill so the 4.x-era upstream runs under our 5.x venv --
and emits TWO sections:

  (1) "pipeline"   : per input string, every intermediate stage of
                     ``process_all_internal`` (normalized -> sentences ->
                     per-sentence {words, remaining_text} -> word lists after
                     _process_yo / _process_omographs / _process_accent ->
                     stress_usages entities -> rendered sentence), plus the final
                     ``acc.process_all(text)``.
  (2) "tokenizers" : standalone byte-exact tokenizer oracles for the 4 models
                     (M1 char, M2 BPE pair, M3 wordpiece, M4 wordpiece).

This script does NOT modify dump_ruaccent_fixtures.py; it reuses its patterns.

Run:
  export HF_HOME=/Users/ilia/Downloads/HF_HOME
  /Users/ilia/Developer/ruaccent-coreml/.venv/bin/python converter/dump_swift_golden.py
"""
import os, sys, json, re
os.environ.setdefault("HF_HOME", "/Users/ilia/Downloads/HF_HOME")
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.path.join(REPO_ROOT, "converter", "_work")           # gitignored (downloaded weights)
OUT = os.path.join(REPO_ROOT, "converter", "fixtures")
os.makedirs(OUT, exist_ok=True)

ROUND = 6  # decimal places for JSON-friendly floats

# ---- output-neutral neutralization of the unused RuleEngine (koziev/crfsuite) ----
# (identical mechanism to dump_ruaccent_fixtures.py; rule_accent.accentuate is never
# invoked in process_all, only rule_accent.load -- so stubbing the load is output-neutral.)
import types
_stub = types.ModuleType("ruaccent.rule_accent_engine")
class _NoopRuleEngine:
    def load(self, *a, **k):
        return None
_stub.RuleEngine = _NoopRuleEngine
sys.modules["ruaccent.rule_accent_engine"] = _stub

from ruaccent import RUAccent
from ruaccent.text_preprocessor import TextPreprocessor


def install_safe_run(session):
    """Auto-fill any required ONNX input the (transformers 5.x) tokenizer omits with zeros.

    The M1/M3 graphs require ``token_type_ids``; newer transformers tokenizers don't emit it
    for single-segment inputs. For single-segment text token_type_ids is all-zeros, so this is
    numerically IDENTICAL to the author's contemporaneous (4.x) behavior. Copied verbatim from
    dump_ruaccent_fixtures.py."""
    orig = session.run
    required = [i.name for i in session.get_inputs()]
    def run(output_names, feed, *a, **k):
        if "input_ids" in feed:
            for name in required:
                if name not in feed:
                    feed[name] = np.zeros_like(feed["input_ids"])
        return orig(output_names, feed, *a, **k)
    session.run = run


def jround(x):
    """Round a python float to ROUND dp for stable JSON."""
    return round(float(x), ROUND)


# ----------------------------- pipeline inputs -----------------------------
# All inputs from fixtures.json (every group, in order), plus extra coverage:
# punctuation, abbreviations, multi-sentence, digits, ё, hyphens, manual '+'.
FIXTURE_INPUTS = [
    # homograph_context
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
    # plan_named_cases
    "мой",
    "русский",
    "тёплый",
    "ёжик",
    "йога",
    "мой тёплый дом",
    # yo_restoration
    "ежик нашел гриб",
    "теплый летний вечер",
    "она все еще здесь",
    "все люди разные",
    "он надел черный пиджак",
    "это была теплая встреча",
    # oov_neural
    "квазиблагоприятный",
    "гиперинтенсификация",
    "монокристаллический",
    "флибустьерствующий",
    # stress_usage_function_words
    "я и ты пошли в кино",
    "он не был там вчера",
    "что бы это значило",
    "кот сидит на столе у окна",
    # multi_sentence_punct
    "Привет! Как дела? Всё хорошо.",
    "Мама мыла раму, а папа читал газету.",
    "В 2024 году, в марте, выпал снег.",
    "Доктор И. И. Иванов пришёл на работу.",
    # edge
    "",
    "...",
    "123",
    "abc",
    "кот",
    "Россия",
]

# Extra coverage requested: punctuation, abbreviations, multi-sentence, digits, ё, hyphens, manual '+'.
EXTRA_INPUTS = [
    "Кто-то пришёл, и всё-таки ушёл.",          # hyphens + multi-clause punctuation
    "Это стоит 100 рублей и 50 копеек.",         # digits in a sentence
    "Идём в кафе? Да, конечно!",                 # multi-sentence with ? and !
    "по-русски говорил он чётко",                # hyphenated adverb + ё
    "т. е. так и есть",                          # abbreviation (paired shortening т. е.)
    "г. Москва — столица России",                 # abbreviation 'г.' + em dash
    "она зам+ок повесила на дверь",              # manual '+' already present (must win)
    "Ёлка стояла в углу.",                        # leading ё + capital
    "А и Б сидели на трубе",                       # single-letter words
    "2+2=4",                                       # digits with operators / symbols
]

# A '+' inside a word should be preserved by _process_accent (manual stress wins).
PIPELINE_INPUTS = FIXTURE_INPUTS + EXTRA_INPUTS


# --------------------- standalone tokenizer oracle inputs ---------------------
# M1 char tokenizer: single lowercased words (OOV-ish + dictionary words).
M1_WORDS = [
    "молоко", "квазиблагоприятный", "монокристаллический", "переподготовка",
    "флибустьерствующий", "кот", "ёжик", "тёплый", "россия", "абвгд",
]

# M3 / M4 wordpiece: diverse strings (words, sentences, punctuation, ё, digits,
# latin, hyphenated, OOV nonsense).
TOKENCLS_STRINGS = [
    "на двери висит замок",
    "она все еще здесь",
    "мама мыла раму а папа читал газету",
    "привет",
    "ёжик нашёл гриб",
    "в 2024 году выпал снег",
    "hello world",
    "по-русски",
    "квазиблагоприятный",
    "флибустьерствующий нечтоподобное",
]

# M2 BPE: (sentence_with_marked_word, hypothesis) PAIRS built EXACTLY as
# _process_omographs builds them (see below for the faithful construction).
# These omograph inputs each contain a known omograph word.
M2_OMOGRAPH_SENTENCES = [
    "на двери висит замок",
    "на горе стоит старинный замок",
    "мужики косили траву косой",
    "уже прошёл целый час",
    "большие белки прыгали по веткам",
]


# ===================== faithful per-stage pipeline capture =====================
def _build_omograph_pairs(acc, words):
    """Replicate _process_omographs' construction of (marked-text, hypothesis) pairs
    WITHOUT running the classifier -- used only for the M2 tokenizer oracle so the pairs
    are byte-identical to what the model actually sees.

    Mirrors ruaccent.py _process_omographs exactly: for each omograph word at position i,
    wrap t[i] as ' <w>WORD</w> ', join the word list, run delete_spaces_before_punc, and
    emit ONE marked text per hypothesis variant."""
    splitted_text = list(words)
    founded = []
    for i, word in enumerate(splitted_text):
        variants = acc.omographs.get(word)
        if variants:
            founded.append({"word": word, "variants": variants, "position": i})

    pairs = []  # list of {position, word, variants, marked_text (per variant), hypothesis}
    for o in founded:
        position = o["position"]
        t = list(splitted_text)
        t_back = t[position]
        t[position] = ' <w>' + t[position] + '</w> '
        marked = acc.delete_spaces_before_punc(" ".join(t.copy()))
        t[position] = t_back
        # one (marked_text, hypothesis) per variant, exactly as texts_batch / hypotheses_batch
        for hyp in o["variants"]:
            pairs.append({
                "position": position,
                "word": o["word"],
                "variants": list(o["variants"]),
                "marked_text": marked,
                "hypothesis": hyp,
            })
    return pairs


def capture_pipeline(acc, text):
    """Faithfully replicate process_all_internal, recording every intermediate stage.

    The control flow & method calls mirror ruaccent.py::process_all_internal line-for-line
    (normalize regex -> split_by_sentences -> per sentence split_by_words ->
    predict_stress_usage -> _process_yo -> _process_omographs -> _process_accent ->
    reassemble -> delete_spaces_before_punc)."""
    rec = {"input": text}

    # text = re.sub(self.normalize, "", text)
    normalized = re.sub(acc.normalize, "", text)
    rec["normalized"] = normalized

    # sentences = TextPreprocessor.split_by_sentences(text)
    sentences = TextPreprocessor.split_by_sentences(normalized)
    rec["sentences"] = list(sentences)

    sent_records = []
    rendered_parts = []
    for sentence in sentences:
        srec = {"sentence": sentence}

        # words, remaining_text = TextPreprocessor.split_by_words(sentence)
        words, remaining_text = TextPreprocessor.split_by_words(sentence)
        srec["words"] = list(words)
        srec["remaining_text"] = list(remaining_text)

        if len(words) == 0:
            # outputs.append("".join(remaining_text)); continue
            srec["empty_words"] = True
            srec["stress_usages"] = []
            srec["after_yo"] = []
            srec["after_omographs"] = []
            srec["after_accent"] = []
            srec["omograph_pairs"] = []
            processed_sentence = "".join(remaining_text)
            srec["rendered"] = processed_sentence
            rendered_parts.append(processed_sentence)
            sent_records.append(srec)
            continue

        # stress_usages = extract_entities(predict_stress_usage(sentence))   (tiny_mode is False)
        su_entities = acc.stress_usage_predictor.predict_stress_usage(sentence)
        srec["stress_usages_full"] = [
            {"word": e["word"], "entity": e["entity"], "score": jround(e["score"])}
            for e in su_entities
        ]
        stress_usages = acc.extract_entities(su_entities)
        srec["stress_usages"] = list(stress_usages)

        # capture the (marked-text, hypothesis) pairs that _process_omographs WILL feed M2,
        # computed on the post-yo word list (matches the real call order below).
        # NOTE: _process_yo mutates its `words` argument in place and returns it, so we must
        # capture pairs AFTER yo and BEFORE omographs -- done below.

        # processed_words = self._process_yo(words, sentence)
        # _process_yo mutates `words` in place; pass a copy so srec["words"] stays the raw split.
        yo_words = acc._process_yo(list(words), sentence)
        srec["after_yo"] = list(yo_words)

        # omograph pairs (oracle only) are built from the post-yo list, mirroring real order
        srec["omograph_pairs"] = _build_omograph_pairs(acc, yo_words)

        # processed_words = self._process_omographs(processed_words)
        # _process_omographs mutates in place too; pass a copy.
        omo_words = acc._process_omographs(list(yo_words))
        srec["after_omographs"] = list(omo_words)

        # processed_words = self._process_accent(processed_words, stress_usages)
        acc_words = acc._process_accent(list(omo_words), stress_usages)
        srec["after_accent"] = list(acc_words)

        # processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)] + [remaining_text[-1]])
        processed_sentence = "".join(
            [l + r for l, r in zip(remaining_text, acc_words)] + [remaining_text[-1]]
        )
        # processed_sentence = self.delete_spaces_before_punc(processed_sentence)
        processed_sentence = acc.delete_spaces_before_punc(processed_sentence)
        srec["rendered"] = processed_sentence
        rendered_parts.append(processed_sentence)
        sent_records.append(srec)

    rec["per_sentence"] = sent_records
    # process_all_internal returns "".join(outputs)
    rec["rendered_join"] = "".join(rendered_parts)

    # final = acc.process_all(text)  (no skip_regex -> == process_all_internal(text))
    rec["final"] = acc.process_all(text)
    return rec


# ===================== standalone tokenizer oracles =====================
def dump_M1_char(acc):
    am = acc.accent_model
    tok = am.tokenizer
    # NOTE: upstream CharTokenizer.convert_ids_to_tokens is broken (its
    # _convert_id_to_token references a non-existent self.ids_to_tokens), so we
    # reconstruct tokens from the inverse vocab. This is byte-faithful: a char
    # absent from the vocab maps to [unk], exactly as the input_ids encode it.
    inv_vocab = {idx: t for t, idx in tok.get_vocab().items()}
    out = []
    for w in M1_WORDS:
        lw = w.lower()
        chars = list(lw)                          # _tokenize == list(text)
        enc = tok(lw, return_tensors="np")
        input_ids = [int(x) for x in enc["input_ids"][0]]
        attention_mask = [int(x) for x in enc["attention_mask"][0]]
        ttid = None
        if "token_type_ids" in enc:
            ttid = [int(x) for x in enc["token_type_ids"][0]]
        tokens = [inv_vocab[i] for i in input_ids]   # [bos] + chars(or [unk]) + [eos]
        out.append({
            "word": w,
            "lower": lw,
            "chars": chars,                        # list(text)
            "tokens": tokens,                      # incl. [bos]/[eos] wrapping
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": ttid,
            "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id,
            "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
            "unk_token": tok.unk_token, "unk_token_id": tok.unk_token_id,
            "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
            "put_accent": am.put_accent(w),        # OOV neural-accent rendered output
        })
    return out


def _dump_wordpiece(model, strings, predict_fn):
    tok = model.tokenizer
    has_ttid = None
    out = []
    for s in strings:
        enc = tok(
            s,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )
        input_ids = [int(x) for x in enc["input_ids"][0]]
        attention_mask = [int(x) for x in enc["attention_mask"][0]]
        tokens = tok.convert_ids_to_tokens(input_ids)
        offset_mapping = [[int(a), int(b)] for a, b in enc["offset_mapping"][0]]
        special_tokens_mask = [int(x) for x in enc["special_tokens_mask"][0]]
        rec = {
            "text": s,
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }
        if "token_type_ids" in enc:
            rec["token_type_ids"] = [int(x) for x in enc["token_type_ids"][0]]
            has_ttid = True
        else:
            rec["token_type_ids"] = None
            if has_ttid is None:
                has_ttid = False
        # aggregated per-word entities (word, entity, score) from the model's predict fn
        entities = predict_fn(s)
        rec["entities"] = [
            {"word": e["word"], "entity": e["entity"], "score": jround(e["score"]),
             "start": (int(e["start"]) if e["start"] is not None else None),
             "end": (int(e["end"]) if e["end"] is not None else None)}
            for e in entities
        ]
        out.append(rec)
    return out


def dump_M2_bpe(acc):
    """For each omograph sentence, run the SAME first-stage split (normalize ->
    split_by_sentences -> split_by_words -> _process_yo) so the word list matches the
    real pipeline, then build (marked_text, hypothesis) pairs exactly like
    _process_omographs and tokenize + score each pair with M2 (prob_true = softmax[0][1])."""
    om = acc.omograph_model
    tok = om.tokenizer
    out = []
    for raw in M2_OMOGRAPH_SENTENCES:
        normalized = re.sub(acc.normalize, "", raw)
        sentences = TextPreprocessor.split_by_sentences(normalized)
        # these inputs are single-sentence; handle generally anyway
        for sentence in sentences:
            words, _ = TextPreprocessor.split_by_words(sentence)
            if len(words) == 0:
                continue
            yo_words = acc._process_yo(list(words), sentence)
            pairs = _build_omograph_pairs(acc, yo_words)
            for p in pairs:
                marked = p["marked_text"]
                hyp = p["hypothesis"]
                # OmographModel.classify single-call path: tokenizer(t, hp, max_length=512,
                # truncation=True, return_tensors="np"); preprocessing of the text removes
                # spaces before trailing punctuation (re.sub r'\s+(?=([,.?!:;…]))').
                preprocessed = re.sub(r'\s+(?=(?:[,.?!:;…]))', r'', marked)
                enc = tok(preprocessed, hyp, max_length=512, truncation=True, return_tensors="np")
                feed = {k: v.astype(np.int64) for k, v in enc.items()}
                logits = om.session.run(None, feed)[0]
                prob_true = float(om.softmax(logits)[0][1])
                input_ids = [int(x) for x in enc["input_ids"][0]]
                attention_mask = [int(x) for x in enc["attention_mask"][0]]
                tokens = tok.convert_ids_to_tokens(input_ids)
                rec = {
                    "source_sentence": raw,
                    "word": p["word"],
                    "variants": p["variants"],
                    "marked_text": marked,
                    "preprocessed_text": preprocessed,
                    "hypothesis": hyp,
                    "tokens": tokens,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "prob_true": jround(prob_true),
                }
                if "token_type_ids" in enc:
                    rec["token_type_ids"] = [int(x) for x in enc["token_type_ids"][0]]
                else:
                    rec["token_type_ids"] = None
                out.append(rec)
    return out


def main():
    print(">> loading RUAccent turbo3.1 (use_dictionary=True, tiny_mode=False)...", flush=True)
    acc = RUAccent()
    acc.koziev_paths = []            # output-neutral: rule engine unused in process_all
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
             device="CPU", workdir=WORKDIR)
    print(">> loaded. installing zero-fill for required-but-omitted token_type_ids...", flush=True)
    for m in (acc.accent_model, acc.omograph_model, acc.stress_usage_predictor, acc.yo_homograph_model):
        install_safe_run(m.session)

    golden = {
        "meta": {
            "variant": "turbo3.1",
            "use_dictionary": True,
            "tiny_mode": False,
            "notation_internal": "plus_before_vowel",
            "ruaccent_version": __import__("ruaccent").__version__,
            "float_round_dp": ROUND,
            "normalize_regex": acc.normalize.pattern,
            "labels": {
                "M1_accent": {"0": "NO", "1": "STRESS_PRIMARY", "2": "STRESS_SECONDARY"},
                "M3_stress_usage": {"0": "NO_STRESS", "1": "PUNCT", "2": "STRESS"},
                "M4_yo_homograph": {"0": "NO_YO", "1": "PUNCT", "2": "YO"},
                "M2_omograph": {"0": "false", "1": "true"},
            },
        },
        "pipeline": [],
        "tokenizers": {},
    }

    # ----------------- (1) pipeline -----------------
    print(">> capturing pipeline stages...", flush=True)
    errored = []
    for s in PIPELINE_INPUTS:
        try:
            rec = capture_pipeline(acc, s)
            # sanity: final must equal the join of per-sentence rendered parts
            rec["final_matches_join"] = (rec["final"] == rec["rendered_join"])
            golden["pipeline"].append(rec)
            print(f"   [pipe] {s!r} -> {rec['final']!r}"
                  f"{'' if rec['final_matches_join'] else '  <<JOIN MISMATCH>>'}", flush=True)
        except Exception as e:
            errored.append((s, f"{type(e).__name__}: {e}"))
            golden["pipeline"].append({"input": s, "error": f"{type(e).__name__}: {e}"})
            print(f"   [pipe] {s!r} -> <<ERROR: {type(e).__name__}: {e}>>", flush=True)

    # ----------------- (2) tokenizers -----------------
    print(">> dumping M1 char tokenizer oracle...", flush=True)
    golden["tokenizers"]["M1_char"] = dump_M1_char(acc)

    print(">> dumping M3 wordpiece tokenizer oracle...", flush=True)
    golden["tokenizers"]["M3_wordpiece"] = _dump_wordpiece(
        acc.stress_usage_predictor, TOKENCLS_STRINGS,
        acc.stress_usage_predictor.predict_stress_usage)

    print(">> dumping M4 wordpiece tokenizer oracle...", flush=True)
    golden["tokenizers"]["M4_wordpiece"] = _dump_wordpiece(
        acc.yo_homograph_model, TOKENCLS_STRINGS,
        acc.yo_homograph_model.predict_yo_homographs)

    print(">> dumping M2 BPE pair tokenizer + scoring oracle...", flush=True)
    golden["tokenizers"]["M2_bpe"] = dump_M2_bpe(acc)

    # ----------------- write -----------------
    out_path = os.path.join(OUT, "golden.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    # ----------------- summary -----------------
    n_pipe = len(golden["pipeline"])
    n_sent = sum(len(r.get("per_sentence", [])) for r in golden["pipeline"])
    n_join_mismatch = sum(1 for r in golden["pipeline"]
                          if r.get("final_matches_join") is False)
    print("\n======== SUMMARY ========", flush=True)
    print(f"file: {out_path}", flush=True)
    print(f"pipeline rows         : {n_pipe}  (sentences captured: {n_sent})", flush=True)
    print(f"  join mismatches     : {n_join_mismatch}", flush=True)
    print(f"  pipeline errors     : {len(errored)}", flush=True)
    for s, e in errored:
        print(f"    ERROR {s!r}: {e}", flush=True)
    print(f"tokenizers.M1_char    : {len(golden['tokenizers']['M1_char'])}", flush=True)
    print(f"tokenizers.M3_wordpiece: {len(golden['tokenizers']['M3_wordpiece'])}", flush=True)
    print(f"tokenizers.M4_wordpiece: {len(golden['tokenizers']['M4_wordpiece'])}", flush=True)
    print(f"tokenizers.M2_bpe     : {len(golden['tokenizers']['M2_bpe'])}", flush=True)
    print(f"json bytes            : {os.path.getsize(out_path)}", flush=True)
    print("=========================", flush=True)


if __name__ == "__main__":
    main()
