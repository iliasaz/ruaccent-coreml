export const meta = {
  name: 'ruaccent-phase0-playbooks',
  description: 'Per-model CoreML conversion playbooks + workstream map for the RUAccent port',
  phases: [
    { title: 'Playbooks', detail: 'one agent per model/component -> structured conversion spec' },
    { title: 'Verify', detail: 'adversarial numeric-equivalency review of each playbook' },
    { title: 'Synthesize', detail: 'workstream/dependency map + risk register' },
  ],
}

// Shared context handed to every agent. Facts are CONFIRMED from Phase 0 investigation
// (upstream source read + HF config.json inspected). Agents must verify against the
// real files in the repo before relying on anything.
const SHARED = `
PROJECT: Port RUAccent (Russian lexical stress) to on-device CoreML, numerically
equivalent to the Python/ONNX upstream, shipped as the Swift package RUAccentCoreML.
Targets macOS 15+/iOS 18+ arm64, ANE-resident fp16, validated ON DEVICE.

REPO: /Users/ilia/Developer/ruaccent-coreml
  Upstream Python (read it):     ruaccent-src/ruaccent/*.py
  Downloaded HF configs/tokens:  converter/_inspect/nn/...   (config.json, tokenizer_config.json, etc.)
  Migration plan + learnings:    docs/MIGRATION_PLAN.md  and  CLAUDE.md
  Phase-0 I/O + fixtures (may still be generating): converter/fixtures/model_io.json, fixtures.json

CONFIRMED PIPELINE (process_all, turbo3.1, use_dictionary=True, tiny_mode=False), per sentence:
  normalize(regex) -> razdel sentenize -> word-split(regex)
  -> M3 stress_usage over sentence (per-word STRESS/NO_STRESS/PUNCT; gates which words get accented)
  -> _process_yo: if 'е' in sentence run M4 yo_homograph; then yo_words / yo_homographs dicts
  -> _process_omographs: for words in omographs dict, run M2 omograph cross-encoder over hypotheses
  -> _process_accent: word in STRESS & not already '+': accents-dict lookup; else if >1 vowel
       & no punct -> M1 accent char-model; render '+' before stressed char (label STRESS_PRIMARY, score>=0.55)
  -> postprocess (join, fix_capital, delete_spaces_before_punc). Output marks stress with '+' BEFORE vowel.
  NOTE: RuleEngine (koziev CRF) is loaded but NEVER called in process_all -> out of scope.

FOUR NEURAL MODELS (ONNX upstream; we re-target to CoreML):
  M1 accent       RoFormerForTokenClassification    4L/128h/8heads, char tok vocab=45, max_pos=60,
                  labels NO/STRESS_PRIMARY/STRESS_SECONDARY. ~0.8MB. ROTARY embeddings.
  M2 omograph     DebertaForSequenceClassification  6L/768h/12heads, RoBERTa-BPE vocab=60258
                  (+10002 added '+word' tokens), max_pos=512, binary seq-cls. ~359MB.
                  DEBERTA v1 = disentangled attention + relative position (the hard one).
  M3 stress_usage BertForTokenClassification        3L/312h/12heads, wordpiece vocab=83828,
                  max_pos=2048, labels NO_STRESS/PUNCT/STRESS. ~116MB (embedding-dominated).
  M4 yo_homograph DistilBertForTokenClassification  wordpiece vocab=5031, max_pos=512,
                  labels NO_YO/PUNCT/YO. ~14MB.

CONVERSION CONSTRAINT: coremltools 9.0 has NO ONLINE ONNX frontend. Weights exist only as
ONNX (no safetensors upstream). The ANE-correct path is: rebuild the HF PyTorch module from
config.json, map ONNX initializers -> HF state_dict, torch.jit.trace with STATIC int shapes,
ct.convert(compute_precision=FLOAT16, compute_units=ALL). (Evaluate onnx2torch as a fallback,
but it yields ANE-unfriendly graphs with no control over attention.)

CARRIED ANE LEARNINGS (hard-won, see CLAUDE.md / MIGRATION_PLAN.md):
- fp16 LayerNorm/RMSNorm overflow when activations > ~256 -> inf -> zeroed outputs. Keep norms fp32
  or rescale-in-fp16.
- ANE fused attention DROPS the explicit attn_mask at q_len>>1 -> manually decompose SDPA
  (matmul(q,kT)*scale + mask -> softmax -> matmul) for any prefill/encoder attention.
- Keep ALL traced shapes static python ints (any .shape-derived size -> aten::Int -> frontend reject).
  Inline positional encodings (rotary / relative-position helpers) that emit dynamic ops.
- Palettize (8-bit k-means) for size; skip the classifier/output head.
- Mac CoreML parity != iPhone ANE. Every model must be device-validated.
`

const SPEC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['component', 'io_contract', 'rebuild_strategy', 'ane_risks', 'conversion_steps', 'parity_plan', 'size_strategy', 'open_questions', 'confidence'],
  properties: {
    component: { type: 'string' },
    io_contract: {
      type: 'object', additionalProperties: false,
      required: ['inputs', 'outputs', 'notes'],
      properties: {
        inputs: { type: 'array', items: { type: 'string' }, description: 'name: dtype [shape] — meaning' },
        outputs: { type: 'array', items: { type: 'string' } },
        notes: { type: 'string' },
      },
    },
    rebuild_strategy: { type: 'string', description: 'How to obtain a traceable fp16 PyTorch module + load ONNX weights; the arch-specific gotcha' },
    ane_risks: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['risk', 'mitigation'], properties: { risk: { type: 'string' }, mitigation: { type: 'string' } } } },
    conversion_steps: { type: 'array', items: { type: 'string' } },
    parity_plan: { type: 'string', description: 'Exact numeric-equivalency test: oracle, tolerance, on-device step' },
    size_strategy: { type: 'string' },
    open_questions: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const COMPONENTS = [
  { key: 'M1_accent_RoFormer', prompt: `Produce the CoreML conversion playbook for **M1 accent** (RoFormerForTokenClassification, char-level OOV accentor). Read ruaccent-src/ruaccent/accent_model.py and char_tokenizer.py and converter/_inspect/nn/nn_accent/*. Nail: exact ONNX inputs (input_ids/attention_mask/token_type_ids?) and the char tokenizer ([bos]/[eos], vocab.txt order, the U+0301 char in vocab!), how RoFormer rotary must be inlined as static ops, render_stress logic (only STRESS_PRIMARY, score>=0.55, '+' before the char). Smallest model => ideal first end-to-end conversion to prove the toolchain.` },
  { key: 'M2_omograph_DeBERTa', prompt: `Produce the CoreML conversion playbook for **M2 omograph** (DebertaForSequenceClassification, 359MB, the HARD one). Read ruaccent-src/ruaccent/omograph_model.py and converter/_inspect/nn/nn_omograph/turbo3.1/*. Focus on DeBERTa-v1 DISENTANGLED ATTENTION + relative position bias: how to express it with static shapes and manual SDPA on ANE, whether to cap/bucket relative positions, the cross-encoder I/O (text + hypothesis pair, the <w>..</w> marking, the 10002 added '+word' tokens in the BPE vocab). Address the 359MB size (palettization, ANE residency, whether to split). This is the dominant risk — be concrete and skeptical.` },
  { key: 'M3_stress_usage_BERT', prompt: `Produce the CoreML conversion playbook for **M3 stress_usage** (BertForTokenClassification, 116MB). Read ruaccent-src/ruaccent/stress_usage_model.py and converter/_inspect/nn/nn_stress_usage_predictor/*. Standard BERT (absolute pos, max_pos=2048, vocab 83828 dominates size). Cover: I/O (input_ids/attention_mask/token_type_ids), manual-SDPA for the attn_mask-at-q_len>>1 issue, the subword->word AVERAGE aggregation done in Python (stays in Swift, not the model), and whether the huge embedding table should stay fp16 vs palettized.` },
  { key: 'M4_yo_homograph_DistilBERT', prompt: `Produce the CoreML conversion playbook for **M4 yo_homograph** (DistilBertForTokenClassification, 14MB, the EASIEST). Read ruaccent-src/ruaccent/yo_homograph_model.py and converter/_inspect/nn/nn_yo_homograph_resolver/*. DistilBERT has no token_type_ids and no pooler. Cover I/O, manual SDPA, and propose this as the 2nd conversion after M1 to lock the BERT-family recipe before tackling M2/M3.` },
  { key: 'tokenizers_swift_port', prompt: `Produce the spec for porting ALL FOUR tokenizers to Swift with byte-exact parity: (1) char tokenizer (vocab.txt, list(text), [bos]/[eos]); (2) RoBERTa BPE for M2 (vocab.json+merges.txt+10002 added tokens, byte-level, add_prefix_space); (3) BERT wordpiece for M3 (vocab 83828, do_lower_case?, strip_accents, basic tokenize); (4) DistilBERT wordpiece for M4. Read char_tokenizer.py + the *_model.py token handling + converter/_inspect tokenizer_config.json/special_tokens_map. Identify which need offset_mapping (M3/M4 use return_offsets_mapping for subword->word alignment) and how to reproduce that in Swift. Flag the hardest parity traps (byte-level BPE, NFKD, ё/й).` },
  { key: 'dicts_and_orchestration', prompt: `Produce the spec for the NON-model components: (a) dictionaries — accents (full vs accents_nn), omographs, yo_words, yo_homographs: formats (read converter/_inspect/dictionary samples), sizes, packed on-device representation + Swift loader, the ~11MB target vs the real 21MB-gz full accents dict (recommend which to ship); (b) preprocessing/orchestration to reproduce in Swift: the normalize regex, razdel sentenize (how to port razdel's rule-based sentence splitter — or whether text_split.py is the real splitter), split_by_words regex, delete_spaces_before_punc, fix_capital, and the exact resolution order in process_all_internal. Read ruaccent-src/ruaccent/ruaccent.py, text_preprocessor.py, text_split.py, text_postprocessor.py.` },
]

phase('Playbooks')
const playbooks = await pipeline(
  COMPONENTS,
  (c) => agent(`${SHARED}\n\n=== YOUR TASK ===\n${c.prompt}\n\nReturn a precise, skeptical, implementation-ready spec. Verify claims against the actual files; do not invent I/O — if unknown, say so in open_questions.`,
    { label: `spec:${c.key}`, phase: 'Playbooks', schema: SPEC_SCHEMA }),
  // adversarial numeric-equivalency review of each playbook, as soon as it lands
  (spec, c) => agent(`${SHARED}\n\nAdversarially review this CoreML conversion playbook for **${c.key}** for anything that would BREAK numeric equivalency with the Python/ONNX upstream or fail on ANE. Be a skeptic: hunt for wrong I/O, ignored fp16-overflow/attn-mask/rotary/relative-position traps, tokenizer mismatches, missing static-shape handling, untested device steps. Return the SAME schema with fields corrected/sharpened and every weakness added to ane_risks or open_questions. Playbook:\n\n${JSON.stringify(spec)}`,
    { label: `verify:${c.key}`, phase: 'Verify', schema: SPEC_SCHEMA }),
)

phase('Synthesize')
const map = await agent(
  `${SHARED}\n\nYou are given ${playbooks.length} verified per-component conversion playbooks for the RUAccent->CoreML port. Synthesize the PROGRAM PLAN:\n` +
  `1) A parallel-workstream/dependency map: which workstreams run concurrently vs. what gates what, from Phase 0 -> on-device end-to-end parity -> chatterbox integration. Account for FOUR models (not two), four tokenizers, dictionaries, preprocessing.\n` +
  `2) A ranked RISK REGISTER (highest first) with the single mitigation for each.\n` +
  `3) A recommended EXECUTION ORDER (which model to convert first to de-risk the toolchain, and why), and which GitHub phase issues must be added/renumbered (current issues #2-#9 assume only 2 models).\n` +
  `Playbooks:\n${JSON.stringify(playbooks)}`,
  { label: 'synthesize:program-plan', phase: 'Synthesize' },
)

return { playbooks, program_plan: map }
