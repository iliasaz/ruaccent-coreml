# converter/

Python → CoreML conversion + fixture-dump scripts for the RUAccent port.

Planned (see `../docs/MIGRATION_PLAN.md`):
- `dump_ruaccent_fixtures.py` — run upstream RUAccent over a Russian fixture set and dump
  `(input → stressed)` ground truth (Phase 0).
- `convert_accentor_coreml.py` — neural accentor model → CoreML fp16 + ANE validation (Phase 1).
- `convert_homograph_coreml.py` — context disambiguation model → CoreML (Phase 2).
- `pack_dictionaries.py` — extract the `accents`/`omographs` subset → packed resource (Phase 3).

**All generated outputs are gitignored** (`.mlpackage`, `.npy`, weights, `ruaccent-src/`,
`.venv`). Keep only scripts under version control. Mirror the chatterbox `.venv`
(coremltools 9.0 / torch 2.8 / Python 3.13).
