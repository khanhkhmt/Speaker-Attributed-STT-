# Speaker-Attributed STT (`sastt`)

Implementation of [`docs/production-technical-spec.md`](docs/production-technical-spec.md) v1.0.
The specification is the requirement source; every module names the sections it implements.

## Status — Milestone 0 complete (spec 18)

| Milestone | Scope | State |
|---|---|---|
| **M0 — Foundation and contracts** | package, config validation, domain models, ports, JSON Schema v2, fake adapters, state machine, revision/idempotency primitives, CI | **done** |
| M1 — Offline 2-speaker path | real pyannote / faster-whisper / MossFormer2 / CAM++ adapters, model manifests pinned, smoke tests | not started |
| M2 — Linking and Voice ID | pgvector registry, enrollment quality, deletion/audit | not started |
| M3 — Near-realtime | WebSocket ingest, queues, backpressure, latency instrumentation | not started |
| M4 — Beta/advanced overlap | SepFormer 3mix, concurrent counter, GSS, WeSep | not started |
| M5 — Production hardening | benchmark corpus, calibrators, load/soak, SBOM, capacity | not started |

M0 acceptance (spec 18): scenarios **S01**, **S04** and **S12** pass structurally on fake
adapters. S02, S03, S11 and S13 are covered too, since the fakes support them.

## What is deliberately absent

* **No model weights and no accuracy claims.** The fake adapters are integration fixtures.
  Spec 19.1 states the deterministic harness cannot evaluate model quality, and spec 18
  rule 6 forbids presenting an oracle as a model test — so `tests/model/` skips with a
  reason instead of falling back to fakes.
* **No invented confidences.** Every component confidence is `null` with
  `confidence_status="uncalibrated"` until a calibration release exists (spec 0.3).
* **No calibrated thresholds.** `source_linking` and `voice_id` thresholds ship as `null`
  and the pipeline fails closed: sources become `Unknown`/temporary rather than guessed
  (spec 5.10, 18 rule 7). Tests that exercise linking supply thresholds explicitly.
* **No research checkpoint in production.** `mono_four_five_source_research` is refused at
  startup in a production environment, and Multi-Decoder DPRNN is `deny` in its manifest
  (spec 0.2, 20).

## Layout (spec 17)

```
src/sastt/
  domain/       audio.py events.py speakers.py transcript.py errors.py
  ports/        audio.py diarization.py separation.py asr.py embedding.py
                registry.py fusion.py storage.py
  application/  offline_pipeline.py streaming_pipeline.py overlap_router.py
                source_linking.py session_state.py fusion.py
  adapters/     fake/ persistence/            (real model adapters: M1+)
  api/          schemas.py                    (http.py, websocket.py: M1/M3)
  config.py observability.py
configs/            canonical configuration of spec 12
model-manifests/    per-backend licence/pin manifests (spec 11.2, 20)
tests/              unit/ contract/ integration/ model/ load/
```

`ports/audio.py` and `ports/fusion.py` extend the spec 17 tree because spec 9 mandates
`AudioDecoder`, `ConfidenceCalibrator` and `FusionEngine` as ports as well.

## Chạy thử / demo console

```bash
pip install -e ".[dev,api]"
uvicorn --factory sastt.api.http:create_app --app-dir src --port 8000
# mở http://localhost:8000
```

The console runs the built-in scenarios through the real pipeline and shows the
speaker timeline, the transcript with its metadata, and — in realtime mode — the
spec 8.2 event stream with provisional → revision → final and reconnect replay.

`web/` and the `/v1/demo/*` routes are a development aid, not part of the product
spec. The console states the active engine on every page: with the Milestone 0
`fake` engine the output demonstrates pipeline **structure** only and is never a
model result (spec 18 rule 6, 19.1). Uploading real audio returns
`MODEL_NOT_READY` until the Milestone 1 adapters land, rather than pretending.

## Development

```bash
pip install -e ".[dev]"

pytest                 # unit + contract + integration, no weights, no Hub token
pytest -m model        # skips with a reason unless weights are staged in /models
ruff check src tests && ruff format --check src tests
mypy
```

Configuration is loaded and gated in one place:

```python
from sastt.config import load_config
config = load_config("configs/default.yaml", environment="production")
```

A production environment refuses to start on a research flag, a denied or unpinned
checkpoint, or an uncalibrated Voice ID that is not failing closed (spec 12, 20).

## Design invariants worth knowing before changing code

1. Two speakers talking at once produce **two** segments with overlapping timestamps.
   Fusion raises rather than silently dropping a word (spec 0.1.7).
2. Internal time is a sample index or integer milliseconds; floats are never accumulated
   (spec 0.3, 5.1.7).
3. A separator returns waveforms, never identities. `source_0` in one chunk and `source_0`
   in the next may be different people; identity comes from Hungarian assignment over
   speaker embeddings (spec 5.4, 5.8).
4. There is no exhaustive permutation path — assignment is cubic, not factorial
   (spec 5.8, 16.3).
5. `session_speaker_id` is stable and never reused; display labels may be revised, and a
   delivered event is superseded, never rewritten (spec 5.7, 6, FR-011).
