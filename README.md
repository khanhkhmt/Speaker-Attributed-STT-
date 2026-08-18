# Speaker-Attributed STT (`sastt`)

Implementation of [`docs/production-technical-spec.md`](docs/production-technical-spec.md) v1.0.
The specification is the requirement source; every module names the sections it implements.

## Status

Full detail — gates, weight pin state, blockers, known debt — lives in
[`docs/implementation-status.md`](docs/implementation-status.md).

Hướng dẫn cài đặt, stage model, chạy API/worker, xử lý queue và sự cố nằm tại
[`docs/running-guide.md`](docs/running-guide.md).

| Milestone | Scope | State |
|---|---|---|
| **M0 — Foundation and contracts** | package, config validation, domain models, ports, JSON Schema v2, fake adapters, state machine, revision/idempotency primitives, CI | **done** |
| **M1 — Offline 2-speaker path** | ffmpeg decode, pyannote / faster-whisper / MossFormer2 / CAM++ adapters, pinned manifests, smoke tests | **done (functional)** — real offline path and plausibility guard verified; production evidence remains pending |
| M2 — Linking and Voice ID | registry, enrollment quality, deletion/audit | **~85%** — local Voice Registry API runs; persistent deploy wiring remains |
| M3 — Near-realtime | queues, backpressure, latency instrumentation | **~60%** — bounded realtime ring + spool, revisions/replay, Prometheus metrics; no measured SLO/autoscale yet |
| M4 — Beta/advanced overlap | SepFormer 3mix, concurrent counter, GSS, WeSep | **~45%** — gated local SepFormer adapter, and the concurrent counter now runs on diarization evidence so the router's K≥3 rows are reachable; 3-source checkpoint, GSS and WeSep pending. **Direction not yet chosen — see [status §6](docs/implementation-status.md).** |
| M5 — Production hardening | benchmark corpus, calibrators, load/soak, SBOM, capacity | **~30%** — local calibration/report/SBOM tooling; corpus and real evidence pending |

M0 acceptance (spec 18): scenarios **S01**, **S04** and **S12** pass structurally on fake
adapters; S02, S03, S11 and S13 are covered too. The latest non-model quality run
completed with **278 passed, 59 deselected**, alongside **34 model tests** on pinned
weights and **21 infrastructure tests** against a real PostgreSQL and Redis. A
real-engine rerun on a 20-minute upload also verified the transcript plausibility
guard: text whose VAD duration or word-timestamp span is physically impossible is
withheld and the job is marked degraded with an explicit warning. This is a safety
guard, not an accuracy claim.

**Work in progress.** Overlap regions still return `Unknown` for most short
sources, and three concurrent speakers cannot be separated at all — the counter now
reports K=3 honestly and the router degrades the region rather than inventing two
sources from three voices. Closing that needs a choice between staging a 3-source
separator and moving to diarization-conditioned ASR, and either way it needs
labelled overlap audio first. Both options, with their costs, are set out in
[status §6](docs/implementation-status.md).

## What is deliberately absent

* **No model weights and no accuracy claims.** The fake adapters are integration fixtures.
  Spec 19.1 states the deterministic harness cannot evaluate model quality, and spec 18
  rule 6 forbids presenting an oracle as a model test — so `tests/model/` skips with a
  reason instead of falling back to fakes.
* **No invented confidences.** A versioned JSON calibration release can be configured; until one is approved and configured, every component confidence is `null` with `confidence_status="uncalibrated"` (spec 0.3).
* **Language handling.** Offline uploads default to Whisper auto-detection, resolved **once per session** from pooled speech and then reused by every ASR call. Identifying per crop asks Whisper to decide from a few hundred milliseconds of a separated overlap source, which is where it emits memorised subtitle credits instead of a transcription (measured: 18% correct at 0.3 s against 100% at 30 s). The console can pin `vi` or `en` per job; the selected hint is part of that job's config version, so an English upload is never implicitly forced through the Vietnamese decoder path.
* **No calibrated thresholds.** `source_linking` and `voice_id` thresholds ship as `null`
  and the pipeline fails closed: sources become `Unknown`/temporary rather than guessed
  (spec 5.10, 18 rule 7). Tests that exercise linking supply thresholds explicitly, and
  the development API and worker load `configs/linking-thresholds.demo.yaml` so linking is
  observable at all — that file declares itself `status: unapproved` because its two
  numbers were never measured. Point `SASTT_LINKING_THRESHOLDS` at an approved calibration
  release to replace it; configure nothing and the null thresholds stand.
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

The console shows the active engine, speaker timeline, transcript metadata and — in
realtime mode — the spec 8.2 event stream with provisional → revision → final and
reconnect replay. `web/` and `/v1/demo/*` are development aids, not product APIs.

The default `fake` engine demonstrates pipeline **structure** only and rejects real
audio with `MODEL_NOT_READY`. To transcribe a real upload, start API and worker with
`SASTT_ENGINE=real`, verify `/readyz`, then follow the queue workflow in the
[runbook](docs/running-guide.md#8-chạy-topology-queue-ở-local-api--worker--hạ-tầng).

## Development

```bash
pip install -e ".[dev]"

pytest                 # unit + contract + integration, no weights, no Hub token
pytest -m model        # skips with a reason unless weights are staged in /models
ruff check src tests deploy && ruff format --check src tests deploy
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
   speaker embeddings, narrowed to the speakers diarization reports as active over the
   region (spec 5.2, 5.4, 5.8).
4. There is no exhaustive permutation path — assignment is cubic, not factorial
   (spec 5.8, 16.3).
5. `session_speaker_id` is stable and never reused; display labels may be revised, and a
   delivered event is superseded, never rewritten (spec 5.7, 6, FR-011).
6. ASR text must be plausible for both VAD-confirmed speech and word timestamps.
   Invalid output is not attributed or emitted; the job is explicitly degraded with
   an `unreliable_*_transcript` warning, while the original audio remains available.
