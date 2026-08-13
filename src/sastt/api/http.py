"""HTTP surface — spec 8.1, 8.2, 8.4.

The routes follow the spec 8 tables. Two deliberate deviations, both fenced off
so they cannot reach production:

* jobs run **in-process**, not on the queue/worker topology of spec 11.1/11.3,
  which arrives with Milestone 3;
* the tenant comes from a development auth stub. Spec 14.2 requires the tenant
  to come from auth claims, so the stub refuses to start in a production
  environment.

Which engine is wired in is explicit in every response (``engine`` field) and on
the readiness probe: with the ``fake`` engine the output is a structural
demonstration, never a model result (spec 18 rule 6, 19.1).
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from sastt.adapters.fake.scenario import Scenario
from sastt.adapters.persistence import InMemoryJobStore
from sastt.adapters.persistence.memory import IdempotencyConflictError
from sastt.application.offline_pipeline import OfflinePipeline, OfflineResult, PipelineAdapters
from sastt.application.streaming_pipeline import StreamingSession
from sastt.config import Environment, SasttConfig, load_config, load_manifests
from sastt.domain.audio import CANONICAL_SAMPLE_RATE
from sastt.domain.errors import ConfigurationError, ErrorCode, ModelNotReadyError, SasttError
from sastt.domain.events import JobRecord, JobState, new_id
from sastt.domain.transcript import render_transcript
from sastt.observability import CallContext, InMemoryMetrics, tenant_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = REPO_ROOT / "web"
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures"

DEV_TENANT_HEADER = "X-Tenant-Id"
DEFAULT_DEV_TENANT = "tenant_dev"


# --------------------------------------------------------------------------- #
# Engine wiring
# --------------------------------------------------------------------------- #


@dataclass
class Engine:
    """Which adapters back this process."""

    name: str
    config: SasttConfig
    #: Built per scenario while the fake engine is in use.
    adapters_for: Any = None
    ready: bool = True
    detail: str = ""


def build_fake_engine(config: SasttConfig) -> Engine:
    """Milestone 0 adapters: deterministic, weight-free, **not** a model result."""
    from sastt.adapters.fake import (
        FakeAudioDecoder,
        FakeDiarizer,
        FakeOverlapDetector,
        FakeSpeakerEmbedder,
        FakeSpeechRecognizer,
        FakeSpeechSeparator,
        FakeVoiceActivityDetector,
    )

    def adapters_for(scenario: Scenario) -> PipelineAdapters:
        return PipelineAdapters(
            decoder=FakeAudioDecoder(sample_rate=scenario.sample_rate),
            vad=FakeVoiceActivityDetector(),
            diarizer=FakeDiarizer(scenario.speakers),
            overlap_detector=FakeOverlapDetector(scenario.speakers),
            recognizer=FakeSpeechRecognizer(scenario),
            embedder=FakeSpeakerEmbedder(scenario.speakers),
            separator=FakeSpeechSeparator(scenario.speakers, scenario=scenario),
        )

    return Engine(
        name="fake",
        config=config,
        adapters_for=adapters_for,
        ready=True,
        detail=(
            "Milestone 0 fake adapters: structure only. Spec 19.1 — this harness "
            "cannot evaluate model accuracy."
        ),
    )


def build_real_engine(config: SasttConfig) -> Engine:
    """Real model adapters — Milestone 1 (spec 18)."""
    raise ModelNotReadyError(
        "the real model adapters land in Milestone 1; run with SASTT_ENGINE=fake",
        details={"engine": "real"},
    )


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


@dataclass
class AppState:
    config: SasttConfig
    engine: Engine
    environment: Environment
    jobs: InMemoryJobStore = field(default_factory=InMemoryJobStore)
    results: dict[str, OfflineResult] = field(default_factory=dict)
    job_scenarios: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, StreamingSession] = field(default_factory=dict)
    metrics: InMemoryMetrics = field(default_factory=InMemoryMetrics)


def available_scenarios() -> dict[str, Path]:
    return {path.stem: path for path in sorted(SCENARIO_DIR.glob("s*.json"))}


def load_demo_scenario(name: str) -> Scenario:
    paths = available_scenarios()
    if name not in paths:
        raise HTTPException(status_code=404, detail=f"unknown demo scenario {name!r}")
    return Scenario.load(paths[name])


def scenario_wav(scenario: Scenario) -> bytes:
    """Render a scenario to a 16-bit mono WAV (the stream format of spec 1.1)."""
    samples = scenario.render().samples[0]
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    rate = scenario.sample_rate
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    wav: bytes = header + pcm
    return wav


# --------------------------------------------------------------------------- #
# Auth stub (development only — spec 14.2)
# --------------------------------------------------------------------------- #


def dev_tenant(request: Request, x_tenant_id: str | None = Header(default=None)) -> str:
    """Resolve the tenant.

    Spec 14.2: the tenant comes from auth claims and a client-supplied
    ``tenant_id`` is never trusted. This header stub exists for local
    development only; :func:`create_app` refuses to start it in production.
    """
    state: AppState = request.app.state.sastt
    if state.environment.is_production:  # pragma: no cover - guarded at startup
        raise HTTPException(status_code=500, detail="auth stub must not run in production")
    return x_tenant_id or DEFAULT_DEV_TENANT


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def create_app(
    config: SasttConfig | None = None,
    *,
    engine_name: str = "fake",
    environment: Environment | str = Environment.DEVELOPMENT,
) -> FastAPI:
    env = environment if isinstance(environment, Environment) else Environment(environment)
    config = config or load_config(
        REPO_ROOT / "configs" / "default.yaml",
        environment=env,
        manifest_dir=REPO_ROOT / "model-manifests",
        # Linking thresholds are null in the shipped config and the pipeline then
        # fails closed (spec 5.10). The demo supplies a calibration-shaped
        # override so speaker linking is observable; a real deployment gets these
        # from a calibration release (spec 21.3).
        overrides={"source_linking": {"accept_threshold": 0.55, "ambiguous_margin": 0.10}},
    )
    if env.is_production:
        raise ConfigurationError(
            "this API ships a development auth stub and an in-process job runner; "
            "production needs real auth and the worker topology of spec 11.1"
        )

    engine = build_fake_engine(config) if engine_name == "fake" else build_real_engine(config)
    app = FastAPI(title="Speaker-Attributed STT", version="0.1.0")
    app.state.sastt = AppState(config=config, engine=engine, environment=env)

    from sastt.api.websocket import register_websocket_routes

    register_websocket_routes(app)

    @app.exception_handler(SasttError)
    async def _domain_error(_: Request, exc: SasttError) -> JSONResponse:
        status = 503 if exc.code is ErrorCode.MODEL_NOT_READY else 400
        return JSONResponse(status_code=status, content=exc.to_dict())

    # -- probes (spec 11.2) -------------------------------------------------- #

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        state: AppState = app.state.sastt
        manifests = load_manifests(REPO_ROOT / "model-manifests")
        return {
            "status": "ready" if state.engine.ready else "not_ready",
            "engine": state.engine.name,
            "engine_detail": state.engine.detail,
            "environment": state.environment.value,
            "config_version": state.config.config_version,
            "max_session_speakers": state.config.product.max_session_speakers,
            "max_supported_concurrent_speakers": (
                state.config.product.max_supported_concurrent_speakers
            ),
            "models": [
                {
                    "backend": manifest.backend,
                    "pinned": manifest.is_pinned,
                    "revision": manifest.revision,
                    "production_action": manifest.production_action.value,
                }
                for manifest in sorted(manifests.values(), key=lambda m: m.backend)
            ],
        }

    # -- demo scenarios (development aid, not part of the spec) -------------- #

    @app.get("/v1/demo/scenarios")
    async def list_scenarios() -> dict[str, Any]:
        items = []
        for name, path in available_scenarios().items():
            scenario = Scenario.load(path)
            items.append(
                {
                    "name": name,
                    "title": scenario.name,
                    "speakers": list(scenario.speakers),
                    "duration_ms": scenario.duration_ms,
                    "overlap_regions": len(scenario.overlap_intervals()),
                }
            )
        return {"scenarios": items}

    @app.get("/v1/demo/scenarios/{name}/audio.wav")
    async def scenario_audio(name: str) -> Response:
        return Response(content=scenario_wav(load_demo_scenario(name)), media_type="audio/wav")

    # -- offline jobs (spec 8.1) --------------------------------------------- #

    @app.post("/v1/jobs", status_code=201)
    async def create_job(
        payload: dict[str, Any],
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        tenant_id: str = Depends(dev_tenant),
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
        state: AppState = request.app.state.sastt

        scenario_name = payload.get("scenario")
        if not scenario_name:
            raise ModelNotReadyError(
                "uploading real audio needs the Milestone 1 model adapters; "
                'use {"scenario": "..."} against the fake engine for now',
            )
        scenario = load_demo_scenario(str(scenario_name))
        audio = scenario_wav(scenario)

        try:
            job, created = state.jobs.create_or_get(
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                input_hash=str(hash(audio)),
                config_version=state.config.config_version,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=exc.message) from exc

        if created:
            state.job_scenarios[job.job_id] = str(scenario_name)
            _run_job(state, job, scenario, audio, tenant_id)
        return _job_view(state, job, created)

    @app.get("/v1/jobs/{job_id}")
    async def get_job(
        job_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        return _job_view(state, state.jobs.get(tenant_id, job_id), created=False)

    @app.get("/v1/jobs/{job_id}/result")
    async def get_job_result(
        job_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        job = state.jobs.get(tenant_id, job_id)
        result = state.results.get(job.job_id)
        if result is None:
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}, not finished")
        return {
            "job_id": job.job_id,
            "engine": state.engine.name,
            "schema_version": "2.0",
            "state": job.state.value,
            "config_version": result.config_version,
            "model_versions": result.model_versions.to_dict(),
            "warnings": result.warnings,
            "degraded_mode": result.degraded,
            "estimated_session_speakers": result.estimated_session_speakers,
            "segments": [segment.to_public_dict() for segment in result.segments],
            "text": render_transcript(result.segments),
        }

    @app.delete("/v1/jobs/{job_id}", status_code=202)
    async def delete_job(
        job_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        job = state.jobs.get(tenant_id, job_id)
        state.results.pop(job.job_id, None)
        if not job.is_terminal:
            job.transition(JobState.CANCELLED)
        return {"job_id": job.job_id, "state": job.state.value}

    # -- realtime sessions (spec 8.2) ---------------------------------------- #

    @app.post("/v1/sessions", status_code=201)
    async def create_session(
        payload: dict[str, Any], request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        scenario = load_demo_scenario(str(payload.get("scenario") or "s02_two_speaker_overlap"))
        session_id = new_id("ses")
        state.sessions[session_id] = StreamingSession(
            session_id=session_id,
            config=state.config,
            adapters=state.engine.adapters_for(scenario),
            tenant_id=tenant_id,
        )
        return {
            "session_id": session_id,
            "engine": state.engine.name,
            "websocket_url": f"/v1/sessions/{session_id}/audio",
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "frame_ms": state.config.streaming.frame_ms,
            "config_version": state.config.config_version,
        }

    @app.post("/v1/sessions/{session_id}/finalize")
    async def finalize_session(session_id: str, request: Request) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        session = _session(state, session_id)
        events = session.finalize()
        return {"events": [event.to_dict() for event in events]}

    @app.get("/v1/sessions/{session_id}/result")
    async def session_result(session_id: str, request: Request) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        session = _session(state, session_id)
        segments = session.result()
        return {
            "session_id": session_id,
            "schema_version": "2.0",
            "segments": [segment.to_public_dict() for segment in segments],
            "text": render_transcript(segments),
        }

    # -- the demo UI ---------------------------------------------------------- #

    @app.get("/")
    async def index() -> Response:
        page = WEB_DIR / "index.html"
        if not page.exists():  # pragma: no cover - only when web/ is removed
            raise HTTPException(status_code=404, detail="web/index.html is missing")
        return FileResponse(page)

    return app


def _session(state: AppState, session_id: str) -> StreamingSession:
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
    return session


def _job_view(state: AppState, job: JobRecord, created: bool) -> dict[str, Any]:
    result = state.results.get(job.job_id)
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "created": created,
        "engine": state.engine.name,
        "scenario": state.job_scenarios.get(job.job_id),
        "config_version": job.config_version,
        "warnings": result.warnings if result else [],
        "degraded_mode": bool(result and result.degraded),
        "error_code": job.error_code,
    }


def _run_job(
    state: AppState,
    job: JobRecord,
    scenario: Scenario,
    audio: bytes,
    tenant_id: str,
) -> None:
    """Run one job in-process (the worker topology of spec 11.1 arrives in M3)."""
    pipeline = OfflinePipeline(
        state.config, state.engine.adapters_for(scenario), tenant_id=tenant_id
    )
    ctx = CallContext(
        stage="offline_job",
        job_id=job.job_id,
        tenant_hash=tenant_hash(tenant_id),
        metrics=state.metrics,
    )
    job.transition(JobState.PREPROCESSING)
    try:
        # WAV header is stripped: the fake decoder consumes raw PCM s16le.
        result = pipeline.run(_strip_wav_header(audio), ctx, job=job)
    except SasttError as exc:
        job.error_code = exc.code.value if exc.code else "INTERNAL"
        job.transition(JobState.FAILED)
        raise
    state.results[job.job_id] = result
    job.transition(result.state)


def _strip_wav_header(payload: bytes) -> bytes:
    if not payload.startswith(b"RIFF"):
        return payload
    stream = io.BytesIO(payload)
    stream.seek(12)
    while chunk_header := stream.read(8):
        name, size = struct.unpack("<4sI", chunk_header)
        if name == b"data":
            return stream.read(size)
        stream.seek(size, io.SEEK_CUR)
    return payload


app_factory = create_app

__all__ = ["AppState", "Engine", "create_app", "scenario_wav"]
