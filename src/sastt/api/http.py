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

import base64
import binascii
import hashlib
import io
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from sastt.adapters.fake.scenario import Scenario
from sastt.adapters.persistence import InMemoryJobStore, InMemoryObjectStore
from sastt.adapters.persistence.memory import IdempotencyConflictError
from sastt.adapters.queue.redis_queue import QUEUE_SPEAKER_BATCH, RedisTaskQueue, build_client
from sastt.adapters.storage import S3ObjectStore
from sastt.application.fusion import ConfidenceCalibrator, load_confidence_calibrator
from sastt.application.offline_pipeline import OfflinePipeline, OfflineResult, PipelineAdapters
from sastt.application.streaming_pipeline import StreamingSession
from sastt.config import (
    Environment,
    SasttConfig,
    load_config,
    load_linking_overlay,
    load_manifests,
)
from sastt.domain.audio import CANONICAL_SAMPLE_RATE, AudioBuffer, TimeInterval, seconds_to_ms
from sastt.domain.errors import (
    ConfigurationError,
    ErrorCode,
    ModelNotReadyError,
    SasttError,
    TenantAccessDeniedError,
)
from sastt.domain.events import JobRecord, JobState, new_id
from sastt.domain.transcript import render_transcript
from sastt.observability import CallContext, MetricsSink, PrometheusMetrics, tenant_hash
from sastt.ports.registry import VoiceRegistry
from sastt.ports.storage import JobStore, ObjectStore

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = REPO_ROOT / "web"
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_LINKING_THRESHOLDS = REPO_ROOT / "configs" / "linking-thresholds.demo.yaml"

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
            three_source_separator=(
                FakeSpeechSeparator(
                    scenario.speakers,
                    backend="fake_three_source",
                    supported_source_counts=(3,),
                    scenario=scenario,
                )
                if config.product.three_source_beta
                else None
            ),
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
    """Milestone 1 adapters: pyannote, faster-whisper, MossFormer2 and CAM++.

    Every model version string is the manifest ``release_id``, so each output
    records the exact pinned revision it came from (spec FR-013, 11.2). A
    missing or unpinned checkpoint raises :class:`ModelNotReadyError` instead of
    quietly degrading to a fake (spec 18 rule 6).
    """
    from sastt.adapters.clearvoice import MossFormer2Separator
    from sastt.adapters.faster_whisper import FasterWhisperRecognizer, SileroVoiceActivityDetector
    from sastt.adapters.ffmpeg import FfmpegAudioDecoder
    from sastt.adapters.pyannote import PyannoteDiarizer, PyannoteOverlapDetector
    from sastt.adapters.speaker3d import CamPlusPlusEmbedder
    from sastt.adapters.speechbrain import SepFormerLibri3MixSeparator

    manifests = load_manifests(REPO_ROOT / "model-manifests")

    def version(backend: str) -> str:
        manifest = manifests.get(backend)
        return manifest.release_id if manifest else f"{backend}@unpinned"

    def path(value: str | None, what: str) -> str:
        if not value:
            raise ModelNotReadyError(f"no model path configured for {what}")
        return value

    diarizer = PyannoteDiarizer(
        path(config.diarization.model_path, "diarization"),
        model_version=version("pyannote-community-1"),
        exclusive_tracks=config.diarization.exclusive_output_for_non_overlap_alignment,
    )
    overlap_detector = PyannoteOverlapDetector(
        path(config.overlap_detection.model_path, "overlap detection"),
        model_version=version("pyannote_segmentation_3.0"),
        onset=config.overlap_detection.onset,
        offset=config.overlap_detection.offset,
        min_duration_ms=seconds_to_ms(config.overlap_detection.min_duration_seconds),
        merge_gap_ms=seconds_to_ms(config.overlap_detection.merge_gap_seconds),
    )
    recognizer = FasterWhisperRecognizer(
        path(config.asr.realtime_model_path, "ASR"),
        model_version=version("faster_whisper"),
        compute_type=config.asr.compute_type,
        language=config.asr.language,
    )
    embedder = CamPlusPlusEmbedder(
        path(config.speaker_embedding.model_path, "speaker embedding"),
        model_version=version("3d_speaker_campplus"),
        minimum_speech_ms=seconds_to_ms(config.speaker_embedding.minimum_clean_speech_seconds),
        target_speech_ms=seconds_to_ms(config.speaker_embedding.target_clean_speech_seconds),
    )
    separator = MossFormer2Separator(
        path(config.separation.two_source_model_path, "two-source separation"),
        separator_version=version("mossformer2_ss_16k"),
    )
    three_source_separator = (
        SepFormerLibri3MixSeparator(
            path(config.separation.three_source_model_path, "three-source separation"),
            separator_version=version("sepformer_libri3mix"),
        )
        if config.product.three_source_beta
        else None
    )
    bundle = PipelineAdapters(
        decoder=FfmpegAudioDecoder(max_hours=config.audio.max_file_hours),
        vad=SileroVoiceActivityDetector(),
        diarizer=diarizer,
        overlap_detector=overlap_detector,
        recognizer=recognizer,
        embedder=embedder,
        separator=separator,
        three_source_separator=three_source_separator,
    )

    unpinned = [name for name, manifest in manifests.items() if not manifest.is_pinned]
    return Engine(
        name="real",
        config=config,
        adapters_for=lambda _scenario=None: bundle,
        ready=True,
        detail=(
            "Milestone 1 adapters on pinned weights."
            + (f" Unpinned backends present: {', '.join(sorted(unpinned))}." if unpinned else "")
        ),
    )


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


@dataclass
class AppState:
    config: SasttConfig
    engine: Engine
    environment: Environment
    jobs: JobStore = field(default_factory=InMemoryJobStore)
    objects: ObjectStore = field(default_factory=InMemoryObjectStore)
    task_queue: RedisTaskQueue | None = None
    registry: VoiceRegistry | None = None
    identity_metadata: dict[tuple[str, str], dict[str, str | None]] = field(default_factory=dict)
    results: dict[str, OfflineResult] = field(default_factory=dict)
    job_scenarios: dict[str, str] = field(default_factory=dict)
    #: ``job_id -> (tenant_id, object key)`` for the retained input audio. The
    #: tenant travels with the key because eviction happens outside the request
    #: that stored it, and an object must never be deleted under another tenant.
    job_audio_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: ``job_id -> playback re-encode``. Derived data, cached, evicted with its input.
    job_audio_previews: dict[str, bytes] = field(default_factory=dict)
    job_languages: dict[str, str | None] = field(default_factory=dict)
    sessions: dict[str, StreamingSession] = field(default_factory=dict)
    metrics: MetricsSink = field(default_factory=PrometheusMetrics)
    calibrator: ConfidenceCalibrator = field(
        default_factory=lambda: load_confidence_calibrator(None)
    )


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
# Input audio retained for playback (development console aid)
# --------------------------------------------------------------------------- #

#: How many job inputs the *in-memory* object store keeps so the console can play
#: a segment back. Matches the console's own history depth, so every job still
#: listed is still playable. Raw audio is biometric data (spec 10.3, 14.4): a
#: real deployment keeps it in object storage under a retention policy, not in
#: the API process, so this bound only guards development RAM. Nothing evicts
#: objects from a real ``ObjectStore`` — the worker still has to read them.
MAX_RETAINED_JOB_AUDIO = 12

#: Playback-only re-encode of the input, for listening back over a slow link.
#:
#: Mono 16 kHz is what the pipeline itself works in, so nothing a labeller needs
#: to hear is lost; a 20-minute upload goes from ~15 MB to ~3.4 MB. Measured on
#: real audio: the re-encode stays time-aligned with the original to 0.0 ms
#: (cross-correlation peak 0.996), so seeking to a segment still lands on it.
#: ``compression_level 0`` is deliberate — it costs ~3 s instead of ~24 s for the
#: same size, and the difference never reaches a transcript.
AUDIO_PREVIEW_ARGS: tuple[str, ...] = (
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "libopus",
    "-b:a",
    "24k",
    "-compression_level",
    "0",
    "-f",
    "ogg",
)
AUDIO_PREVIEW_MEDIA_TYPE = "audio/ogg"

#: Container signatures of spec 1.1, longest prefix first.
_AUDIO_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "audio/wav"),
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
)


def job_audio_key(job_id: str) -> str:
    """Where a job's input audio lives. Derivable, so a restart can still find it."""
    return f"jobs/{job_id}/input"


def audio_media_type(payload: bytes) -> str:
    """Sniff the container so the browser gets a type it can decode.

    The upload arrives as opaque bytes — the API never asked for a filename — so
    the byte signature is the only honest source. An unrecognised container is
    reported as such rather than mislabelled as WAV.
    """
    for signature, media_type in _AUDIO_SIGNATURES:
        if payload.startswith(signature):
            return media_type
    if payload[4:8] == b"ftyp":
        return "audio/mp4"
    if payload[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3"):
        return "audio/mpeg"
    return "application/octet-stream"


def build_audio_preview(payload: bytes) -> bytes | None:
    """Re-encode the input for listening back, or ``None`` if ffmpeg cannot.

    Returning ``None`` rather than raising keeps playback working on a build
    without libopus: the endpoint falls back to the exact bytes, which is slower
    over a tunnel but never wrong.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["ffmpeg", "-v", "error", "-i", "pipe:0", *AUDIO_PREVIEW_ARGS, "pipe:1"],
            input=payload,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def _retain_job_audio(state: AppState, tenant_id: str, job_id: str, audio: bytes) -> str:
    """Store the input so ``GET /v1/jobs/{id}/audio`` can serve it back."""
    key = job_audio_key(job_id)
    state.objects.put(tenant_id, key, audio)
    state.job_audio_keys[job_id] = (tenant_id, key)
    if isinstance(state.objects, InMemoryObjectStore):
        while len(state.job_audio_keys) > MAX_RETAINED_JOB_AUDIO:
            evicted_job, (evicted_tenant, evicted_key) = next(iter(state.job_audio_keys.items()))
            del state.job_audio_keys[evicted_job]
            state.objects.delete(evicted_tenant, evicted_key)
    # A preview is derived data: it must never outlive the input it came from.
    for stale in [job for job in state.job_audio_previews if job not in state.job_audio_keys]:
        del state.job_audio_previews[stale]
    return key


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
    engine_name: str | None = None,
    environment: Environment | str = Environment.DEVELOPMENT,
    jobs: JobStore | None = None,
    objects: ObjectStore | None = None,
    task_queue: RedisTaskQueue | None = None,
    registry: VoiceRegistry | None = None,
) -> FastAPI:
    env = environment if isinstance(environment, Environment) else Environment(environment)
    engine_name = engine_name or os.environ.get("SASTT_ENGINE", "fake")
    config = config or load_config(
        REPO_ROOT / "configs" / "default.yaml",
        environment=env,
        manifest_dir=REPO_ROOT / "model-manifests",
        # Linking thresholds are null in the shipped config and the pipeline then
        # fails closed (spec 5.10). Development loads an overlay file so linking
        # is observable; a deployment points SASTT_LINKING_THRESHOLDS at an
        # approved calibration release instead (spec 21.3).
        overrides=load_linking_overlay(DEFAULT_LINKING_THRESHOLDS),
    )
    if env.is_production:
        raise ConfigurationError(
            "this API ships a development auth stub and an in-process job runner; "
            "production needs real auth and the worker topology of spec 11.1"
        )

    engine = build_fake_engine(config) if engine_name == "fake" else build_real_engine(config)
    if os.environ.get("SASTT_JOB_RUNNER") == "queue" and task_queue is None:
        from sastt.adapters.persistence.postgres import PostgresJobStore, build_pool

        jobs = PostgresJobStore(build_pool(os.environ.get("DATABASE_URL")))
        objects = S3ObjectStore.from_environment()
        task_queue = RedisTaskQueue(build_client(os.environ.get("REDIS_URL")))
    app = FastAPI(title="Speaker-Attributed STT", version="0.1.0")
    app.state.sastt = AppState(
        config=config,
        engine=engine,
        environment=env,
        jobs=jobs or InMemoryJobStore(),
        objects=objects or InMemoryObjectStore(),
        task_queue=task_queue,
        registry=registry or _build_local_registry(engine, config),
        calibrator=load_confidence_calibrator(config.confidence.calibration_path),
    )

    from sastt.api.websocket import register_websocket_routes

    register_websocket_routes(app)

    @app.exception_handler(SasttError)
    async def _domain_error(_: Request, exc: SasttError) -> JSONResponse:
        status = (
            503
            if exc.code is ErrorCode.MODEL_NOT_READY
            else 429
            if exc.code is ErrorCode.QUEUE_OVERLOADED
            else 400
        )
        return JSONResponse(status_code=status, content=exc.to_dict())

    # -- probes (spec 11.2) -------------------------------------------------- #

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Prometheus text exposition for the metrics named in spec 13.1."""
        state: AppState = app.state.sastt
        if isinstance(state.metrics, PrometheusMetrics):
            return PlainTextResponse(state.metrics.render(), media_type="text/plain; version=0.0.4")
        return PlainTextResponse("", media_type="text/plain; version=0.0.4")

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

        requested_language = _requested_asr_language(payload.get("language"))
        job_config = _config_with_asr_language(state.config, requested_language)
        scenario_name = payload.get("scenario")
        scenario: Scenario | None = None
        if scenario_name:
            scenario = load_demo_scenario(str(scenario_name))
            audio = scenario_wav(scenario)
        else:
            encoded = payload.get("audio_base64")
            if not encoded:
                raise HTTPException(
                    status_code=400, detail="provide either 'scenario' or 'audio_base64'"
                )
            if state.engine.name == "fake":
                # The fake adapters read synthetic tones, so real speech would come
                # back empty. Say so rather than return a hollow transcript
                # (spec 18 rule 6).
                raise ModelNotReadyError(
                    "real audio needs the model engine; start the API with SASTT_ENGINE=real"
                )
            try:
                audio = base64.b64decode(str(encoded), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise HTTPException(status_code=400, detail="audio_base64 is not valid") from exc

        try:
            job, created = state.jobs.create_or_get(
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                input_hash=hashlib.sha256(audio).hexdigest(),
                config_version=job_config.config_version,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=exc.message) from exc

        if created:
            state.job_scenarios[job.job_id] = str(scenario_name or "upload")
            state.job_languages[job.job_id] = requested_language
            # The queue runner has always stored the input because the worker
            # reads it. The in-process runner stores it too, so the console can
            # play a segment back against the audio it was cut from.
            audio_key = _retain_job_audio(state, tenant_id, job.job_id, audio)
            if state.task_queue is None:
                _run_job(state, job, scenario, audio, tenant_id, job_config)
            else:
                try:
                    state.task_queue.enqueue(
                        QUEUE_SPEAKER_BATCH,
                        tenant_id=tenant_id,
                        job_id=job.job_id,
                        payload={
                            "audio_key": audio_key,
                            "scenario": scenario_name,
                            "language": requested_language,
                        },
                    )
                except SasttError:
                    state.objects.delete(tenant_id, audio_key)
                    state.job_audio_keys.pop(job.job_id, None)
                    state.jobs.update_state(tenant_id, job.job_id, JobState.CANCELLED)
                    raise
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
        if result is None and not job.is_terminal:
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}, not finished")
        if result is None:
            loader = getattr(state.jobs, "load_result", None)
            if loader is None:
                raise HTTPException(
                    status_code=409, detail=f"job is {job.state.value}, not finished"
                )
            segments = loader(tenant_id, job.job_id)
            if not segments:
                raise HTTPException(
                    status_code=409, detail=f"job is {job.state.value}, not finished"
                )
            return {
                "job_id": job.job_id,
                "engine": state.engine.name,
                "schema_version": "2.0",
                "state": job.state.value,
                "config_version": job.config_version,
                "warnings": job.warnings,
                "degraded_mode": job.degraded,
                "session_speakers": count_session_speakers(segments),
                "segments": segments,
                "text": render_transcript_from_public_segments(segments),
            }
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
            "session_speakers": count_session_speakers(
                [segment.to_public_dict() for segment in result.segments]
            ),
            "segments": [segment.to_public_dict() for segment in result.segments],
            "text": render_transcript(result.segments),
        }

    @app.get("/v1/jobs/{job_id}/audio")
    async def get_job_audio(
        job_id: str,
        request: Request,
        preview: bool = False,
        tenant_id: str = Depends(dev_tenant),
    ) -> Response:
        """Serve the job's input audio so a segment can be listened back.

        This is the audio the transcript was cut from, never a derived or
        separated source: a separated source is a model artefact and playing one
        back as if it were the recording would misrepresent what the pipeline
        did. Retention is bounded (:data:`MAX_RETAINED_JOB_AUDIO`) and a deleted
        job takes its audio with it, so a missing object is a normal 404.

        ``?preview=1`` serves a mono 16 kHz re-encode instead — same recording,
        same timeline, a fraction of the bytes, so a console reached over a
        tunnel can still play a segment. Without the flag the response is the
        input byte for byte.
        """
        state: AppState = request.app.state.sastt
        job = state.jobs.get(tenant_id, job_id)
        _, key = state.job_audio_keys.get(job.job_id, (tenant_id, job_audio_key(job.job_id)))
        try:
            payload = state.objects.get(tenant_id, key)
        except (SasttError, KeyError) as exc:
            raise HTTPException(
                status_code=404, detail="input audio is no longer retained for this job"
            ) from exc
        headers = {
            "Content-Disposition": f'inline; filename="{job.job_id}"',
            # Biometric data: never let a shared cache keep a copy (spec 14.4).
            "Cache-Control": "private, no-store",
        }
        if preview:
            encoded = state.job_audio_previews.get(job.job_id)
            if encoded is None:
                encoded = build_audio_preview(payload)
                if encoded is not None:
                    state.job_audio_previews[job.job_id] = encoded
            if encoded is not None:
                return Response(
                    content=encoded, media_type=AUDIO_PREVIEW_MEDIA_TYPE, headers=headers
                )
        return Response(content=payload, media_type=audio_media_type(payload), headers=headers)

    @app.delete("/v1/jobs/{job_id}", status_code=202)
    async def delete_job(
        job_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        job = state.jobs.get(tenant_id, job_id)
        state.results.pop(job.job_id, None)
        retained = state.job_audio_keys.pop(job.job_id, None)
        state.job_audio_previews.pop(job.job_id, None)
        if retained:
            state.objects.delete(*retained)
        if not job.is_terminal:
            job = state.jobs.update_state(tenant_id, job.job_id, JobState.CANCELLED)
        return {"job_id": job.job_id, "state": job.state.value}

    # -- Voice registry (spec 8.3, FR-014) ------------------------------------ #

    @app.get("/v1/voice-identities")
    async def list_voice_identities(
        request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        """List only the local tenant's voice identities for the management UI."""
        state: AppState = request.app.state.sastt
        identities = [
            _voice_identity_view(state, tenant_id, identity_id)
            for owner, identity_id in state.identity_metadata
            if owner == tenant_id
        ]
        return {
            "identities": sorted(identities, key=lambda item: str(item["display_name"]).lower())
        }

    @app.post("/v1/voice-identities", status_code=201)
    async def create_voice_identity(
        payload: dict[str, Any], request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        """Create local identity metadata; templates arrive through enrollment."""
        state: AppState = request.app.state.sastt
        external_id = payload.get("external_id")
        identity_id = str(external_id) if external_id else new_id("vid")
        if not identity_id.strip():
            raise HTTPException(status_code=400, detail="external_id must not be empty")
        key = (tenant_id, identity_id)
        if key in state.identity_metadata:
            raise HTTPException(status_code=409, detail="voice identity already exists")
        state.identity_metadata[key] = {
            "display_name": str(payload.get("display_name") or identity_id),
            "consent_ref": str(payload["consent_ref"]) if payload.get("consent_ref") else None,
        }
        return _voice_identity_view(state, tenant_id, identity_id)

    @app.get("/v1/voice-identities/{identity_id}")
    async def get_voice_identity(
        identity_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        return _voice_identity_view(request.app.state.sastt, tenant_id, identity_id)

    @app.post("/v1/voice-identities/{identity_id}/enrollments", status_code=201)
    async def enroll_voice_identity(
        identity_id: str,
        payload: dict[str, Any],
        request: Request,
        tenant_id: str = Depends(dev_tenant),
    ) -> dict[str, Any]:
        """Extract a clean embedding from enrollment audio and return quality details."""
        state: AppState = request.app.state.sastt
        if state.registry is None:  # pragma: no cover - state always wires one
            raise ModelNotReadyError("voice registry is not available")
        metadata = state.identity_metadata.get((tenant_id, identity_id))
        if metadata is None:
            raise TenantAccessDeniedError("voice identity not found for this tenant")
        consent_ref = payload.get("consent_ref") or metadata.get("consent_ref")
        if not consent_ref:
            raise HTTPException(status_code=400, detail="consent_ref is required for enrollment")
        embedding = _extract_enrollment_embedding(state, payload, tenant_id)
        report = state.registry.enroll(
            tenant_id,
            identity_id,
            [embedding],
            CallContext(
                stage="voice_enrollment", tenant_hash=tenant_hash(tenant_id), metrics=state.metrics
            ),
            display_name=metadata.get("display_name"),
            consent_ref=str(consent_ref),
        )
        metadata["consent_ref"] = str(consent_ref)
        return report.to_dict()

    @app.delete("/v1/voice-identities/{identity_id}", status_code=202)
    async def delete_voice_identity(
        identity_id: str, request: Request, tenant_id: str = Depends(dev_tenant)
    ) -> dict[str, Any]:
        state: AppState = request.app.state.sastt
        if state.registry is None:  # pragma: no cover - state always wires one
            raise ModelNotReadyError("voice registry is not available")
        if (tenant_id, identity_id) not in state.identity_metadata:
            raise TenantAccessDeniedError("voice identity not found for this tenant")
        deleted = state.registry.delete_identity(
            tenant_id,
            identity_id,
            CallContext(
                stage="voice_delete", tenant_hash=tenant_hash(tenant_id), metrics=state.metrics
            ),
        )
        del state.identity_metadata[(tenant_id, identity_id)]
        return {"identity_id": identity_id, "deleted": deleted}

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
            metrics=state.metrics,
            calibrator=state.calibrator,
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
        # The console is a single file with its script inline, and it has no
        # build step or content hash in its URL. Without an explicit directive a
        # browser applies heuristic freshness and serves a stale console after
        # the file changes — which reads as "the change did not deploy".
        # ``no-cache`` still revalidates against the ETag, so an unchanged file
        # costs a 304, not a re-download.
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    return app


def _build_local_registry(engine: Engine, config: SasttConfig) -> VoiceRegistry:
    """Local registry works with either fake or real embeddings without a database."""
    from sastt.adapters.fake import FakeVoiceRegistry

    model_version = "fake-campplus@1"
    if engine.name == "real":
        model_version = engine.adapters_for(None).embedder.model_version
    return FakeVoiceRegistry(
        embedding_model_version=model_version,
        accept_threshold=config.voice_id.accept_threshold,
        ambiguous_margin=config.voice_id.ambiguous_margin,
        minimum_clips=config.voice_id.minimum_enrollment_clips,
        minimum_total_speech_ms=int(config.voice_id.minimum_total_speech_seconds * 1000),
    )


def _extract_enrollment_embedding(state: AppState, payload: dict[str, Any], tenant_id: str) -> Any:
    """Decode enrollment audio once, VAD it, then extract a clean embedding."""
    ctx = CallContext(
        stage="voice_enrollment_extract", tenant_hash=tenant_hash(tenant_id), metrics=state.metrics
    )
    scenario_name = payload.get("scenario")
    if scenario_name:
        if state.engine.name != "fake":
            raise HTTPException(
                status_code=400, detail="scenario enrollment is only available on fake engine"
            )
        scenario = load_demo_scenario(str(scenario_name))
        speaker = str(payload.get("speaker") or "")
        if speaker not in scenario.speakers:
            raise HTTPException(
                status_code=400, detail="speaker must be one of the scenario speakers"
            )
        interval = TimeInterval(0, scenario.duration_ms)
        samples = scenario.render_speaker(speaker, interval)
        buffer = AudioBuffer(
            samples=np.ascontiguousarray(samples[None, :]),
            sample_rate=scenario.sample_rate,
            start_sample=0,
            channel_layout=("mono",),
            source_clock_hz=scenario.sample_rate,
        )
        adapters = state.engine.adapters_for(scenario)
        return adapters.embedder.embed(
            buffer, ctx, speech_intervals=adapters.vad.detect(buffer, ctx)
        )

    encoded = payload.get("audio_base64")
    if not encoded:
        raise HTTPException(status_code=400, detail="provide scenario/speaker or audio_base64")
    if state.engine.name == "fake":
        raise ModelNotReadyError("real enrollment audio needs SASTT_ENGINE=real")
    try:
        audio = base64.b64decode(str(encoded), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="audio_base64 is not valid") from exc
    adapters = state.engine.adapters_for(None)
    asset = adapters.decoder.decode(audio, ctx)
    speech = adapters.vad.detect(asset.mono_16k, ctx)
    return adapters.embedder.embed(asset.mono_16k, ctx, speech_intervals=speech)


def _voice_identity_view(state: AppState, tenant_id: str, identity_id: str) -> dict[str, Any]:
    metadata = state.identity_metadata.get((tenant_id, identity_id))
    if metadata is None:
        raise TenantAccessDeniedError("voice identity not found for this tenant")
    prototypes = []
    if state.registry is not None and state.registry.identity_exists(tenant_id, identity_id):
        prototypes_of = getattr(state.registry, "prototypes_of", None)
        if prototypes_of is not None:
            prototypes = prototypes_of(tenant_id, identity_id)
    return {
        "identity_id": identity_id,
        "display_name": metadata["display_name"],
        "status": "active",
        "prototype_count": len(prototypes),
        "embedding_model_version": state.registry.embedding_model_version
        if state.registry
        else None,
        "consent_ref_present": bool(metadata["consent_ref"]),
    }


def count_session_speakers(segments: list[dict[str, Any]]) -> int:
    """Distinct identified people in a transcript — the console's "Global speakers".

    Derived from the segments so it is available on both result paths: a job
    replayed from PostgreSQL has no in-memory ``OfflineResult`` to read the
    diarizer's own estimate from. The unattributed sink is excluded — it is one
    bucket holding every source that could not be identified, not a person.
    """
    return len(
        {
            str(segment.get("session_speaker_id"))
            for segment in segments
            if segment.get("session_speaker_id") and segment.get("identity_status") != "unknown"
        }
    )


def render_transcript_from_public_segments(segments: list[dict[str, Any]]) -> str:
    """Render durable JSON results without reconstructing domain objects."""
    lines = []
    for segment in sorted(
        segments, key=lambda item: (item["start_ms"], item.get("session_speaker_id") or "")
    ):
        lines.append(
            f"{segment['start_ms'] / 1000:0.3f}–{segment['end_ms'] / 1000:0.3f} — "
            f"{segment['speaker_label']}: {segment['text']}"
        )
    return "\n".join(lines)


def _session(state: AppState, session_id: str) -> StreamingSession:
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
    return session


_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$")


def _requested_asr_language(value: object) -> str | None:
    """Validate the optional per-job language hint; ``auto`` keeps detection on."""
    if value is None or value == "auto":
        return None
    if not isinstance(value, str) or not _LANGUAGE_CODE.fullmatch(value):
        raise HTTPException(
            status_code=400, detail="language must be 'auto' or an ISO code such as en/vi"
        )
    return value


def _config_with_asr_language(config: SasttConfig, language: str | None) -> SasttConfig:
    return config.model_copy(update={"asr": config.asr.model_copy(update={"language": language})})


def _job_view(state: AppState, job: JobRecord, created: bool) -> dict[str, Any]:
    result = state.results.get(job.job_id)
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "created": created,
        "engine": state.engine.name,
        "scenario": state.job_scenarios.get(job.job_id),
        "asr_language": state.job_languages.get(job.job_id) or "auto",
        "config_version": job.config_version,
        "warnings": result.warnings if result else [],
        "degraded_mode": bool(result and result.degraded),
        "error_code": job.error_code,
    }


def _run_job(
    state: AppState,
    job: JobRecord,
    scenario: Scenario | None,
    audio: bytes,
    tenant_id: str,
    config: SasttConfig,
) -> None:
    """Run one job in-process (the worker topology of spec 11.1 arrives in M3)."""
    adapters = replace(state.engine.adapters_for(scenario), registry=state.registry)
    pipeline = OfflinePipeline(config, adapters, tenant_id=tenant_id, calibrator=state.calibrator)
    ctx = CallContext(
        stage="offline_job",
        job_id=job.job_id,
        tenant_hash=tenant_hash(tenant_id),
        metrics=state.metrics,
    )
    job.transition(JobState.PREPROCESSING)
    payload = _strip_wav_header(audio) if state.engine.name == "fake" else audio
    try:
        result = pipeline.run(payload, ctx, job=job)
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
