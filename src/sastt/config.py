"""Configuration and startup gates — spec 12, 11.2, 20.

The loader refuses to start a production process when a research/beta flag is
enabled together with a checkpoint whose licence is not allowlisted (spec 12),
and when a threshold that gates identity decisions is still uncalibrated while
fail-closed behaviour has been switched off (spec 5.10).
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sastt.domain.errors import ConfigurationError, ErrorCode

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_MANIFEST_DIR = Path("model-manifests")


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION


class ProductionAction(str, Enum):
    """Per-component production gate — spec 20 table."""

    ALLOW = "allow"
    PRODUCTION_CANDIDATE = "production_candidate"
    BETA_ONLY = "beta_only"
    PHASE_2 = "phase_2"
    EXPERIMENTAL = "experimental"
    DENY = "deny"


#: Actions that may serve production traffic without an explicit beta flag.
PRODUCTION_ALLOWLIST: frozenset[ProductionAction] = frozenset(
    {ProductionAction.ALLOW, ProductionAction.PRODUCTION_CANDIDATE}
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProductConfig(_Base):
    max_session_speakers: int = Field(5, ge=1, le=5)
    max_supported_concurrent_speakers: int = Field(2, ge=1, le=5)
    three_source_beta: bool = False
    mono_four_five_source_research: bool = False
    multichannel_gss: bool = False
    target_speaker_extraction: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> ProductConfig:
        if self.max_supported_concurrent_speakers > self.max_session_speakers:
            raise ValueError("max_supported_concurrent_speakers cannot exceed max_session_speakers")
        if self.max_supported_concurrent_speakers >= 3 and not self.three_source_beta:
            raise ValueError(
                "three or more concurrent speakers requires three_source_beta (spec 0.1)"
            )
        if self.max_supported_concurrent_speakers >= 4 and not (
            self.multichannel_gss or self.mono_four_five_source_research
        ):
            raise ValueError(
                "four or five concurrent speakers requires multichannel GSS or the research flag "
                "(spec 0.1.4, 1.2)"
            )
        return self


class AudioConfig(_Base):
    canonical_sample_rate: Literal[16000] = 16000
    preserve_input_channels: bool = True
    max_channels: int = Field(8, ge=1, le=8)
    max_file_hours: float = Field(4, gt=0)
    overlap_context_seconds: float = Field(0.50, ge=0)

    @field_validator("preserve_input_channels")
    @classmethod
    def _must_preserve(cls, value: bool) -> bool:
        if not value:
            raise ValueError("input channels must be preserved (spec 1.1, 5.1.4)")
        return value


class StreamingConfig(_Base):
    frame_ms: int = Field(40, ge=20, le=100)
    ring_buffer_seconds: float = Field(30, gt=0)
    diarization_window_seconds: float = Field(10, gt=0)
    diarization_hop_seconds: float = Field(2, gt=0)
    finalize_after_silence_seconds: float = Field(1.2, gt=0)
    provisional_updates: bool = True
    allow_revision: bool = True

    @model_validator(mode="after")
    def _check_window(self) -> StreamingConfig:
        if self.diarization_hop_seconds > self.diarization_window_seconds:
            raise ValueError("diarization hop cannot exceed the diarization window")
        if self.ring_buffer_seconds < self.diarization_window_seconds:
            raise ValueError("ring buffer must hold at least one diarization window")
        if self.provisional_updates and not self.allow_revision:
            raise ValueError(
                "provisional events without revisions would strand wrong labels (spec FR-011)"
            )
        return self


class DiarizationConfig(_Base):
    primary: str = "pyannote-community-1"
    model_path: str | None = "/models/pyannote-community-1"
    min_speakers: int = Field(1, ge=1, le=5)
    max_speakers: int = Field(5, ge=1, le=5)
    regular_output_for_overlap: bool = True
    exclusive_output_for_non_overlap_alignment: bool = True

    @model_validator(mode="after")
    def _check_speakers(self) -> DiarizationConfig:
        if self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers cannot exceed max_speakers")
        if not self.regular_output_for_overlap:
            raise ValueError(
                "regular tracks are the source of truth for overlap and cannot be disabled "
                "(spec 5.2)"
            )
        return self


class OverlapDetectionConfig(_Base):
    model_path: str | None = "/models/pyannote-segmentation-3.0"
    onset: float = Field(0.60, ge=0.0, le=1.0)
    offset: float = Field(0.50, ge=0.0, le=1.0)
    min_duration_seconds: float = Field(0.30, gt=0)
    merge_gap_seconds: float = Field(0.20, ge=0)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> OverlapDetectionConfig:
        if self.offset > self.onset:
            raise ValueError("offset must not be higher than onset")
        return self


class SourceCountConfig(_Base):
    production_default: Literal[
        "fixed_two", "ts_vad", "multichannel_activity", "multidecoder_research", "unknown"
    ] = "fixed_two"
    minimum_confidence: float = Field(0.75, ge=0.0, le=1.0)
    multidecoder_research_model_path: str | None = None


class SeparationConfig(_Base):
    two_source_backend: str = "mossformer2_ss_16k"
    two_source_model_path: str | None = "/models/mossformer2-ss-16k"
    three_source_backend: str = "sepformer_libri3mix"
    three_source_model_path: str | None = None
    max_crop_seconds: float = Field(10, gt=0)
    stitch_overlap_seconds: float = Field(1, ge=0)

    @model_validator(mode="after")
    def _check_stitch(self) -> SeparationConfig:
        if self.stitch_overlap_seconds >= self.max_crop_seconds:
            raise ValueError("stitch overlap must be shorter than the maximum crop")
        return self


class SpeakerEmbeddingConfig(_Base):
    backend: str = "3d_speaker_campplus"
    model_path: str | None = "/models/campplus"
    minimum_clean_speech_seconds: float = Field(1.5, gt=0)
    target_clean_speech_seconds: float = Field(3.0, gt=0)
    update_from_separated_sources: bool = False

    @model_validator(mode="after")
    def _check_targets(self) -> SpeakerEmbeddingConfig:
        if self.target_clean_speech_seconds < self.minimum_clean_speech_seconds:
            raise ValueError("target clean speech must be >= the minimum")
        return self


class SourceLinkingConfig(_Base):
    accept_threshold: float | None = None
    ambiguous_margin: float | None = None
    continuity_bonus: float = Field(0.02, ge=0.0, le=0.05)
    algorithm: Literal["hungarian"] = "hungarian"

    @property
    def is_calibrated(self) -> bool:
        return self.accept_threshold is not None and self.ambiguous_margin is not None


class VoiceIdConfig(_Base):
    enabled: bool = True
    accept_threshold: float | None = None
    ambiguous_margin: float | None = None
    minimum_enrollment_clips: int = Field(3, ge=3)
    minimum_total_speech_seconds: float = Field(15, ge=15)
    fail_closed_when_uncalibrated: bool = True

    @property
    def is_calibrated(self) -> bool:
        return self.accept_threshold is not None and self.ambiguous_margin is not None

    @model_validator(mode="after")
    def _check_fail_closed(self) -> VoiceIdConfig:
        if self.enabled and not self.is_calibrated and not self.fail_closed_when_uncalibrated:
            raise ValueError("Voice ID without calibrated thresholds must fail closed (spec 5.10)")
        return self


class AsrConfig(_Base):
    realtime_model_path: str | None = "/models/faster-whisper-large-v3-turbo"
    final_model_path: str | None = "/models/faster-whisper-large-v3"
    language: str = "vi"
    word_timestamps: bool = True
    compute_type: str = "int8_float16"
    final_rescore: bool = False

    @model_validator(mode="after")
    def _check_rescore(self) -> AsrConfig:
        if self.final_rescore and not self.final_model_path:
            raise ValueError("final_rescore requires final_model_path")
        return self


class ConfidenceConfig(_Base):
    calibration_path: str | None = None
    return_null_when_uncalibrated: bool = True

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_path is not None

    @model_validator(mode="after")
    def _check_null_policy(self) -> ConfidenceConfig:
        if not self.is_calibrated and not self.return_null_when_uncalibrated:
            raise ValueError("uncalibrated confidences must be returned as null (spec 0.3)")
        return self


class SasttConfig(_Base):
    """Full configuration tree of spec 12."""

    product: ProductConfig = ProductConfig()
    audio: AudioConfig = AudioConfig()
    streaming: StreamingConfig = StreamingConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    overlap_detection: OverlapDetectionConfig = OverlapDetectionConfig()
    source_count: SourceCountConfig = SourceCountConfig()
    separation: SeparationConfig = SeparationConfig()
    speaker_embedding: SpeakerEmbeddingConfig = SpeakerEmbeddingConfig()
    source_linking: SourceLinkingConfig = SourceLinkingConfig()
    voice_id: VoiceIdConfig = VoiceIdConfig()
    asr: AsrConfig = AsrConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()

    @model_validator(mode="after")
    def _cross_section(self) -> SasttConfig:
        if self.diarization.max_speakers > self.product.max_session_speakers:
            raise ValueError("diarization.max_speakers cannot exceed product.max_session_speakers")
        if self.product.three_source_beta and not self.separation.three_source_model_path:
            raise ValueError(
                "three_source_beta requires separation.three_source_model_path (spec 5.3 router)"
            )
        if (
            self.product.mono_four_five_source_research
            and not self.source_count.multidecoder_research_model_path
        ):
            raise ValueError(
                "the research branch requires source_count.multidecoder_research_model_path"
            )
        return self

    @property
    def config_version(self) -> str:
        """Stable digest of the whole tree — recorded on every output (FR-013)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "cfg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ModelManifest(_Base):
    """Model release manifest — spec 11.2 and the licence gate of spec 20."""

    component: str
    backend: str
    repository: str
    revision: str | None = None
    sha256: str | None = None
    code_license: str
    weight_license: str
    training_data_caveat: str | None = None
    production_action: ProductionAction
    enabled: bool = True
    requires_flag: str | None = None

    @property
    def release_id(self) -> str:
        return f"{self.backend}@{self.revision or self.sha256 or 'unpinned'}"


def load_manifests(directory: Path = DEFAULT_MANIFEST_DIR) -> dict[str, ModelManifest]:
    """Load ``model-manifests/*.yaml`` keyed by backend id."""
    manifests: dict[str, ModelManifest] = {}
    if not directory.exists():
        return manifests
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            manifest = ModelManifest.model_validate(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"invalid model manifest {path}: {exc}") from exc
        if manifest.backend in manifests:
            raise ConfigurationError(f"duplicate manifest for backend {manifest.backend!r}")
        manifests[manifest.backend] = manifest
    return manifests


def _flag_enabled(config: SasttConfig, flag: str | None) -> bool:
    if flag is None:
        return False
    return bool(getattr(config.product, flag, False))


def _active_backends(config: SasttConfig) -> list[tuple[str, str]]:
    """(component, backend) pairs that this configuration may actually execute."""
    active: list[tuple[str, str]] = [
        ("diarization", config.diarization.primary),
        ("embedding", config.speaker_embedding.backend),
        ("asr", "faster_whisper"),
        ("separation_two_source", config.separation.two_source_backend),
    ]
    if config.overlap_detection.model_path:
        active.append(("overlap_detection", "pyannote_segmentation_3.0"))
    if config.product.three_source_beta:
        active.append(("separation_three_source", config.separation.three_source_backend))
    if config.product.mono_four_five_source_research:
        active.append(("separation_research", "multidecoder_dprnn"))
    if config.product.multichannel_gss:
        active.append(("separation_multichannel", "gpu_gss"))
    if config.product.target_speaker_extraction:
        active.append(("target_extraction", "wesep"))
    if config.source_count.production_default == "multidecoder_research":
        active.append(("source_count", "multidecoder_dprnn"))
    return active


def validate_for_environment(
    config: SasttConfig,
    environment: Environment,
    manifests: dict[str, ModelManifest] | None = None,
) -> None:
    """Startup gate — spec 12, 20.

    Raises :class:`ConfigurationError` instead of letting the process serve
    traffic with a denied checkpoint or an uncalibrated fail-open threshold.
    """
    manifests = manifests if manifests is not None else {}

    if environment.is_production:
        if config.product.mono_four_five_source_research:
            raise ConfigurationError(
                "mono_four_five_source_research is research-only and MUST be disabled in "
                "production (spec 0.2, 20)",
                details={"error_code": ErrorCode.MODEL_LICENSE_DISABLED.value},
            )
        if config.source_count.production_default == "multidecoder_research":
            raise ConfigurationError(
                "the Multi-Decoder DPRNN counter is denied in production (spec 20)",
                details={"error_code": ErrorCode.MODEL_LICENSE_DISABLED.value},
            )
        if (
            config.voice_id.enabled
            and not config.voice_id.is_calibrated
            and not config.voice_id.fail_closed_when_uncalibrated
        ):
            raise ConfigurationError(
                "Voice ID must fail closed while uncalibrated (spec 5.10)",
                details={"error_code": ErrorCode.VOICE_ID_UNCALIBRATED.value},
            )

    for component, backend in _active_backends(config):
        manifest = manifests.get(backend)
        if manifest is None:
            if environment.is_production:
                raise ConfigurationError(
                    f"no model manifest for backend {backend!r} ({component}); production "
                    "requires a pinned, licence-reviewed manifest (spec 11.2)",
                    details={"error_code": ErrorCode.MODEL_NOT_READY.value},
                )
            continue
        if not manifest.enabled:
            raise ConfigurationError(
                f"backend {backend!r} is disabled by its manifest",
                details={"error_code": ErrorCode.MODEL_LICENSE_DISABLED.value},
            )
        if not environment.is_production:
            continue
        if manifest.production_action is ProductionAction.DENY:
            raise ConfigurationError(
                f"backend {backend!r} is denied in production by licence review (spec 20): "
                f"{manifest.training_data_caveat or manifest.weight_license}",
                details={"error_code": ErrorCode.MODEL_LICENSE_DISABLED.value},
            )
        if manifest.production_action not in PRODUCTION_ALLOWLIST and not _flag_enabled(
            config, manifest.requires_flag
        ):
            raise ConfigurationError(
                f"backend {backend!r} is {manifest.production_action.value} and needs the "
                f"{manifest.requires_flag!r} flag to run in production (spec 12, 20)",
                details={"error_code": ErrorCode.MODEL_LICENSE_DISABLED.value},
            )
        if not manifest.revision and not manifest.sha256:
            raise ConfigurationError(
                f"backend {backend!r} is not pinned by revision or SHA-256 (spec 0.3, 11.2)",
                details={"error_code": ErrorCode.MODEL_NOT_READY.value},
            )


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    environment: Environment | str | None = None,
    manifest_dir: Path | str | None = DEFAULT_MANIFEST_DIR,
    overrides: dict[str, Any] | None = None,
) -> SasttConfig:
    """Load, validate and gate a configuration file.

    ``environment`` defaults to ``$SASTT_ENV`` and then to ``development``.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    try:
        config = SasttConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"invalid configuration {config_path}: {exc}") from exc

    env = _resolve_environment(environment)
    manifests = load_manifests(Path(manifest_dir)) if manifest_dir else {}
    validate_for_environment(config, env, manifests)
    return config


def _resolve_environment(environment: Environment | str | None) -> Environment:
    if isinstance(environment, Environment):
        return environment
    raw = environment or os.environ.get("SASTT_ENV") or Environment.DEVELOPMENT.value
    try:
        return Environment(raw.lower())
    except ValueError as exc:
        raise ConfigurationError(f"unknown environment {raw!r}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MANIFEST_DIR",
    "PRODUCTION_ALLOWLIST",
    "AsrConfig",
    "AudioConfig",
    "ConfidenceConfig",
    "DiarizationConfig",
    "Environment",
    "ModelManifest",
    "OverlapDetectionConfig",
    "ProductConfig",
    "ProductionAction",
    "SasttConfig",
    "SeparationConfig",
    "SourceCountConfig",
    "SourceLinkingConfig",
    "SpeakerEmbeddingConfig",
    "StreamingConfig",
    "VoiceIdConfig",
    "load_config",
    "load_manifests",
    "validate_for_environment",
]
