"""Configuration and licence gates — spec 12, 11.2, 20."""

from __future__ import annotations

from pathlib import Path

import pytest

from sastt.config import (
    Environment,
    ModelFile,
    ModelManifest,
    ProductionAction,
    SasttConfig,
    aggregate_sha256,
    load_config,
    load_manifests,
    validate_for_environment,
)
from sastt.domain.errors import ConfigurationError

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
MANIFEST_DIR = REPO_ROOT / "model-manifests"


class TestDefaultConfig:
    def test_matches_the_spec_12_defaults(self, base_config: SasttConfig) -> None:
        assert base_config.product.max_session_speakers == 5
        assert base_config.product.max_supported_concurrent_speakers == 2
        assert base_config.product.three_source_beta is False
        assert base_config.product.mono_four_five_source_research is False
        assert base_config.audio.canonical_sample_rate == 16000
        assert base_config.streaming.frame_ms == 40
        assert base_config.separation.two_source_backend == "mossformer2_ss_16k"
        assert base_config.asr.language is None

    def test_thresholds_are_null_until_calibration(self, base_config: SasttConfig) -> None:
        assert base_config.source_linking.accept_threshold is None
        assert base_config.source_linking.ambiguous_margin is None
        assert base_config.voice_id.accept_threshold is None
        assert base_config.voice_id.fail_closed_when_uncalibrated is True
        assert base_config.confidence.calibration_path is None
        assert base_config.confidence.return_null_when_uncalibrated is True
        assert base_config.source_linking.is_calibrated is False

    def test_config_version_is_stable_and_changes_with_content(
        self, base_config: SasttConfig, calibrated_config: SasttConfig
    ) -> None:
        assert base_config.config_version == base_config.config_version
        assert base_config.config_version != calibrated_config.config_version

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"product": {"unknown_flag": True}},
            )


class TestValidationRules:
    def test_three_concurrent_speakers_require_the_beta_flag(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"product": {"max_supported_concurrent_speakers": 3}},
            )

    def test_beta_flag_requires_a_model_path(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"product": {"three_source_beta": True}},
            )

    def test_input_channels_must_be_preserved(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"audio": {"preserve_input_channels": False}},
            )

    def test_voice_id_cannot_fail_open_while_uncalibrated(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"voice_id": {"fail_closed_when_uncalibrated": False}},
            )

    def test_frame_size_stays_within_20_to_100_ms(self) -> None:
        with pytest.raises(ConfigurationError):
            load_config(
                CONFIG_PATH,
                environment="development",
                manifest_dir=None,
                overrides={"streaming": {"frame_ms": 250}},
            )


class TestManifests:
    def test_shipped_manifests_load(self) -> None:
        manifests = load_manifests(MANIFEST_DIR)
        assert "mossformer2_ss_16k" in manifests
        assert manifests["multidecoder_dprnn"].production_action is ProductionAction.DENY
        assert manifests["sepformer_libri3mix"].requires_flag == "three_source_beta"

    def test_production_refuses_unpinned_weights(self, base_config: SasttConfig) -> None:
        """A backend without a revision or digest never reaches production (spec 0.3, 11.2)."""
        manifests = {
            manifest.backend: manifest.model_copy(
                update={"revision": None if manifest.backend == "3d_speaker_campplus" else "rev"}
            )
            for manifest in load_manifests(MANIFEST_DIR).values()
        }
        manifests["3d_speaker_campplus"] = manifests["3d_speaker_campplus"].model_copy(
            update={"sha256": None, "files": ()}
        )
        with pytest.raises(ConfigurationError, match="not pinned"):
            validate_for_environment(base_config, Environment.PRODUCTION, manifests)

    def test_production_refuses_the_research_flag(self, base_config: SasttConfig) -> None:
        config = base_config.model_copy(
            update={
                "product": base_config.product.model_copy(
                    update={"mono_four_five_source_research": True}
                )
            }
        )
        with pytest.raises(ConfigurationError, match="research-only"):
            validate_for_environment(config, Environment.PRODUCTION, {})

    def test_production_refuses_a_denied_checkpoint(self, base_config: SasttConfig) -> None:
        manifests = {
            manifest.backend: manifest.model_copy(update={"revision": "pinned-rev"})
            for manifest in load_manifests(MANIFEST_DIR).values()
        }
        manifests["mossformer2_ss_16k"] = ModelManifest(
            component="separation_two_source",
            backend="mossformer2_ss_16k",
            repository="https://example.invalid",
            revision="abc123",
            code_license="Apache-2.0",
            weight_license="research-only",
            production_action=ProductionAction.DENY,
        )
        with pytest.raises(ConfigurationError, match="denied in production"):
            validate_for_environment(base_config, Environment.PRODUCTION, manifests)

    def test_production_requires_a_manifest_for_every_active_backend(
        self, base_config: SasttConfig
    ) -> None:
        with pytest.raises(ConfigurationError, match="no model manifest"):
            validate_for_environment(base_config, Environment.PRODUCTION, {})

    def test_development_tolerates_missing_manifests(self, base_config: SasttConfig) -> None:
        validate_for_environment(base_config, Environment.DEVELOPMENT, {})

    def test_pinned_manifests_pass_production(self, base_config: SasttConfig) -> None:
        manifests = {
            manifest.backend: manifest.model_copy(update={"revision": "pinned-rev"})
            for manifest in load_manifests(MANIFEST_DIR).values()
        }
        config = base_config.model_copy(
            update={
                "voice_id": base_config.voice_id.model_copy(
                    update={"accept_threshold": 0.7, "ambiguous_margin": 0.1}
                )
            }
        )
        validate_for_environment(config, Environment.PRODUCTION, manifests)


class TestWeightPinning:
    """Per-file digests recorded by ``deploy/prestage_models.py`` (spec 11.2)."""

    def _files(self) -> list[ModelFile]:
        return [
            ModelFile(path="model.bin", sha256="a" * 64, size_bytes=10),
            ModelFile(path="config.json", sha256="b" * 64, size_bytes=2),
        ]

    def test_aggregate_digest_is_order_independent(self) -> None:
        files = self._files()
        assert aggregate_sha256(files) == aggregate_sha256(list(reversed(files)))

    def test_aggregate_digest_changes_with_content(self) -> None:
        files = self._files()
        tampered = [files[0].model_copy(update={"sha256": "c" * 64}), files[1]]
        assert aggregate_sha256(files) != aggregate_sha256(tampered)

    def test_verify_digest_detects_tampering(self) -> None:
        files = self._files()
        manifest = ModelManifest(
            component="asr",
            backend="faster_whisper",
            repository="https://example.invalid",
            revision="rev",
            sha256=aggregate_sha256(files),
            files=tuple(files),
            code_license="MIT",
            weight_license="MIT",
            production_action=ProductionAction.PRODUCTION_CANDIDATE,
        )
        assert manifest.is_pinned is True
        assert manifest.verify_digest() is True
        tampered = manifest.model_copy(
            update={"files": (files[0].model_copy(update={"sha256": "c" * 64}), files[1])}
        )
        assert tampered.verify_digest() is False

    def test_unpinned_manifest_reports_itself(self) -> None:
        manifest = ModelManifest(
            component="asr",
            backend="faster_whisper",
            repository="https://example.invalid",
            code_license="MIT",
            weight_license="MIT",
            production_action=ProductionAction.PRODUCTION_CANDIDATE,
        )
        assert manifest.is_pinned is False
        assert manifest.verify_digest() is False
        assert manifest.release_id.endswith("unpinned")

    def test_gated_backends_record_who_accepted_the_terms(self) -> None:
        """Spec 20: gated pyannote models need terms acceptance and attribution recorded."""
        manifests = load_manifests(MANIFEST_DIR)
        for backend in ("pyannote-community-1", "pyannote_segmentation_3.0"):
            manifest = manifests[backend]
            assert manifest.terms_accepted_by
            assert manifest.attribution

    def test_staged_backends_carry_file_digests(self) -> None:
        """Whatever has been pre-staged must be pinned by revision *and* digest."""
        for manifest in load_manifests(MANIFEST_DIR).values():
            if manifest.local_path is None:
                continue
            assert manifest.revision
            assert manifest.files
            assert manifest.verify_digest()
