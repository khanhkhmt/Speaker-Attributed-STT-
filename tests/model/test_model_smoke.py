"""Model smoke tests — spec 16.1.4.

These require pinned local weights under ``/models`` and a GPU. They are marked
``model`` and are excluded from the default run, so ordinary CI never downloads
weights and never needs a Hub token (spec 16.3).

Spec 18 rule 6 is explicit: when weights, a token or a GPU are missing, the test
must skip with a reason. It must never fall back to the fake adapters and report
a pass — that would be an oracle wearing a model test's badge. The real adapters
themselves arrive with Milestone 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sastt.config import load_config

pytestmark = pytest.mark.model

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _weights_present(path: str | None) -> bool:
    return bool(path) and Path(str(path)).exists()


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH, environment="development", manifest_dir=None)


def test_diarization_weights_are_staged(config) -> None:
    if not _weights_present(config.diarization.model_path):
        pytest.skip(
            f"pinned diarization weights not staged at {config.diarization.model_path} "
            "(spec 11.2 pre-stages them at build time)"
        )
    pytest.skip("real pyannote adapter lands in Milestone 1 (spec 18)")


def test_asr_weights_are_staged(config) -> None:
    if not _weights_present(config.asr.realtime_model_path):
        pytest.skip(
            f"pinned ASR weights not staged at {config.asr.realtime_model_path} "
            "(spec 11.2 pre-stages them at build time)"
        )
    pytest.skip("real faster-whisper adapter lands in Milestone 1 (spec 18)")


def test_separation_weights_are_staged(config) -> None:
    if not _weights_present(config.separation.two_source_model_path):
        pytest.skip(
            f"pinned MossFormer2 weights not staged at {config.separation.two_source_model_path}"
        )
    pytest.skip("real ClearerVoice adapter lands in Milestone 1 (spec 18)")


def test_embedding_weights_are_staged(config) -> None:
    if not _weights_present(config.speaker_embedding.model_path):
        pytest.skip(f"pinned CAM++ weights not staged at {config.speaker_embedding.model_path}")
    pytest.skip("real 3D-Speaker adapter lands in Milestone 1 (spec 18)")
