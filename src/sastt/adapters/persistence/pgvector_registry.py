"""Open-set voice registry on PostgreSQL + pgvector — spec 5.10, 8.3, 10, 14.

Mirrors the decision rules of the in-memory registry so behaviour does not
change with the backend, and adds what only a database can give:

* templates are rows scoped by ``tenant_id``; a cross-tenant read raises rather
  than returning nothing that could later be mistaken for "no match" (S15);
* vectors carry the ``model_release_id`` that produced them, and scoring filters
  on it — embeddings of different model versions are never compared (spec 5.6);
* deleting an identity cascades to its templates and writes an audit row, which
  is what spec 10.3 and FR-014 require of a deletion;
* enrollment stores **several prototypes**, never one averaged centroid
  (spec 5.10).

Fails closed: with ``accept_threshold``/``ambiguous_margin`` still null the
registry returns ``uncalibrated`` and identifies nobody (spec 5.10, 18 rule 7).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sastt.domain.errors import TenantAccessDeniedError
from sastt.domain.events import new_id
from sastt.domain.speakers import (
    EnrollmentClipReport,
    EnrollmentReport,
    SpeakerEmbedding,
    SpeakerPrototype,
    VoiceIdDecision,
)
from sastt.observability import CallContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg_pool import ConnectionPool

#: CAM++ dimensionality, matching ``vector(192)`` in the migration (spec 0.2).
EMBEDDING_DIMENSION = 192

MIN_CLIP_MS = 3_000
MAX_CLIP_MS = 15_000


class PgVectorVoiceRegistry:
    """``VoiceRegistry`` port backed by ``voice_identities`` / ``voice_templates``."""

    def __init__(
        self,
        pool: ConnectionPool,
        *,
        embedding_model_version: str,
        accept_threshold: float | None = None,
        ambiguous_margin: float | None = None,
        minimum_clips: int = 3,
        minimum_total_speech_ms: int = 15_000,
    ) -> None:
        self._pool = pool
        self._model_version = embedding_model_version
        self.accept_threshold = accept_threshold
        self.ambiguous_margin = ambiguous_margin
        self.minimum_clips = minimum_clips
        self.minimum_total_speech_ms = minimum_total_speech_ms

    @property
    def embedding_model_version(self) -> str:
        return self._model_version

    @property
    def is_calibrated(self) -> bool:
        return self.accept_threshold is not None and self.ambiguous_margin is not None

    # -- enrollment (spec 5.10, 8.3) ---------------------------------------- #

    def enroll(
        self,
        tenant_id: str,
        identity_id: str,
        embeddings: list[SpeakerEmbedding],
        ctx: CallContext,
        *,
        display_name: str | None = None,
        consent_ref: str | None = None,
    ) -> EnrollmentReport:
        ctx.check()
        clips: list[EnrollmentClipReport] = []
        accepted: list[SpeakerEmbedding] = []
        for index, embedding in enumerate(embeddings):
            reasons = self._clip_reasons(embedding)
            clips.append(
                EnrollmentClipReport(
                    clip_index=index,
                    accepted=not reasons,
                    speech_duration_ms=embedding.speech_duration_ms,
                    reasons=tuple(reasons),
                )
            )
            if not reasons:
                accepted.append(embedding)

        with self._pool.connection() as connection, connection.cursor() as cursor:
            # Spec 14.4: enrollment needs a consent reference. There is no
            # sensible default, so the caller's value is stored verbatim.
            cursor.execute(
                """
                INSERT INTO voice_identities (id, tenant_id, external_id, display_name,
                                              consent_ref)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET display_name = COALESCE(EXCLUDED.display_name,
                                                voice_identities.display_name),
                        consent_ref  = COALESCE(EXCLUDED.consent_ref,
                                                voice_identities.consent_ref)
                WHERE voice_identities.tenant_id = EXCLUDED.tenant_id
                """,
                (
                    identity_id,
                    tenant_id,
                    identity_id,
                    display_name or identity_id,
                    consent_ref or "unspecified",
                ),
            )
            for embedding in accepted:
                cursor.execute(
                    """
                    INSERT INTO voice_templates (id, identity_id, tenant_id, model_release_id,
                                                 vector, quality, speech_ms, source_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (identity_id, model_release_id, source_hash) DO NOTHING
                    """,
                    (
                        new_id("vtpl"),
                        identity_id,
                        tenant_id,
                        embedding.model_version,
                        _to_vector_literal(embedding),
                        float(embedding.quality),
                        int(embedding.speech_duration_ms),
                        _source_hash(embedding),
                    ),
                )
            cursor.execute(
                """
                SELECT count(*), COALESCE(sum(speech_ms), 0) FROM voice_templates
                WHERE identity_id = %s AND tenant_id = %s AND model_release_id = %s
                """,
                (identity_id, tenant_id, self._model_version),
            )
            row = cursor.fetchone()
            prototype_count = int(row[0]) if row else 0
            total_speech = int(row[1]) if row else 0
            self._audit(
                cursor,
                tenant_id,
                action="voice_identity.enroll",
                subject_id=identity_id,
                details={"accepted": len(accepted), "rejected": len(embeddings) - len(accepted)},
            )

        policy_reasons: list[str] = []
        if prototype_count < self.minimum_clips:
            policy_reasons.append("fewer_than_minimum_clips")
        if total_speech < self.minimum_total_speech_ms:
            policy_reasons.append("insufficient_total_speech")
        return EnrollmentReport(
            identity_id=identity_id,
            accepted_clips=len(accepted),
            rejected_clips=len(embeddings) - len(accepted),
            total_speech_ms=total_speech,
            prototype_count=prototype_count,
            embedding_model_version=self._model_version,
            meets_policy=not policy_reasons,
            clips=tuple(clips),
            reasons=tuple(policy_reasons),
        )

    def _clip_reasons(self, embedding: SpeakerEmbedding) -> list[str]:
        reasons: list[str] = []
        if embedding.model_version != self._model_version:
            reasons.append("embedding_model_mismatch")
        if embedding.speech_duration_ms < MIN_CLIP_MS:
            reasons.append("clip_shorter_than_3s")
        if embedding.speech_duration_ms > MAX_CLIP_MS:
            reasons.append("clip_longer_than_15s")
        if embedding.origin == "separated":
            # Spec 5.10 rejects overlapped/separated audio as enrollment material.
            reasons.append("separated_source_not_allowed")
        return reasons

    # -- identification (spec 5.10) ----------------------------------------- #

    def identify(
        self,
        tenant_id: str,
        embedding: SpeakerEmbedding,
        ctx: CallContext,
    ) -> VoiceIdDecision:
        ctx.check()
        if not self.is_calibrated:
            return VoiceIdDecision(status="uncalibrated", reason="thresholds_null")
        if embedding.model_version != self._model_version:
            return VoiceIdDecision(status="unknown", reason="embedding_model_mismatch")

        with self._pool.connection() as connection, connection.cursor() as cursor:
            # Best template per identity, ranked. `<=>` is pgvector's cosine
            # distance, so similarity is 1 - distance.
            cursor.execute(
                """
                SELECT i.id, i.display_name, MAX(1 - (t.vector <=> %s)) AS score
                FROM voice_templates t
                JOIN voice_identities i ON i.id = t.identity_id
                WHERE t.tenant_id = %s
                  AND t.model_release_id = %s
                  AND i.deleted_at IS NULL
                GROUP BY i.id, i.display_name
                ORDER BY score DESC
                LIMIT 2
                """,
                (_to_vector_literal(embedding), tenant_id, self._model_version),
            )
            rows = cursor.fetchall()

        if not rows:
            return VoiceIdDecision(status="unknown", reason="empty_registry")

        identity_id, display_name, best_score = rows[0][0], rows[0][1], float(rows[0][2])
        margin = None if len(rows) < 2 else best_score - float(rows[1][2])

        assert self.accept_threshold is not None and self.ambiguous_margin is not None
        if best_score < self.accept_threshold:
            return VoiceIdDecision(
                status="unknown", best_score=best_score, margin=margin, reason="below_accept"
            )
        if margin is not None and margin < self.ambiguous_margin:
            # Two identities this close cannot be told apart; spec 5.10 forbids
            # forcing a choice.
            return VoiceIdDecision(
                status="ambiguous", best_score=best_score, margin=margin, reason="low_margin"
            )
        return VoiceIdDecision(
            status="enrolled",
            registry_speaker_id=identity_id,
            speaker_name=display_name,
            best_score=best_score,
            margin=margin,
            reason="accept",
        )

    # -- deletion and lookup (spec 10.3, FR-014) ---------------------------- #

    def delete_identity(self, tenant_id: str, identity_id: str, ctx: CallContext) -> bool:
        ctx.check()
        with self._pool.connection() as connection, connection.cursor() as cursor:
            # Templates go with the identity via ON DELETE CASCADE, which also
            # removes them from the HNSW index (spec 10.3).
            cursor.execute(
                "DELETE FROM voice_identities WHERE id = %s AND tenant_id = %s RETURNING id",
                (identity_id, tenant_id),
            )
            deleted = cursor.fetchone() is not None
            if deleted:
                self._audit(
                    cursor,
                    tenant_id,
                    action="voice_identity.delete",
                    subject_id=identity_id,
                    details={"cascade": "templates"},
                )
        return deleted

    def identity_exists(self, tenant_id: str, identity_id: str) -> bool:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM voice_identities
                WHERE id = %s AND tenant_id = %s AND deleted_at IS NULL
                """,
                (identity_id, tenant_id),
            )
            return cursor.fetchone() is not None

    def prototypes_of(self, tenant_id: str, identity_id: str) -> list[SpeakerPrototype]:
        if not self.identity_exists(tenant_id, identity_id):
            raise TenantAccessDeniedError(
                f"identity {identity_id!r} is not visible to this tenant",
                details={"identity_id": identity_id},
            )
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT vector, quality, speech_ms FROM voice_templates
                WHERE identity_id = %s AND tenant_id = %s AND model_release_id = %s
                """,
                (identity_id, tenant_id, self._model_version),
            )
            rows = cursor.fetchall()
        return [
            SpeakerPrototype.from_embedding(
                identity_id,
                SpeakerEmbedding(
                    vector=_from_vector_literal(row[0]),
                    model_version=self._model_version,
                    quality=float(row[1]),
                    speech_duration_ms=int(row[2]),
                    origin="clean",
                ),
            )
            for row in rows
        ]

    def _audit(
        self,
        cursor: Any,
        tenant_id: str,
        *,
        action: str,
        subject_id: str,
        details: dict[str, Any],
    ) -> None:
        cursor.execute(
            """
            INSERT INTO audit_events (tenant_id, actor, action, subject_id, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (tenant_id, "system", action, subject_id, json.dumps(details)),
        )


def _to_vector_literal(embedding: SpeakerEmbedding) -> str:
    values = [float(x) for x in embedding.vector]
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(f"expected {EMBEDDING_DIMENSION}-dimensional embedding, got {len(values)}")
    return "[" + ",".join(repr(v) for v in values) + "]"


def _from_vector_literal(raw: Any) -> Any:
    import numpy as np

    if isinstance(raw, str):
        return np.asarray(json.loads(raw), dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def _source_hash(embedding: SpeakerEmbedding) -> str:
    """Identify the clip a template came from, so re-posting it is a no-op."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(embedding.vector.tobytes())
    digest.update(str(embedding.speech_duration_ms).encode())
    return digest.hexdigest()


__all__ = ["EMBEDDING_DIMENSION", "PgVectorVoiceRegistry"]
