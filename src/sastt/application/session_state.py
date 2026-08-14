"""Session speaker state — spec 5.6, 5.7, 5.9, 6.

Holds every speaker of one session: internal UUID, identity state machine,
quality-weighted prototype, cannot-link constraints, merges and display labels.
Internal IDs are never reused; display labels may be revised (spec 5.7, 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sastt.config import SasttConfig
from sastt.domain.audio import TimeInterval, seconds_to_ms
from sastt.domain.errors import InvalidStateTransitionError
from sastt.domain.events import new_ulid
from sastt.domain.speakers import (
    IdentityState,
    SessionSpeaker,
    SpeakerEmbedding,
    SpeakerIdentityStateMachine,
    SpeakerPrototype,
    VoiceIdDecision,
)

TEMPORARY_LABEL = "Temporary Speaker {index}"
SPEAKER_LABEL = "Speaker {index}"
UNATTRIBUTED_LABEL = "Unknown"

REASON_TOO_SHORT = "speech_shorter_than_minimum"
REASON_NOT_ANONYMOUS = "identity_not_resolved"
REASON_SEPARATED_DISABLED = "separated_updates_disabled"
REASON_LOW_SOURCE_QUALITY = "source_quality_gate"
REASON_LOW_LINK_MARGIN = "link_margin_gate"
REASON_MODEL_MISMATCH = "embedding_model_mismatch"
REASON_UPDATED = "updated"
REASON_CREATED = "created"


@dataclass(frozen=True)
class PrototypeUpdate:
    """Outcome of a prototype update attempt (spec 5.6)."""

    accepted: bool
    reason: str
    speaker_id: str | None = None
    version: int | None = None


@dataclass
class LabelChange:
    """A display-label change that must be published as a revision (spec 5.7, 6)."""

    session_speaker_id: str
    previous_label: str
    new_label: str
    reason: str


@dataclass
class SessionSpeakerState:
    """All speakers of one session and the rules that move them between states."""

    session_id: str
    config: SasttConfig
    embedding_model_version: str
    speakers: dict[str, SessionSpeaker] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    pending_label_changes: list[LabelChange] = field(default_factory=list)
    _temporary_counter: int = 0
    _label_counter: int = 0
    _unattributed_id: str | None = None

    # -- creation ------------------------------------------------------------ #

    def _new_id(self) -> str:
        return f"sess_spk_{new_ulid()}"

    def create_session_speaker(
        self,
        *,
        cluster_id: str | None = None,
        label: str | None = None,
    ) -> SessionSpeaker:
        """A clean cluster enters as ``SESSION_ANONYMOUS`` (spec 6)."""
        if self.active_speaker_count >= self.config.product.max_session_speakers:
            raise InvalidStateTransitionError(
                f"session already holds {self.config.product.max_session_speakers} speakers "
                "(spec 0.1.1)"
            )
        self._label_counter += 1
        speaker = SessionSpeaker(
            session_speaker_id=self._new_id(),
            machine=SpeakerIdentityStateMachine(IdentityState.SESSION_ANONYMOUS),
            display_label=label or SPEAKER_LABEL.format(index=self._label_counter),
            cluster_id=cluster_id,
        )
        self._register(speaker)
        return speaker

    def create_temporary_speaker(self) -> SessionSpeaker:
        """Overlap before any clean centroid exists — spec 5.9 step 1.

        A temporary identity counts against ``max_session_speakers`` like any
        other: spec 0.1.1 bounds the whole session at five speakers, and a
        provisional identity is still one of them.
        """
        if not self.can_admit_speaker:
            raise InvalidStateTransitionError(
                f"session already holds {self.config.product.max_session_speakers} speakers "
                "(spec 0.1.1)"
            )
        self._temporary_counter += 1
        speaker = SessionSpeaker(
            session_speaker_id=self._new_id(),
            machine=SpeakerIdentityStateMachine(IdentityState.PROVISIONAL),
            display_label=TEMPORARY_LABEL.format(index=self._temporary_counter),
        )
        self._register(speaker)
        return speaker

    def unattributed_speaker(self) -> SessionSpeaker:
        """The single sink for words no speaker turn claims — FR-012.

        This is not a sixth person: it carries no evidence and stands for the
        *absence* of an identity, so it is exempt from the
        ``max_session_speakers`` bound of spec 0.1.1. It is created at most once
        per session and reused, so unattributable audio can never inflate the
        speaker roster the way one-identity-per-fragment would.

        Fusion keeps the words rather than dropping them (spec 0.1.7).
        """
        if self._unattributed_id is not None:
            return self.speakers[self._unattributed_id]
        speaker = SessionSpeaker(
            session_speaker_id=self._new_id(),
            machine=SpeakerIdentityStateMachine(IdentityState.PROVISIONAL),
            display_label=UNATTRIBUTED_LABEL,
        )
        # Spec 6 only allows PROVISIONAL/SESSION_ANONYMOUS as initial states, so
        # the sink enters provisional and immediately takes the documented
        # "insufficient evidence" edge to UNKNOWN. It is born without evidence and
        # can never gain any, and fusion creates it *after*
        # ``finalize_unresolved`` has run, so it would otherwise be reported as
        # `provisional` under an `Unknown` label for the rest of the session.
        speaker.machine.transition(IdentityState.UNKNOWN, "insufficient_evidence")
        self._register(speaker)
        self._unattributed_id = speaker.session_speaker_id
        return speaker

    def _register(self, speaker: SessionSpeaker) -> None:
        if speaker.session_speaker_id in self.speakers:
            raise InvalidStateTransitionError("session speaker IDs are never reused (spec 5.7)")
        self.speakers[speaker.session_speaker_id] = speaker
        self.order.append(speaker.session_speaker_id)

    # -- lookup -------------------------------------------------------------- #

    def get(self, session_speaker_id: str) -> SessionSpeaker:
        speaker = self.speakers[session_speaker_id]
        while speaker.merged_into is not None:
            speaker = self.speakers[speaker.merged_into]
        return speaker

    def by_cluster(self, cluster_id: str) -> SessionSpeaker | None:
        for speaker_id in self.order:
            speaker = self.speakers[speaker_id]
            if speaker.cluster_id == cluster_id and speaker.merged_into is None:
                return speaker
        return None

    def ensure_cluster_speaker(self, cluster_id: str) -> SessionSpeaker:
        existing = self.by_cluster(cluster_id)
        if existing is not None:
            return existing
        return self.create_session_speaker(cluster_id=cluster_id)

    @property
    def active(self) -> list[SessionSpeaker]:
        return [
            self.speakers[speaker_id]
            for speaker_id in self.order
            if self.speakers[speaker_id].merged_into is None
        ]

    @property
    def active_speaker_count(self) -> int:
        return len(self.active)

    @property
    def can_admit_speaker(self) -> bool:
        """Whether another session speaker still fits under spec 0.1.1 / 12.

        Callers check this before minting an identity; the create methods raise
        rather than silently exceeding ``max_session_speakers``.
        """
        return self.active_speaker_count < self.config.product.max_session_speakers

    def prototypes(self) -> list[SpeakerPrototype]:
        """Prototypes usable as linking targets (spec 5.8).

        Provisional identities are excluded on purpose: their buffered embedding
        is evidence awaiting reconciliation (spec 5.9 step 4), not a session
        centroid. Offering it as a linking target would place a near-duplicate
        of a real speaker in the score matrix and collapse the top1/top2 margin.
        """
        return [
            s.prototype
            for s in self.active
            if s.prototype is not None and s.state is not IdentityState.PROVISIONAL
        ]

    def provisional_prototypes(self) -> list[SpeakerPrototype]:
        """Buffered evidence of temporary identities (spec 5.9 step 4)."""
        return [s.prototype for s in self.provisional_speakers() if s.prototype is not None]

    def linking_prototypes(self) -> list[SpeakerPrototype]:
        """Candidates for separated-source continuity within the same session.

        A temporary centroid is not eligible for *global reconciliation*, but it
        is valid evidence that the next separated crop belongs to that same
        temporary source.  Keeping it here avoids minting a new temporary label
        for every 10-second overlap crop.  One-to-one Hungarian assignment and
        cannot-link constraints still prevent concurrent sources sharing it.
        """
        return [speaker.prototype for speaker in self.active if speaker.prototype is not None]

    def provisional_speakers(self) -> list[SessionSpeaker]:
        return [s for s in self.active if s.state is IdentityState.PROVISIONAL]

    # -- prototypes (spec 5.6) ---------------------------------------------- #

    @property
    def minimum_speech_ms(self) -> int:
        return seconds_to_ms(self.config.speaker_embedding.minimum_clean_speech_seconds)

    def update_prototype(
        self,
        session_speaker_id: str,
        embedding: SpeakerEmbedding,
        *,
        link_margin: float | None = None,
        source_quality_passed: bool = True,
    ) -> PrototypeUpdate:
        """Quality-weighted centroid update with the gates of spec 5.6.

        Clean non-overlap speech is preferred; a separated source may only update
        a prototype when the feature flag is on and linking score, margin and
        source quality all pass.
        """
        speaker = self.get(session_speaker_id)

        if embedding.model_version != self.embedding_model_version:
            return PrototypeUpdate(False, REASON_MODEL_MISMATCH, speaker.session_speaker_id)
        if embedding.speech_duration_ms < self.minimum_speech_ms:
            return PrototypeUpdate(False, REASON_TOO_SHORT, speaker.session_speaker_id)
        if speaker.state in (IdentityState.UNKNOWN, IdentityState.AMBIGUOUS):
            return PrototypeUpdate(False, REASON_NOT_ANONYMOUS, speaker.session_speaker_id)

        if embedding.origin == "separated":
            if not self.config.speaker_embedding.update_from_separated_sources:
                return PrototypeUpdate(False, REASON_SEPARATED_DISABLED, speaker.session_speaker_id)
            if not source_quality_passed:
                return PrototypeUpdate(False, REASON_LOW_SOURCE_QUALITY, speaker.session_speaker_id)
            margin_threshold = self.config.source_linking.ambiguous_margin
            if margin_threshold is None or link_margin is None or link_margin < margin_threshold:
                return PrototypeUpdate(False, REASON_LOW_LINK_MARGIN, speaker.session_speaker_id)

        if speaker.prototype is None:
            speaker.prototype = SpeakerPrototype.from_embedding(
                speaker.session_speaker_id, embedding
            )
            return PrototypeUpdate(
                True, REASON_CREATED, speaker.session_speaker_id, speaker.prototype.version
            )

        speaker.prototype_history.append(speaker.prototype)
        speaker.prototype = speaker.prototype.updated_with(embedding)
        return PrototypeUpdate(
            True, REASON_UPDATED, speaker.session_speaker_id, speaker.prototype.version
        )

    def buffer_provisional_embedding(
        self, session_speaker_id: str, embedding: SpeakerEmbedding
    ) -> PrototypeUpdate:
        """Buffer a separated embedding on a temporary identity — spec 5.9 step 4.

        This is the temporary speaker's *own* evidence, kept so reconciliation
        can merge it later. It never touches a global centroid, which is what
        spec 5.9 step 2 forbids.
        """
        speaker = self.get(session_speaker_id)
        if speaker.state is not IdentityState.PROVISIONAL:
            return PrototypeUpdate(False, REASON_NOT_ANONYMOUS, speaker.session_speaker_id)
        if embedding.model_version != self.embedding_model_version:
            return PrototypeUpdate(False, REASON_MODEL_MISMATCH, speaker.session_speaker_id)
        if embedding.speech_duration_ms < self.minimum_speech_ms:
            return PrototypeUpdate(False, REASON_TOO_SHORT, speaker.session_speaker_id)

        if speaker.prototype is None:
            speaker.prototype = SpeakerPrototype.from_embedding(
                speaker.session_speaker_id, embedding
            )
            return PrototypeUpdate(
                True, REASON_CREATED, speaker.session_speaker_id, speaker.prototype.version
            )
        speaker.prototype_history.append(speaker.prototype)
        speaker.prototype = speaker.prototype.updated_with(embedding)
        return PrototypeUpdate(
            True, REASON_UPDATED, speaker.session_speaker_id, speaker.prototype.version
        )

    # -- constraints and merges (spec 5.6, 5.7) ------------------------------ #

    def add_cannot_link(self, first_id: str, second_id: str) -> None:
        """Clusters active at the same time cannot be the same person (spec 5.6)."""
        first = self.get(first_id)
        second = self.get(second_id)
        if first.session_speaker_id == second.session_speaker_id:
            return
        first.cannot_link.add(second.session_speaker_id)
        second.cannot_link.add(first.session_speaker_id)

    def can_merge(self, source_id: str, target_id: str) -> bool:
        source = self.get(source_id)
        target = self.get(target_id)
        if source.session_speaker_id == target.session_speaker_id:
            return False
        return target.session_speaker_id not in source.cannot_link

    def merge(self, source_id: str, target_id: str, reason: str) -> LabelChange | None:
        """Fold ``source`` into ``target``; the target stays canonical (spec 6)."""
        source = self.get(source_id)
        target = self.get(target_id)
        if not self.can_merge(source.session_speaker_id, target.session_speaker_id):
            raise InvalidStateTransitionError(
                "cannot-link constraint forbids merging these speakers (spec 5.6)",
                details={"source": source.session_speaker_id, "target": target.session_speaker_id},
            )

        if source.state is IdentityState.PROVISIONAL:
            source.machine.transition(IdentityState.SESSION_ANONYMOUS, reason)
        if source.state is IdentityState.SESSION_ANONYMOUS:
            source.machine.transition(IdentityState.MERGED, reason)

        if target.prototype is None and source.prototype is not None:
            target.prototype = source.prototype
        source.merged_into = target.session_speaker_id
        target.cannot_link |= source.cannot_link - {target.session_speaker_id}

        change = LabelChange(
            session_speaker_id=source.session_speaker_id,
            previous_label=source.display_label,
            new_label=target.display_label,
            reason=reason,
        )
        source.display_label = target.display_label
        self.pending_label_changes.append(change)
        return change

    # -- identity resolution (spec 5.9, 5.10, 6) ----------------------------- #

    def promote_provisional(
        self, session_speaker_id: str, reason: str, *, label: str | None = None
    ) -> LabelChange | None:
        """``PROVISIONAL -> SESSION_ANONYMOUS`` once linked to session evidence."""
        speaker = self.get(session_speaker_id)
        if speaker.state is not IdentityState.PROVISIONAL:
            return None
        speaker.machine.transition(IdentityState.SESSION_ANONYMOUS, reason)
        if label is None:
            self._label_counter += 1
            label = SPEAKER_LABEL.format(index=self._label_counter)
        change = LabelChange(
            session_speaker_id=speaker.session_speaker_id,
            previous_label=speaker.display_label,
            new_label=label,
            reason=reason,
        )
        speaker.display_label = change.new_label
        self.pending_label_changes.append(change)
        return change

    def apply_voice_id(
        self, session_speaker_id: str, decision: VoiceIdDecision
    ) -> LabelChange | None:
        """Apply an open-set Voice ID decision (spec 5.10, 6).

        ``uncalibrated`` and ``unknown`` leave the session label untouched: the
        system falls back to the session speaker instead of guessing.
        """
        speaker = self.get(session_speaker_id)
        if decision.status == "enrolled":
            if not decision.registry_speaker_id or not decision.speaker_name:
                raise InvalidStateTransitionError(
                    "an enrolled decision requires registry_speaker_id and speaker_name (spec 7)"
                )
            if speaker.state in (IdentityState.PROVISIONAL, IdentityState.SESSION_ANONYMOUS):
                speaker.machine.transition(IdentityState.ENROLLED, "voice_id_accept")
            elif speaker.state is IdentityState.AMBIGUOUS:
                speaker.machine.transition(IdentityState.ENROLLED, "voice_id_reaccept")
            speaker.registry_speaker_id = decision.registry_speaker_id
            speaker.speaker_name = decision.speaker_name
            change = LabelChange(
                session_speaker_id=speaker.session_speaker_id,
                previous_label=speaker.display_label,
                new_label=decision.speaker_name,
                reason="voice_id_accept",
            )
            speaker.display_label = decision.speaker_name
            self.pending_label_changes.append(change)
            return change

        if decision.status == "ambiguous" and speaker.state is IdentityState.ENROLLED:
            speaker.machine.transition(IdentityState.AMBIGUOUS, "voice_id_contradiction")
        return None

    def finalize_unresolved(self) -> list[str]:
        """At finalization a provisional speaker without evidence becomes ``UNKNOWN``.

        Realtime keeps the temporary label when the session ends before the
        evidence arrives (spec 5.9 step 6).
        """
        unresolved: list[str] = []
        for speaker in self.active:
            if speaker.state is not IdentityState.PROVISIONAL:
                continue
            if speaker.prototype is None:
                speaker.machine.transition(IdentityState.UNKNOWN, "insufficient_evidence")
                unresolved.append(speaker.session_speaker_id)
                continue
            # Without calibrated linking thresholds this must remain a
            # provisional/unknown outcome. With the reviewed session threshold,
            # separated evidence can prove one stable *session* speaker even
            # when no later clean diarization centroid exists. It is promoted,
            # never silently merged into another identity.
            if self.config.source_linking.is_calibrated:
                self.promote_provisional(speaker.session_speaker_id, "separated_evidence_final")
        return unresolved

    def drain_label_changes(self) -> list[LabelChange]:
        changes = list(self.pending_label_changes)
        self.pending_label_changes.clear()
        return changes


def overlapping_clusters(
    intervals: dict[str, list[TimeInterval]],
) -> set[tuple[str, str]]:
    """Cluster pairs whose activity overlaps — a cannot-link source (spec 5.6)."""
    pairs: set[tuple[str, str]] = set()
    keys = sorted(intervals)
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            if any(a.intersects(b) for a in intervals[first] for b in intervals[second]):
                pairs.add((first, second))
    return pairs


__all__ = [
    "REASON_CREATED",
    "REASON_LOW_LINK_MARGIN",
    "REASON_LOW_SOURCE_QUALITY",
    "REASON_MODEL_MISMATCH",
    "REASON_NOT_ANONYMOUS",
    "REASON_SEPARATED_DISABLED",
    "REASON_TOO_SHORT",
    "REASON_UPDATED",
    "SPEAKER_LABEL",
    "TEMPORARY_LABEL",
    "LabelChange",
    "PrototypeUpdate",
    "SessionSpeakerState",
    "overlapping_clusters",
]
