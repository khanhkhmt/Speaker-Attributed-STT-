"""Speaker identity state machine — spec 6."""

from __future__ import annotations

import pytest

from sastt.domain.errors import InvalidStateTransitionError
from sastt.domain.speakers import (
    ALLOWED_TRANSITIONS,
    IdentityState,
    IdentityStatus,
    SpeakerIdentityStateMachine,
    public_status,
)

pytestmark = pytest.mark.unit


class TestInitialStates:
    def test_only_provisional_and_session_anonymous_start(self) -> None:
        SpeakerIdentityStateMachine(IdentityState.PROVISIONAL)
        SpeakerIdentityStateMachine(IdentityState.SESSION_ANONYMOUS)
        for state in (IdentityState.ENROLLED, IdentityState.MERGED, IdentityState.UNKNOWN):
            with pytest.raises(InvalidStateTransitionError):
                SpeakerIdentityStateMachine(state)


class TestTransitions:
    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (IdentityState.PROVISIONAL, IdentityState.SESSION_ANONYMOUS),
            (IdentityState.PROVISIONAL, IdentityState.ENROLLED),
            (IdentityState.PROVISIONAL, IdentityState.UNKNOWN),
            (IdentityState.SESSION_ANONYMOUS, IdentityState.ENROLLED),
            (IdentityState.SESSION_ANONYMOUS, IdentityState.MERGED),
        ],
    )
    def test_allowed(self, start: IdentityState, target: IdentityState) -> None:
        machine = SpeakerIdentityStateMachine(start)
        machine.transition(target, "test")
        assert machine.state is target

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (IdentityState.PROVISIONAL, IdentityState.MERGED),
            (IdentityState.PROVISIONAL, IdentityState.AMBIGUOUS),
            (IdentityState.SESSION_ANONYMOUS, IdentityState.PROVISIONAL),
            (IdentityState.SESSION_ANONYMOUS, IdentityState.UNKNOWN),
        ],
    )
    def test_forbidden(self, start: IdentityState, target: IdentityState) -> None:
        machine = SpeakerIdentityStateMachine(start)
        with pytest.raises(InvalidStateTransitionError):
            machine.transition(target, "test")

    def test_unknown_is_terminal(self) -> None:
        machine = SpeakerIdentityStateMachine(IdentityState.PROVISIONAL)
        machine.transition(IdentityState.UNKNOWN, "no evidence")
        assert ALLOWED_TRANSITIONS[IdentityState.UNKNOWN] == frozenset()
        for state in IdentityState:
            assert not machine.can_transition(state)

    def test_enrolled_can_become_ambiguous_and_back(self) -> None:
        machine = SpeakerIdentityStateMachine(IdentityState.SESSION_ANONYMOUS)
        machine.transition(IdentityState.ENROLLED, "voice id")
        machine.transition(IdentityState.AMBIGUOUS, "contradiction")
        machine.transition(IdentityState.ENROLLED, "new evidence")
        machine.transition(IdentityState.AMBIGUOUS, "contradiction again")
        machine.transition(IdentityState.UNKNOWN, "final reject")
        assert machine.state is IdentityState.UNKNOWN

    def test_history_is_auditable(self) -> None:
        machine = SpeakerIdentityStateMachine(IdentityState.PROVISIONAL)
        machine.transition(IdentityState.SESSION_ANONYMOUS, "linked")
        machine.transition(IdentityState.MERGED, "reconciliation")
        assert [t.reason for t in machine.history] == ["linked", "reconciliation"]
        assert [t.revision for t in machine.history] == [2, 3]


class TestPublicStatus:
    def test_every_state_maps_to_a_public_status(self) -> None:
        for state in IdentityState:
            assert isinstance(public_status(state), IdentityStatus)

    def test_merged_is_published_as_anonymous(self) -> None:
        assert public_status(IdentityState.MERGED) is IdentityStatus.ANONYMOUS
