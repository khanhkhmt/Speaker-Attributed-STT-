"""Speaker-Attributed STT (``sastt``).

Implementation of ``docs/production-technical-spec.md`` v1.0.

Layering (spec 0.3 / 17):

* ``sastt.domain``      — pure types, invariants, state machine. No framework imports.
* ``sastt.ports``       — typed protocols every model/infrastructure adapter must satisfy.
* ``sastt.application`` — orchestration: routing, linking, session state, fusion, pipelines.
* ``sastt.adapters``    — concrete model/infrastructure implementations behind the ports.
* ``sastt.api``         — public contract (schema v2) and transport.

API, schema and domain logic MUST NOT import a model framework directly (spec 0.3).
"""

SCHEMA_VERSION = "2.0"
SPEC_VERSION = "1.0"

__all__ = ["SCHEMA_VERSION", "SPEC_VERSION"]
