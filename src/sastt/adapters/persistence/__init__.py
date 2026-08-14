"""Persistence adapters.

Two backends behind the same ports (spec 10):

* in-memory stores, used by tests and local development;
* PostgreSQL + pgvector, where the unique constraints of spec 10.2 and tenant
  isolation are enforced by the database rather than by Python.

The PostgreSQL imports are lazy so the package stays importable without psycopg.
"""

from sastt.adapters.persistence.memory import (
    IdempotencyConflictError,
    InMemoryEventStore,
    InMemoryJobStore,
    InMemoryObjectStore,
)

__all__ = [
    "IdempotencyConflictError",
    "InMemoryEventStore",
    "InMemoryJobStore",
    "InMemoryObjectStore",
    "PgVectorVoiceRegistry",
    "PostgresEventStore",
    "PostgresJobStore",
    "build_pool",
]


def __getattr__(name: str) -> object:
    """Import the PostgreSQL adapters on first use (psycopg is optional)."""
    if name in {"PostgresEventStore", "PostgresJobStore", "build_pool"}:
        from sastt.adapters.persistence import postgres

        return getattr(postgres, name)
    if name == "PgVectorVoiceRegistry":
        from sastt.adapters.persistence import pgvector_registry

        return pgvector_registry.PgVectorVoiceRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
