"""Persistence adapters.

Milestone 0 ships in-memory stores that enforce the unique constraints of spec
10.2; the PostgreSQL/pgvector implementation arrives with Milestone 2.
"""

from sastt.adapters.persistence.memory import (
    InMemoryEventStore,
    InMemoryJobStore,
    InMemoryObjectStore,
)

__all__ = ["InMemoryEventStore", "InMemoryJobStore", "InMemoryObjectStore"]
