"""Work queue adapters — spec 11.3.

Redis-backed, one queue per stage, realtime ahead of batch. The import is lazy
so the package stays usable without the redis client installed.
"""

__all__ = ["ALL_QUEUES", "QueueOverloadedError", "RedisTaskQueue", "Task", "build_client"]


def __getattr__(name: str) -> object:
    if name in set(__all__):
        from sastt.adapters.queue import redis_queue

        return getattr(redis_queue, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
