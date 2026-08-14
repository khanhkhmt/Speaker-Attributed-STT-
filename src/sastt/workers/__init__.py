"""Worker processes — spec 11.1.

Each worker is its own process so a GPU stage never blocks the API, and so ASR
and speaker work can be scaled and version-pinned independently.
"""
