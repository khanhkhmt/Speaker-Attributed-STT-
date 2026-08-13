"""Adapters — concrete implementations behind the ports of spec 9.

Real model adapters (pyannote, 3D-Speaker, ClearerVoice, SpeechBrain,
faster-whisper, GSS) arrive with Milestone 1 and later; Milestone 0 ships the
deterministic fake adapters plus in-memory persistence so the pipelines and the
public contract can be tested without weights (spec 18, Milestone 0).
"""
