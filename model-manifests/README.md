# Model manifests

One manifest per model backend, per spec 11.2 and the licence gate of spec 20.

Every manifest records **three independent layers** (spec 20): source-code licence,
model-weight licence and training-data caveat. A permissive repository licence does
not automatically grant commercial rights to a checkpoint.

The six M1 core manifests (CAM++, faster-whisper turbo, faster-whisper large-v3,
MossFormer2, pyannote community-1 and pyannote segmentation-3.0) are currently
pinned with revision and SHA-256. The optional phase/beta/research manifests may
remain `null`; they are not M1 readiness failures.

`sastt.config.validate_for_environment` refuses to start a **production** process
against an unpinned or denied required backend (spec 0.3, 11.2, 20). Verify staged
weights with `python3 deploy/prestage_models.py --verify --models-dir /models`; no
weight is committed to Git. The full operating procedure is in
[`../docs/running-guide.md`](../docs/running-guide.md).

`production_action` values map onto the spec 20 table:

| value | meaning |
|---|---|
| `allow` / `production_candidate` | may serve production traffic |
| `beta_only` | production only when `requires_flag` is explicitly enabled |
| `phase_2`, `experimental` | not part of V1; needs its flag plus a benchmark |
| `deny` | MUST NOT run in production under any flag |
