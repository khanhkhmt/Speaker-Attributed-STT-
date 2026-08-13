# Model manifests

One manifest per model backend, per spec 11.2 and the licence gate of spec 20.

Every manifest records **three independent layers** (spec 20): source-code licence,
model-weight licence and training-data caveat. A permissive repository licence does
not automatically grant commercial rights to a checkpoint.

`revision` / `sha256` are `null` until the weights are actually pinned during the
model preparation stage. `sastt.config.validate_for_environment` refuses to start a
**production** process against an unpinned or denied backend (spec 0.3, 11.2, 20),
so filling these in is part of Milestone 1, not of Milestone 0 — no weights are
downloaded in CI and none are committed to Git (spec 18).

`production_action` values map onto the spec 20 table:

| value | meaning |
|---|---|
| `allow` / `production_candidate` | may serve production traffic |
| `beta_only` | production only when `requires_flag` is explicitly enabled |
| `phase_2`, `experimental` | not part of V1; needs its flag plus a benchmark |
| `deny` | MUST NOT run in production under any flag |
