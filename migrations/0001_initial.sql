-- Initial schema — spec 10.1, 10.2, 10.3.
--
-- Every table carrying customer data is tenant-scoped, because spec 14.2 takes
-- the tenant from auth claims and requires every registry query to be scoped by
-- it. Voice templates are sensitive biometric data (spec 14.3): they live in
-- their own table with a tenant column, a model release reference so vectors of
-- different embedders are never compared (spec 5.6), and ON DELETE CASCADE so
-- deleting an identity really removes them (spec 10.3).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --------------------------------------------------------------------------
-- Model and calibration releases (spec 10.2, 11.2)
-- --------------------------------------------------------------------------

CREATE TABLE model_releases (
    id            TEXT PRIMARY KEY,
    component     TEXT        NOT NULL,
    backend       TEXT        NOT NULL,
    revision      TEXT,
    sha256        TEXT,
    license       TEXT,
    enabled       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Thresholds stay NULL until a calibration release exists; the worker fails
-- closed rather than inventing a number (spec 0.3, 5.10, 18 rule 7).
CREATE TABLE calibration_releases (
    id                TEXT PRIMARY KEY,
    model_release_ids TEXT[]      NOT NULL DEFAULT '{}',
    domain            TEXT,
    metrics           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    thresholds        JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- Jobs and sessions (spec 8.1, 8.2, 10.2)
-- --------------------------------------------------------------------------

CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    idempotency_key TEXT        NOT NULL,
    state           TEXT        NOT NULL,
    input_hash      TEXT,
    config_version  TEXT,
    error_code      TEXT,
    warnings        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    degraded_mode   BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Spec 10.2: retrying with the same key must not create a second job.
CREATE UNIQUE INDEX jobs_tenant_idempotency_key ON jobs (tenant_id, idempotency_key);
CREATE INDEX jobs_tenant_state ON jobs (tenant_id, state);

CREATE TABLE sessions (
    id             TEXT PRIMARY KEY,
    tenant_id      TEXT        NOT NULL,
    mode           TEXT        NOT NULL,
    state          TEXT        NOT NULL,
    config_version TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finalized_at   TIMESTAMPTZ
);

CREATE INDEX sessions_tenant ON sessions (tenant_id);

CREATE TABLE speaker_clusters (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    label                TEXT,
    canonical_cluster_id TEXT,
    prototype_version    INTEGER     NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX speaker_clusters_session ON speaker_clusters (session_id);

-- A delivered event is superseded, never rewritten (spec 6, FR-011), so the
-- log is append-only and unique per (session, sequence).
CREATE TABLE transcript_events (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    sequence_number     BIGINT      NOT NULL,
    revision            INTEGER     NOT NULL DEFAULT 1,
    event_type          TEXT        NOT NULL,
    supersedes_event_id TEXT,
    is_final            BOOLEAN     NOT NULL DEFAULT FALSE,
    payload             JSONB       NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX transcript_events_session_sequence
    ON transcript_events (session_id, sequence_number);
CREATE UNIQUE INDEX transcript_events_session_event_revision
    ON transcript_events (session_id, id, revision);
CREATE INDEX transcript_events_session_final ON transcript_events (session_id, is_final);

-- Canonical final segments, kept separately from the event log so the result
-- endpoint does not have to replay events (spec 8.1).
CREATE TABLE transcript_segments (
    id                 TEXT PRIMARY KEY,
    job_id             TEXT,
    session_id         TEXT        NOT NULL,
    tenant_id          TEXT        NOT NULL,
    start_ms           BIGINT      NOT NULL,
    end_ms             BIGINT      NOT NULL,
    session_speaker_id TEXT,
    source_track       INTEGER,
    is_overlap         BOOLEAN     NOT NULL DEFAULT FALSE,
    payload            JSONB       NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT transcript_segments_interval CHECK (start_ms >= 0 AND start_ms < end_ms)
);

CREATE INDEX transcript_segments_job ON transcript_segments (job_id);
CREATE INDEX transcript_segments_session ON transcript_segments (session_id, start_ms);

-- --------------------------------------------------------------------------
-- Voice registry — sensitive biometric data (spec 5.10, 10.2, 14.3)
-- --------------------------------------------------------------------------

CREATE TABLE voice_identities (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT        NOT NULL,
    external_id  TEXT,
    display_name TEXT,
    status       TEXT        NOT NULL DEFAULT 'active',
    consent_ref  TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at   TIMESTAMPTZ
);

CREATE UNIQUE INDEX voice_identities_tenant_external
    ON voice_identities (tenant_id, external_id)
    WHERE external_id IS NOT NULL;
CREATE INDEX voice_identities_tenant ON voice_identities (tenant_id);

-- 192 dimensions: CAM++ (spec 0.2). model_release_id keeps vectors of different
-- embedders apart — comparing across versions is forbidden (spec 5.6).
CREATE TABLE voice_templates (
    id                TEXT PRIMARY KEY,
    identity_id       TEXT        NOT NULL REFERENCES voice_identities (id) ON DELETE CASCADE,
    tenant_id         TEXT        NOT NULL,
    model_release_id  TEXT        NOT NULL,
    vector            vector(192) NOT NULL,
    quality           REAL        NOT NULL,
    speech_ms         INTEGER     NOT NULL,
    source_hash       TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Spec 10.2: the same enrollment clip cannot be stored twice for one release.
CREATE UNIQUE INDEX voice_templates_identity_release_source
    ON voice_templates (identity_id, model_release_id, source_hash);
CREATE INDEX voice_templates_tenant_release ON voice_templates (tenant_id, model_release_id);

-- Cosine index: open-set Voice ID scores by cosine similarity (spec 5.10).
CREATE INDEX voice_templates_vector_cosine
    ON voice_templates USING hnsw (vector vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Audit (spec 10.3, 14.9, FR-014)
-- --------------------------------------------------------------------------

CREATE TABLE audit_events (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT        NOT NULL,
    actor       TEXT,
    action      TEXT        NOT NULL,
    subject_id  TEXT,
    reason      TEXT,
    details     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX audit_events_tenant_created ON audit_events (tenant_id, created_at DESC);
