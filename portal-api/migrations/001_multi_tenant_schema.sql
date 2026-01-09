-- Multi-tenant schema migration
-- This migration converts the existing single-tenant schema to multi-tenant

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "citext";

-- Drop existing policies if they exist (from old schema)
DROP POLICY IF EXISTS tenant_isolation_calls ON calls;
DROP POLICY IF EXISTS tenant_isolation_events ON call_events_raw;
DROP POLICY IF EXISTS tenant_isolation_vars ON call_variables;

-- Create new tables first (order matters due to foreign keys)

-- Tenants table
CREATE TABLE IF NOT EXISTS tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- VAPI mapping per tenant
CREATE TABLE IF NOT EXISTS tenant_vapi_map (
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  vapi_org_id TEXT,
  vapi_phone_number_id TEXT,
  vapi_assistant_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (tenant_id, vapi_phone_number_id),
  UNIQUE (tenant_id, vapi_assistant_id)
);

-- Users table (for Google OAuth)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email CITEXT UNIQUE NOT NULL,
  full_name TEXT,
  google_id TEXT UNIQUE,
  profile_picture TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- User-tenant role mapping
CREATE TABLE IF NOT EXISTS user_tenant_roles (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  role TEXT NOT NULL,  -- 'admin', 'manager', 'viewer'
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, tenant_id)
);

-- Tenant settings
CREATE TABLE IF NOT EXISTS tenant_settings (
  tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
  sla_target_sec INTEGER NOT NULL DEFAULT 10,
  human_minutes_per_resolved_call NUMERIC NOT NULL DEFAULT 3.5,
  cost_per_minute NUMERIC NOT NULL DEFAULT 0.5,
  currency TEXT NOT NULL DEFAULT 'EUR',
  retention_days INTEGER NOT NULL DEFAULT 90,
  mask_caller BOOLEAN NOT NULL DEFAULT TRUE,
  recordings_enabled BOOLEAN NOT NULL DEFAULT TRUE,
  recordings_require_manager BOOLEAN NOT NULL DEFAULT TRUE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Backup existing calls table
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'calls') THEN
    -- Create backup
    CREATE TABLE IF NOT EXISTS calls_backup AS SELECT * FROM calls;

    -- Drop old table
    DROP TABLE IF EXISTS calls CASCADE;
  END IF;
END $$;

-- Create new calls table with tenant support
CREATE TABLE calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  vapi_call_id TEXT NOT NULL UNIQUE,
  vapi_org_id TEXT,
  vapi_phone_number_id TEXT,
  vapi_assistant_id TEXT,
  direction TEXT NOT NULL DEFAULT 'inbound',
  caller_e164 TEXT,
  called_e164 TEXT,
  started_at TIMESTAMPTZ,
  answered_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  duration_sec INTEGER,
  answer_time_sec INTEGER,
  outcome TEXT,
  ended_reason TEXT,
  escalated BOOLEAN NOT NULL DEFAULT FALSE,
  reason TEXT,
  recording_url TEXT,
  transcript_text TEXT,
  summary TEXT,
  structured_data JSONB,
  success_eval JSONB,
  error_code TEXT,
  error_detail TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_calls_tenant_time ON calls (tenant_id, started_at DESC);
CREATE INDEX idx_calls_tenant_outcome ON calls (tenant_id, outcome);
CREATE INDEX idx_calls_tenant_reason ON calls (tenant_id, reason);
CREATE INDEX idx_calls_vapi_call_id ON calls (vapi_call_id);

-- Call variables
CREATE TABLE IF NOT EXISTS call_variables (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  vapi_call_id TEXT NOT NULL REFERENCES calls(vapi_call_id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value_text TEXT,
  value_number NUMERIC,
  value_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (vapi_call_id, key)
);

CREATE INDEX idx_call_variables_tenant_key ON call_variables (tenant_id, key);

-- Raw webhook events
CREATE TABLE IF NOT EXISTS call_events_raw (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  vapi_call_id TEXT,
  event_type TEXT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  payload JSONB NOT NULL,
  payload_hash TEXT
);

CREATE INDEX idx_call_events_call ON call_events_raw (vapi_call_id, received_at DESC);
CREATE INDEX idx_call_events_tenant_time ON call_events_raw (tenant_id, received_at DESC);
CREATE UNIQUE INDEX uniq_event_payload_per_tenant ON call_events_raw(tenant_id, payload_hash);

-- Drop old tables that are replaced
DROP TABLE IF EXISTS transcripts CASCADE;
DROP TABLE IF EXISTS call_metadata CASCADE;

-- Enable Row Level Security
ALTER TABLE calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_events_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_variables ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY tenant_isolation_calls
ON calls
FOR SELECT
USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

CREATE POLICY tenant_isolation_events
ON call_events_raw
FOR SELECT
USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

CREATE POLICY tenant_isolation_vars
ON call_variables
FOR SELECT
USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

-- Insert default tenant for migration
INSERT INTO tenants (id, name, status)
VALUES ('00000000-0000-0000-0000-000000000001'::UUID, 'Default Tenant', 'active')
ON CONFLICT (id) DO NOTHING;

-- Insert default tenant settings
INSERT INTO tenant_settings (tenant_id)
VALUES ('00000000-0000-0000-0000-000000000001'::UUID)
ON CONFLICT (tenant_id) DO NOTHING;

-- Migration complete message
DO $$
BEGIN
  RAISE NOTICE 'Multi-tenant schema migration completed successfully!';
  RAISE NOTICE 'Default tenant ID: 00000000-0000-0000-0000-000000000001';
END $$;
