create extension if not exists "pgcrypto";
create extension if not exists "citext";

create table tenants (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  status text not null default 'active',
  created_at timestamptz not null default now()
);

create table tenant_vapi_map (
  tenant_id uuid not null references tenants(id) on delete cascade,
  vapi_org_id text,
  vapi_phone_number_id text,
  vapi_assistant_id text,
  created_at timestamptz not null default now(),
  unique (tenant_id, vapi_phone_number_id),
  unique (tenant_id, vapi_assistant_id)
);

create table users (
  id uuid primary key default gen_random_uuid(),
  email citext unique not null,
  full_name text,
  created_at timestamptz not null default now()
);

create table user_tenant_roles (
  user_id uuid not null references users(id) on delete cascade,
  tenant_id uuid not null references tenants(id) on delete cascade,
  role text not null,
  created_at timestamptz not null default now(),
  primary key (user_id, tenant_id)
);

create table calls (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null references tenants(id) on delete cascade,
  vapi_call_id text not null unique,
  vapi_org_id text,
  vapi_phone_number_id text,
  vapi_assistant_id text,
  direction text not null default 'inbound',
  caller_e164 text,
  called_e164 text,
  started_at timestamptz,
  answered_at timestamptz,
  ended_at timestamptz,
  duration_sec integer,
  answer_time_sec integer,
  outcome text,
  ended_reason text,
  escalated boolean not null default false,
  reason text,
  recording_url text,
  transcript_text text,
  summary text,
  structured_data jsonb,
  success_eval jsonb,
  error_code text,
  error_detail text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index idx_calls_tenant_time on calls (tenant_id, started_at desc);
create index idx_calls_tenant_outcome on calls (tenant_id, outcome);
create index idx_calls_tenant_reason on calls (tenant_id, reason);

create table call_variables (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null references tenants(id) on delete cascade,
  vapi_call_id text not null references calls(vapi_call_id) on delete cascade,
  key text not null,
  value_text text,
  value_number numeric,
  value_json jsonb,
  created_at timestamptz not null default now(),
  unique (vapi_call_id, key)
);

create index idx_call_variables_tenant_key on call_variables (tenant_id, key);

create table call_events_raw (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null references tenants(id) on delete cascade,
  vapi_call_id text,
  event_type text not null,
  received_at timestamptz not null default now(),
  payload jsonb not null,
  payload_hash text
);

create index idx_call_events_call on call_events_raw (vapi_call_id, received_at desc);
create index idx_call_events_tenant_time on call_events_raw (tenant_id, received_at desc);
create unique index uniq_event_payload_per_tenant on call_events_raw(tenant_id, payload_hash);

create table tenant_settings (
  tenant_id uuid primary key references tenants(id) on delete cascade,
  sla_target_sec integer not null default 10,
  human_minutes_per_resolved_call numeric not null default 3.5,
  cost_per_minute numeric not null default 0.5,
  currency text not null default 'EUR',
  retention_days integer not null default 90,
  mask_caller boolean not null default true,
  recordings_enabled boolean not null default true,
  recordings_require_manager boolean not null default true,
  updated_at timestamptz not null default now()
);

alter table calls enable row level security;
alter table call_events_raw enable row level security;
alter table call_variables enable row level security;

create policy tenant_isolation_calls
on calls
for select
using (tenant_id = current_setting('app.tenant_id')::uuid);

create policy tenant_isolation_events
on call_events_raw
for select
using (tenant_id = current_setting('app.tenant_id')::uuid);

create policy tenant_isolation_vars
on call_variables
for select
using (tenant_id = current_setting('app.tenant_id')::uuid);
