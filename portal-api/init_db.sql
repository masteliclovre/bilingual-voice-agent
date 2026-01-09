CREATE TABLE IF NOT EXISTS calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vapi_call_id VARCHAR(255) UNIQUE NOT NULL,
  phone_number VARCHAR(50) NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL,
  ended_at TIMESTAMP WITH TIME ZONE,
  duration_seconds INTEGER,
  status VARCHAR(50) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calls_started_at ON calls(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_phone_number ON calls(phone_number);
CREATE INDEX IF NOT EXISTS idx_calls_vapi_call_id ON calls(vapi_call_id);

CREATE TABLE IF NOT EXISTS transcripts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  speaker VARCHAR(50) NOT NULL,
  text TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcripts_call_id ON transcripts(call_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_timestamp ON transcripts(timestamp);

CREATE TABLE IF NOT EXISTS call_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL UNIQUE REFERENCES calls(id) ON DELETE CASCADE,
  ai_summary TEXT,
  customer_intent VARCHAR(255),
  topics TEXT[],
  sentiment VARCHAR(50),
  escalated BOOLEAN DEFAULT FALSE,
  escalation_reason TEXT,
  escalation_priority VARCHAR(50),
  escalation_resolved BOOLEAN DEFAULT FALSE,
  escalation_resolved_at TIMESTAMP WITH TIME ZONE,
  escalation_notes TEXT,
  requires_followup BOOLEAN DEFAULT FALSE,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_metadata_call_id ON call_metadata(call_id);
CREATE INDEX IF NOT EXISTS idx_call_metadata_customer_intent ON call_metadata(customer_intent);
CREATE INDEX IF NOT EXISTS idx_call_metadata_requires_followup ON call_metadata(requires_followup);
CREATE INDEX IF NOT EXISTS idx_call_metadata_escalated ON call_metadata(escalated);
CREATE INDEX IF NOT EXISTS idx_call_metadata_escalation_resolved ON call_metadata(escalation_resolved) WHERE escalated = TRUE;
CREATE INDEX IF NOT EXISTS idx_call_metadata_escalation_priority ON call_metadata(escalation_priority) WHERE escalated = TRUE;
