-- Migration: Add user approval system
-- Description: Add approval status to users and admin management features

-- Add approval_status column to users table
ALTER TABLE users
ADD COLUMN IF NOT EXISTS approval_status TEXT NOT NULL DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS approved_by UUID REFERENCES users(id);

-- Add index for querying pending users
CREATE INDEX IF NOT EXISTS idx_users_approval_status ON users(approval_status);

-- Update existing users to be approved (backward compatibility)
UPDATE users
SET approval_status = 'approved',
    approved_at = created_at
WHERE approval_status = 'pending';

-- Add comment
COMMENT ON COLUMN users.approval_status IS 'User approval status: pending, approved, rejected';
