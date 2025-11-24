-- Supabase Database Schema
-- Run this SQL in your Supabase SQL Editor to create the required tables

-- Users table
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- RL State table
CREATE TABLE IF NOT EXISTS rl_state (
    username TEXT PRIMARY KEY,
    mode_alpha JSONB NOT NULL DEFAULT '{"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0}',
    mode_beta JSONB NOT NULL DEFAULT '{"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0}',
    mode_history JSONB NOT NULL DEFAULT '[]',
    chunk_performance JSONB NOT NULL DEFAULT '{}',
    file_mapping JSONB NOT NULL DEFAULT '{}',
    survey_completed BOOLEAN NOT NULL DEFAULT FALSE,
    initial_preference TEXT,
    total_sessions INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_rl_state_username ON rl_state(username);

-- Enable Row Level Security (RLS) - Optional but recommended
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE rl_state ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your security needs)
-- For now, we'll use service role key which bypasses RLS
-- In production, you may want to add proper RLS policies

