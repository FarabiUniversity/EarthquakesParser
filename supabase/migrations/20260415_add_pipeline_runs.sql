-- Create pipeline_runs table to track script execution history.
-- Each row represents one invocation of an ingestion or analysis script.

CREATE TABLE pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
        CONSTRAINT pipeline_runs_status_check
        CHECK (status IN ('running', 'success', 'error')),
    records_processed INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX idx_pipeline_runs_script_name ON pipeline_runs(script_name);
CREATE INDEX idx_pipeline_runs_started_at ON pipeline_runs(started_at DESC);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);

-- Enable Row Level Security
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users full access
CREATE POLICY "Allow all operations for authenticated users on pipeline_runs"
    ON pipeline_runs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow service role full access (for backend scripts)
CREATE POLICY "Allow all operations for service role on pipeline_runs"
    ON pipeline_runs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
