-- Add 'ingested' to the existing processing_status enum.
-- Safe to run even if the value already exists.

DO $$
BEGIN
    ALTER TYPE processing_status ADD VALUE 'ingested';
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;
