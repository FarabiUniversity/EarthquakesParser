CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS kndc_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    newsid BIGINT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'kndc',
    source_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    main_text TEXT NOT NULL DEFAULT '',
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kndc_news_newsid ON kndc_news(newsid DESC);
CREATE INDEX IF NOT EXISTS idx_kndc_news_published_at ON kndc_news(published_at);

CREATE TABLE IF NOT EXISTS kndc_bulletins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id BIGINT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'kndc',
    source_url TEXT NOT NULL UNIQUE,
    epoch_time BIGINT,
    occurred_at_utc TIMESTAMPTZ,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    depth_km DOUBLE PRECISION,
    mb DOUBLE PRECISION,
    mpv DOUBLE PRECISION,
    energy_class_k DOUBLE PRECISION,
    geographic_region TEXT,
    seismic_region TEXT,
    quality TEXT,
    author TEXT,
    last_updated TEXT,
    parsed JSONB NOT NULL DEFAULT '{}'::jsonb,
    enriched_text TEXT NOT NULL DEFAULT '',
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kndc_bulletins_event_id ON kndc_bulletins(event_id DESC);
CREATE INDEX IF NOT EXISTS idx_kndc_bulletins_epoch_time ON kndc_bulletins(epoch_time DESC);
CREATE INDEX IF NOT EXISTS idx_kndc_bulletins_occurred_at_utc ON kndc_bulletins(occurred_at_utc DESC);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_kndc_news_updated_at ON kndc_news;
CREATE TRIGGER update_kndc_news_updated_at
    BEFORE UPDATE ON kndc_news
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_kndc_bulletins_updated_at ON kndc_bulletins;
CREATE TRIGGER update_kndc_bulletins_updated_at
    BEFORE UPDATE ON kndc_bulletins
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE kndc_news ENABLE ROW LEVEL SECURITY;
ALTER TABLE kndc_bulletins ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all operations for authenticated users on kndc_news" ON kndc_news;
CREATE POLICY "Allow all operations for authenticated users on kndc_news"
    ON kndc_news
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow all operations for authenticated users on kndc_bulletins" ON kndc_bulletins;
CREATE POLICY "Allow all operations for authenticated users on kndc_bulletins"
    ON kndc_bulletins
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow all operations for service role on kndc_news" ON kndc_news;
CREATE POLICY "Allow all operations for service role on kndc_news"
    ON kndc_news
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow all operations for service role on kndc_bulletins" ON kndc_bulletins;
CREATE POLICY "Allow all operations for service role on kndc_bulletins"
    ON kndc_bulletins
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
