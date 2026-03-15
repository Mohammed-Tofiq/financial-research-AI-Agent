-- ============================================================
--  STOCK PORTFOLIO DATABASE  |  PostgreSQL Schema
--  Compatible with the Financial Research AI Agent (notebook)
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- 0. EXTENSIONS
-- ────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()


-- ────────────────────────────────────────────────────────────
-- 1. PORTFOLIO  (mirrors the original SQLite portfolio table)
--    Tracks every stock holding with full detail.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20)     NOT NULL,          -- e.g. RELIANCE.NS, AAPL
    quantity        INTEGER         NOT NULL CHECK (quantity > 0),
    avg_buy_price   NUMERIC(12, 2)  NOT NULL DEFAULT 0.00,   -- average cost basis
    currency        VARCHAR(5)      NOT NULL DEFAULT 'INR',   -- INR, USD, etc.
    exchange        VARCHAR(20),                              -- NSE, BSE, NASDAQ …
    notes           TEXT,
    added_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Index for fast symbol look-ups
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio (symbol);


-- ────────────────────────────────────────────────────────────
-- 2. TRANSACTIONS  (every buy / sell event)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20)     NOT NULL,
    txn_type        VARCHAR(4)      NOT NULL CHECK (txn_type IN ('BUY', 'SELL')),
    quantity        INTEGER         NOT NULL CHECK (quantity > 0),
    price           NUMERIC(12, 2)  NOT NULL CHECK (price > 0),
    currency        VARCHAR(5)      NOT NULL DEFAULT 'INR',
    brokerage_fee   NUMERIC(10, 2)  NOT NULL DEFAULT 0.00,
    notes           TEXT,
    txn_date        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_txn_symbol   ON transactions (symbol);
CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions (txn_date DESC);


-- ────────────────────────────────────────────────────────────
-- 3. WATCHLIST  (stocks the user is monitoring, not yet held)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS watchlist (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20)     NOT NULL UNIQUE,
    target_price    NUMERIC(12, 2),                    -- optional price alert
    alert_above     BOOLEAN         NOT NULL DEFAULT FALSE,
    alert_below     BOOLEAN         NOT NULL DEFAULT FALSE,
    notes           TEXT,
    added_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


-- ────────────────────────────────────────────────────────────
-- 4. PRICE_SNAPSHOTS  (cached prices fetched by the AI agent)
--    Lets you replay historical agent queries without re-hitting yfinance.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS price_snapshots (
    id              BIGSERIAL       PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL,
    price           NUMERIC(12, 2)  NOT NULL,
    currency        VARCHAR(5)      NOT NULL DEFAULT 'INR',
    source          VARCHAR(30)     NOT NULL DEFAULT 'yfinance',
    fetched_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snap_symbol_time
    ON price_snapshots (symbol, fetched_at DESC);


-- ────────────────────────────────────────────────────────────
-- 5. TECHNICAL_SIGNALS  (output of analyze_stock_trend /
--    technical_analysis / moving_average_signal tools)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS technical_signals (
    id              BIGSERIAL       PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL,
    ma5             NUMERIC(12, 2),
    ma20            NUMERIC(12, 2),
    ma50            NUMERIC(12, 2),
    ma200           NUMERIC(12, 2),
    rsi             NUMERIC(6, 2),
    signal          VARCHAR(40),    -- e.g. 'BUY (Golden Cross)', 'SELL (Death Cross)'
    trend           VARCHAR(20),    -- 'Uptrend', 'Downtrend'
    calculated_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signal_symbol_time
    ON technical_signals (symbol, calculated_at DESC);


-- ────────────────────────────────────────────────────────────
-- 6. NEWS_SENTIMENT  (output of analyze_news_sentiment tool)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_sentiment (
    id              BIGSERIAL       PRIMARY KEY,
    query           VARCHAR(100)    NOT NULL,           -- search query used
    symbol          VARCHAR(20),                        -- derived ticker, if any
    sentiment_score NUMERIC(5, 4),                      -- TextBlob polarity: -1 to 1
    sentiment_label VARCHAR(10)     NOT NULL
                        CHECK (sentiment_label IN ('Positive', 'Negative', 'Neutral')),
    raw_headlines   TEXT,                               -- joined headline text
    analyzed_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON news_sentiment (symbol);


-- ────────────────────────────────────────────────────────────
-- 7. AGENT_SESSIONS  (full chat history per session)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_sessions (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name    VARCHAR(100),
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      UUID            NOT NULL
                        REFERENCES agent_sessions (id) ON DELETE CASCADE,
    role            VARCHAR(10)     NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT            NOT NULL,
    tool_name       VARCHAR(50),    -- populated when role = 'tool'
    sent_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_msg_session ON agent_messages (session_id, sent_at);


-- ────────────────────────────────────────────────────────────
-- 8. HELPER VIEW  —  current portfolio value summary
--    (uses the latest price snapshot for each symbol)
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_portfolio_summary AS
SELECT
    p.symbol,
    p.quantity,
    p.avg_buy_price,
    p.currency,
    latest.price                                            AS current_price,
    ROUND(latest.price * p.quantity, 2)                    AS current_value,
    ROUND((latest.price - p.avg_buy_price) * p.quantity, 2) AS unrealized_pnl,
    ROUND(
        ((latest.price - p.avg_buy_price) / NULLIF(p.avg_buy_price, 0)) * 100,
        2
    )                                                       AS pnl_pct,
    latest.fetched_at                                       AS price_as_of
FROM portfolio p
LEFT JOIN LATERAL (
    SELECT price, fetched_at
    FROM   price_snapshots ps
    WHERE  ps.symbol = p.symbol
    ORDER  BY ps.fetched_at DESC
    LIMIT  1
) latest ON TRUE;


-- ────────────────────────────────────────────────────────────
-- 9. TRIGGER — keep portfolio.updated_at fresh on every update
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION trg_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS set_updated_at ON portfolio;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON portfolio
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();


-- ────────────────────────────────────────────────────────────
-- 10. SEED DATA  —  a few sample rows to verify the schema
-- ────────────────────────────────────────────────────────────
INSERT INTO portfolio (symbol, quantity, avg_buy_price, currency, exchange)
VALUES
    ('RELIANCE.NS', 10,  2850.00, 'INR', 'NSE'),
    ('TCS.NS',       5,  3950.00, 'INR', 'NSE'),
    ('INFY.NS',     20,  1480.00, 'INR', 'NSE'),
    ('AAPL',        15,   175.00, 'USD', 'NASDAQ'),
    ('BTC-USD',      1, 42000.00, 'USD', 'CRYPTO')
ON CONFLICT DO NOTHING;

INSERT INTO watchlist (symbol, target_price, alert_below, notes)
VALUES
    ('ZOMATO.NS', 180.00, TRUE,  'Watch for breakout above ₹180'),
    ('TSLA',      200.00, FALSE, 'Waiting for dip')
ON CONFLICT DO NOTHING;


-- ────────────────────────────────────────────────────────────
-- Done!  Connect string example:
--   postgresql://postgres:<password>@localhost:5432/stock_portfolio
-- ────────────────────────────────────────────────────────────
