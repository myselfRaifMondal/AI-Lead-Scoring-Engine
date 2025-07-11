-- AI Lead Scoring Engine Database Schema
-- PostgreSQL with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Leads table
CREATE TABLE leads (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Contact information
    email VARCHAR(255),
    phone VARCHAR(50),
    name VARCHAR(255),
    
    -- Basic demographics
    age INTEGER,
    annual_income DECIMAL(12, 2),
    employment_type VARCHAR(100),
    marital_status VARCHAR(50),
    family_size INTEGER,
    
    -- Location
    city VARCHAR(100),
    state VARCHAR(100),
    zip_code VARCHAR(20),
    
    -- Credit information
    credit_score INTEGER,
    debt_to_income_ratio DECIMAL(5, 2),
    
    -- Property preferences
    preferred_property_type VARCHAR(100),
    price_range_min DECIMAL(12, 2),
    price_range_max DECIMAL(12, 2),
    
    -- Status
    lead_status VARCHAR(50) DEFAULT 'active',
    conversion_status VARCHAR(50) DEFAULT 'pending',
    converted_at TIMESTAMP,
    
    -- Compliance
    consent_marketing BOOLEAN DEFAULT FALSE,
    consent_data_processing BOOLEAN DEFAULT FALSE,
    data_retention_expires_at TIMESTAMP,
    
    -- Indexing
    CONSTRAINT chk_lead_status CHECK (lead_status IN ('active', 'inactive', 'converted', 'disqualified')),
    CONSTRAINT chk_conversion_status CHECK (conversion_status IN ('pending', 'converted', 'not_converted'))
);

-- Behavioral events table
CREATE TABLE behavioral_events (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100) REFERENCES leads(lead_id),
    event_type VARCHAR(100) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_data JSONB,
    
    -- Event categories
    category VARCHAR(50), -- 'website', 'email', 'communication', 'search'
    
    -- Specific metrics
    page_url TEXT,
    session_id VARCHAR(100),
    time_spent_seconds INTEGER,
    
    -- Indexing
    INDEX idx_behavioral_events_lead_id (lead_id),
    INDEX idx_behavioral_events_timestamp (event_timestamp),
    INDEX idx_behavioral_events_type (event_type),
    INDEX idx_behavioral_events_category (category)
);

-- Communication data table
CREATE TABLE communication_data (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100) REFERENCES leads(lead_id),
    communication_type VARCHAR(50), -- 'email', 'whatsapp', 'phone', 'chat'
    direction VARCHAR(20), -- 'inbound', 'outbound'
    content TEXT,
    sentiment_score DECIMAL(3, 2),
    intent_signals JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Email specific
    email_opened BOOLEAN,
    email_clicked BOOLEAN,
    email_subject VARCHAR(255),
    
    -- WhatsApp/Chat specific
    message_read BOOLEAN,
    response_time_minutes INTEGER,
    
    -- Indexing
    INDEX idx_communication_lead_id (lead_id),
    INDEX idx_communication_timestamp (created_at),
    INDEX idx_communication_type (communication_type)
);

-- Feature vectors table (for pgvector)
CREATE TABLE feature_vectors (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100) UNIQUE REFERENCES leads(lead_id),
    feature_vector vector(300), -- 300-dimensional vector
    feature_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing for vector similarity search
    INDEX idx_feature_vectors_lead_id (lead_id),
    INDEX idx_feature_vectors_vector USING ivfflat (feature_vector vector_cosine_ops) WITH (lists = 100)
);

-- Model predictions table
CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100) REFERENCES leads(lead_id),
    model_version VARCHAR(50),
    prediction_score DECIMAL(5, 4), -- 0.0000 to 1.0000
    prediction_class INTEGER, -- 0 or 1
    confidence_score DECIMAL(5, 4),
    
    -- Model breakdown
    xgboost_score DECIMAL(5, 4),
    lightgbm_score DECIMAL(5, 4),
    neural_network_score DECIMAL(5, 4),
    llm_adjustment DECIMAL(5, 4),
    
    -- Explanation
    top_features JSONB,
    explanation_text TEXT,
    
    -- Metadata
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    
    -- Business outcome
    actual_outcome INTEGER, -- Updated when conversion happens
    outcome_timestamp TIMESTAMP,
    
    -- Indexing
    INDEX idx_predictions_lead_id (lead_id),
    INDEX idx_predictions_timestamp (prediction_timestamp),
    INDEX idx_predictions_score (prediction_score DESC),
    INDEX idx_predictions_model_version (model_version)
);

-- Market context table
CREATE TABLE market_context (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100),
    date DATE,
    
    -- Market indicators
    median_home_price DECIMAL(12, 2),
    price_change_3m DECIMAL(5, 2), -- Percentage change
    inventory_months DECIMAL(4, 2),
    days_on_market_avg INTEGER,
    
    -- Economic indicators
    mortgage_rate DECIMAL(5, 3),
    unemployment_rate DECIMAL(5, 2),
    
    -- Indexing
    UNIQUE(location, date),
    INDEX idx_market_context_location (location),
    INDEX idx_market_context_date (date)
);

-- Data quality monitoring table
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10, 4),
    threshold_value DECIMAL(10, 4),
    status VARCHAR(20), -- 'healthy', 'warning', 'critical'
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing
    INDEX idx_data_quality_table (table_name),
    INDEX idx_data_quality_timestamp (measured_at)
);

-- Model performance monitoring table
CREATE TABLE model_performance_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10, 6),
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    sample_size INTEGER,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing
    INDEX idx_model_performance_version (model_version),
    INDEX idx_model_performance_timestamp (calculated_at),
    INDEX idx_model_performance_metric (metric_name)
);

-- Audit log table for compliance
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(100),
    action VARCHAR(100),
    performed_by VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing
    INDEX idx_audit_log_lead_id (lead_id),
    INDEX idx_audit_log_timestamp (timestamp),
    INDEX idx_audit_log_action (action)
);

-- Views for common queries
CREATE VIEW lead_scoring_dashboard AS
SELECT 
    l.lead_id,
    l.name,
    l.email,
    l.lead_status,
    l.conversion_status,
    mp.prediction_score,
    mp.prediction_class,
    mp.confidence_score,
    mp.explanation_text,
    mp.prediction_timestamp,
    l.created_at as lead_created_at,
    l.converted_at
FROM leads l
LEFT JOIN model_predictions mp ON l.lead_id = mp.lead_id
WHERE mp.id IN (
    SELECT MAX(id) 
    FROM model_predictions 
    GROUP BY lead_id
);

-- High-intent leads view
CREATE VIEW high_intent_leads AS
SELECT 
    l.*,
    mp.prediction_score,
    mp.confidence_score,
    mp.explanation_text,
    mp.prediction_timestamp
FROM leads l
JOIN model_predictions mp ON l.lead_id = mp.lead_id
WHERE mp.prediction_score > 0.7
    AND l.lead_status = 'active'
    AND mp.id IN (
        SELECT MAX(id) 
        FROM model_predictions 
        GROUP BY lead_id
    )
ORDER BY mp.prediction_score DESC;

-- Functions for data management
CREATE OR REPLACE FUNCTION update_lead_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_lead_updated_at
    BEFORE UPDATE ON leads
    FOR EACH ROW
    EXECUTE FUNCTION update_lead_timestamp();

CREATE TRIGGER update_feature_vectors_updated_at
    BEFORE UPDATE ON feature_vectors
    FOR EACH ROW
    EXECUTE FUNCTION update_lead_timestamp();

-- Data retention function
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete expired lead data based on retention policy
    DELETE FROM behavioral_events 
    WHERE event_timestamp < NOW() - INTERVAL '2 years';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old predictions (keep last 6 months)
    DELETE FROM model_predictions 
    WHERE prediction_timestamp < NOW() - INTERVAL '6 months';
    
    -- Clean up old performance metrics
    DELETE FROM model_performance_metrics 
    WHERE calculated_at < NOW() - INTERVAL '1 year';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_behavioral_events_lead_timestamp 
ON behavioral_events(lead_id, event_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_predictions_lead_timestamp 
ON model_predictions(lead_id, prediction_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_communication_lead_timestamp 
ON communication_data(lead_id, created_at DESC);

-- Partitioning for large tables (optional, for high volume)
-- CREATE TABLE behavioral_events_y2024m01 PARTITION OF behavioral_events
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Sample data for testing
INSERT INTO leads (lead_id, name, email, age, annual_income, employment_type, credit_score, preferred_property_type, consent_marketing, consent_data_processing) VALUES
('lead_001', 'John Doe', 'john.doe@example.com', 32, 75000, 'Full-time', 720, 'Single Family', TRUE, TRUE),
('lead_002', 'Jane Smith', 'jane.smith@example.com', 28, 65000, 'Full-time', 680, 'Condo', TRUE, TRUE),
('lead_003', 'Bob Johnson', 'bob.johnson@example.com', 45, 120000, 'Full-time', 780, 'Single Family', TRUE, TRUE);

INSERT INTO behavioral_events (lead_id, event_type, category, event_data) VALUES
('lead_001', 'page_view', 'website', '{"page": "property_listing", "property_id": "prop_123", "time_spent": 180}'),
('lead_001', 'form_submission', 'website', '{"form_type": "contact", "fields": ["name", "email", "phone"]}'),
('lead_002', 'email_open', 'email', '{"campaign_id": "welcome_series", "subject": "Welcome to Our Platform"}'),
('lead_003', 'search_activity', 'website', '{"search_terms": "3 bedroom house", "location": "San Francisco", "price_range": "800000-1200000"}');

INSERT INTO market_context (location, date, median_home_price, price_change_3m, inventory_months, mortgage_rate) VALUES
('San Francisco', '2024-01-01', 1200000, 2.5, 2.8, 7.2),
('Los Angeles', '2024-01-01', 800000, 1.8, 3.2, 7.2),
('New York', '2024-01-01', 650000, 3.1, 4.1, 7.2);

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO lead_scoring_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO lead_scoring_app;
