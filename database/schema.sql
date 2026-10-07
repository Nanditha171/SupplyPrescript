-- ============================================================================
-- SUPPLY PRESCRIPT — DATABASE DDL SCHEMA DEFINITION
-- Target Database: PostgreSQL 15+
-- Closed-Loop Workflow: Predict -> Prescribe -> Decide -> Execute -> Measure -> Learn
-- ============================================================================

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS outcomes CASCADE;
DROP TABLE IF EXISTS decisions CASCADE;
DROP TABLE IF EXISTS recommendations CASCADE;
DROP TABLE IF EXISTS optimization_runs CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS supply_events CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS suppliers CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- 1. PRODUCTS TABLE
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(50) NOT NULL UNIQUE,
    product_name VARCHAR(255) NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    unit_cost NUMERIC(12, 2) NOT NULL,
    criticality_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_products_unit_cost_positive CHECK (unit_cost > 0),
    CONSTRAINT chk_products_criticality_level CHECK (criticality_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

CREATE INDEX idx_products_product_code ON products(product_code);
CREATE INDEX idx_products_product_category ON products(product_category);
CREATE INDEX idx_products_criticality_level ON products(criticality_level);

-- 2. SUPPLIERS TABLE
CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    supplier_code VARCHAR(50) NOT NULL UNIQUE,
    supplier_name VARCHAR(255) NOT NULL,
    supplier_country VARCHAR(100) NOT NULL,
    reliability_score FLOAT NOT NULL,
    supplier_capacity_units INTEGER NOT NULL,
    average_lead_time_days FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_suppliers_reliability_range CHECK (reliability_score >= 0.0 AND reliability_score <= 1.0),
    CONSTRAINT chk_suppliers_capacity_positive CHECK (supplier_capacity_units > 0),
    CONSTRAINT chk_suppliers_lead_time_positive CHECK (average_lead_time_days > 0),
    CONSTRAINT chk_suppliers_risk_level CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

CREATE INDEX idx_suppliers_supplier_code ON suppliers(supplier_code);
CREATE INDEX idx_suppliers_supplier_country ON suppliers(supplier_country);
CREATE INDEX idx_suppliers_risk_level ON suppliers(risk_level);

-- 3. INVENTORY TABLE
CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    warehouse_code VARCHAR(50) NOT NULL,
    inventory_units INTEGER NOT NULL,
    reserved_units INTEGER NOT NULL DEFAULT 0,
    available_units INTEGER NOT NULL,
    reorder_point INTEGER NOT NULL,
    safety_stock_units INTEGER NOT NULL,
    inventory_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_inventory_snapshot UNIQUE (product_id, warehouse_code, inventory_date),
    CONSTRAINT chk_inv_units_non_negative CHECK (inventory_units >= 0),
    CONSTRAINT chk_inv_reserved_non_negative CHECK (reserved_units >= 0),
    CONSTRAINT chk_inv_reserved_lte_total CHECK (reserved_units <= inventory_units),
    CONSTRAINT chk_inv_available_non_negative CHECK (available_units >= 0),
    CONSTRAINT chk_inv_reorder_point_non_negative CHECK (reorder_point >= 0),
    CONSTRAINT chk_inv_safety_stock_non_negative CHECK (safety_stock_units >= 0),
    CONSTRAINT chk_inv_available_calc CHECK (available_units = inventory_units - reserved_units)
);

CREATE INDEX idx_inventory_product_id ON inventory(product_id);
CREATE INDEX idx_inventory_warehouse_code ON inventory(warehouse_code);
CREATE INDEX idx_inventory_inventory_date ON inventory(inventory_date);

-- 4. SUPPLY EVENTS TABLE
CREATE TABLE supply_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100) NOT NULL UNIQUE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id) ON DELETE CASCADE,
    event_date DATE NOT NULL,
    demand_units INTEGER NOT NULL,
    supplier_capacity_units INTEGER NOT NULL,
    planned_lead_time_days FLOAT NOT NULL,
    actual_lead_time_days FLOAT,
    delay_occurred BOOLEAN NOT NULL DEFAULT FALSE,
    delay_days FLOAT NOT NULL DEFAULT 0.0,
    transport_cost NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    unit_cost NUMERIC(12, 2) NOT NULL,
    secondary_supplier_premium NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    air_freight_cost NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    service_risk_score FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_supply_events_demand_non_negative CHECK (demand_units >= 0),
    CONSTRAINT chk_supply_events_capacity_non_negative CHECK (supplier_capacity_units >= 0),
    CONSTRAINT chk_supply_events_planned_lead_time_non_negative CHECK (planned_lead_time_days >= 0),
    CONSTRAINT chk_supply_events_actual_lead_time_non_negative CHECK (actual_lead_time_days IS NULL OR actual_lead_time_days >= 0),
    CONSTRAINT chk_supply_events_delay_days_non_negative CHECK (delay_days >= 0),
    CONSTRAINT chk_supply_events_transport_cost_non_negative CHECK (transport_cost >= 0),
    CONSTRAINT chk_supply_events_unit_cost_non_negative CHECK (unit_cost >= 0),
    CONSTRAINT chk_supply_events_secondary_premium_non_negative CHECK (secondary_supplier_premium >= 0),
    CONSTRAINT chk_supply_events_air_freight_cost_non_negative CHECK (air_freight_cost >= 0),
    CONSTRAINT chk_supply_events_service_risk_score_range CHECK (service_risk_score >= 0.0 AND service_risk_score <= 1.0)
);

CREATE INDEX idx_supply_events_event_id ON supply_events(event_id);
CREATE INDEX idx_supply_events_product_id ON supply_events(product_id);
CREATE INDEX idx_supply_events_supplier_id ON supply_events(supplier_id);
CREATE INDEX idx_supply_events_event_date ON supply_events(event_date);
CREATE INDEX idx_supply_events_delay_occurred ON supply_events(delay_occurred);

-- 5. PREDICTIONS TABLE
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    event_id INTEGER NOT NULL REFERENCES supply_events(id) ON DELETE CASCADE,
    model_name VARCHAR(150) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    delay_probability FLOAT NOT NULL,
    predicted_delay_days FLOAT NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prediction_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_pred_delay_prob_range CHECK (delay_probability >= 0.0 AND delay_probability <= 1.0),
    CONSTRAINT chk_pred_delay_days_non_negative CHECK (predicted_delay_days >= 0),
    CONSTRAINT chk_pred_risk_score_range CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    CONSTRAINT chk_pred_type_enum CHECK (prediction_type IN ('DELAY', 'DEMAND', 'SUPPLY_RISK')),
    CONSTRAINT chk_pred_risk_level_enum CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

CREATE INDEX idx_predictions_event_id ON predictions(event_id);

-- 6. OPTIMIZATION RUNS TABLE
CREATE TABLE optimization_runs (
    id SERIAL PRIMARY KEY,
    event_id INTEGER NOT NULL REFERENCES supply_events(id) ON DELETE CASCADE,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    optimization_run_id VARCHAR(100) NOT NULL UNIQUE,
    objective_type VARCHAR(50) NOT NULL,
    optimization_status VARCHAR(50) NOT NULL,
    total_cost_baseline NUMERIC(14, 2) NOT NULL DEFAULT 0.00,
    recommended_cost NUMERIC(14, 2),
    service_level_target FLOAT NOT NULL DEFAULT 0.95,
    risk_weight FLOAT NOT NULL DEFAULT 0.33,
    cost_weight FLOAT NOT NULL DEFAULT 0.34,
    time_weight FLOAT NOT NULL DEFAULT 0.33,
    solver_name VARCHAR(100) NOT NULL DEFAULT 'PULP_CBC',
    solver_status VARCHAR(50) NOT NULL,
    execution_time_ms FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_opt_total_cost_baseline_non_negative CHECK (total_cost_baseline >= 0),
    CONSTRAINT chk_opt_rec_cost_non_negative CHECK (recommended_cost IS NULL OR recommended_cost >= 0),
    CONSTRAINT chk_opt_service_level_target_range CHECK (service_level_target >= 0.0 AND service_level_target <= 1.0),
    CONSTRAINT chk_opt_risk_weight_non_negative CHECK (risk_weight >= 0),
    CONSTRAINT chk_opt_cost_weight_non_negative CHECK (cost_weight >= 0),
    CONSTRAINT chk_opt_time_weight_non_negative CHECK (time_weight >= 0),
    CONSTRAINT chk_opt_exec_time_non_negative CHECK (execution_time_ms >= 0),
    CONSTRAINT chk_opt_objective_type_enum CHECK (objective_type IN ('MIN_COST', 'MIN_RISK', 'MIN_DELAY', 'BALANCED')),
    CONSTRAINT chk_opt_status_enum CHECK (optimization_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    CONSTRAINT chk_opt_solver_status_enum CHECK (solver_status IN ('OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ERROR'))
);

CREATE INDEX idx_optimization_runs_event_id ON optimization_runs(event_id);
CREATE INDEX idx_optimization_runs_prediction_id ON optimization_runs(prediction_id);
CREATE INDEX idx_optimization_runs_optimization_run_id ON optimization_runs(optimization_run_id);

-- 7. RECOMMENDATIONS TABLE
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    optimization_run_id INTEGER NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    recommendation_code VARCHAR(50) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    strategy_description TEXT NOT NULL,
    estimated_cost NUMERIC(14, 2) NOT NULL,
    estimated_delay_days FLOAT NOT NULL,
    estimated_risk FLOAT NOT NULL,
    service_level FLOAT NOT NULL,
    cost_difference NUMERIC(14, 2) NOT NULL DEFAULT 0.00,
    risk_difference FLOAT NOT NULL DEFAULT 0.0,
    rank INTEGER NOT NULL,
    is_recommended BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_run_recommendation_code UNIQUE (optimization_run_id, recommendation_code),
    CONSTRAINT chk_rec_estimated_cost_non_negative CHECK (estimated_cost >= 0),
    CONSTRAINT chk_rec_estimated_delay_non_negative CHECK (estimated_delay_days >= 0),
    CONSTRAINT chk_rec_estimated_risk_range CHECK (estimated_risk >= 0.0 AND estimated_risk <= 1.0),
    CONSTRAINT chk_rec_service_level_range CHECK (service_level >= 0.0 AND service_level <= 1.0),
    CONSTRAINT chk_rec_rank_positive CHECK (rank >= 1),
    CONSTRAINT chk_rec_strategy_type_enum CHECK (strategy_type IN ('AIR_FREIGHT', 'SECONDARY_SUPPLIER', 'PRIMARY_SUPPLIER', 'DELAY_LAUNCH', 'EXPEDITE', 'PARTIAL_ALLOCATION'))
);

CREATE INDEX idx_recommendations_optimization_run_id ON recommendations(optimization_run_id);

-- 8. DECISIONS TABLE
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(100) NOT NULL UNIQUE,
    optimization_run_id INTEGER NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    recommendation_id INTEGER NOT NULL REFERENCES recommendations(id) ON DELETE RESTRICT,
    decision_maker VARCHAR(150) NOT NULL,
    decision_status VARCHAR(50) NOT NULL,
    decision_reason TEXT,
    selected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    execution_status VARCHAR(50) NOT NULL DEFAULT 'NOT_STARTED',
    executed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_decision_status_enum CHECK (decision_status IN ('PENDING', 'APPROVED', 'REJECTED', 'OVERRIDDEN', 'CANCELLED')),
    CONSTRAINT chk_decision_execution_status_enum CHECK (execution_status IN ('NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED'))
);

CREATE INDEX idx_decisions_decision_id ON decisions(decision_id);
CREATE INDEX idx_decisions_optimization_run_id ON decisions(optimization_run_id);
CREATE INDEX idx_decisions_recommendation_id ON decisions(recommendation_id);

-- 9. OUTCOMES TABLE
CREATE TABLE outcomes (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER NOT NULL UNIQUE REFERENCES decisions(id) ON DELETE CASCADE,
    actual_cost NUMERIC(14, 2) NOT NULL,
    actual_delay_days FLOAT NOT NULL,
    actual_service_level FLOAT NOT NULL,
    actual_risk FLOAT NOT NULL,
    actual_quantity_delivered INTEGER NOT NULL,
    cost_variance NUMERIC(14, 2) NOT NULL,
    delay_variance FLOAT NOT NULL,
    service_level_variance FLOAT NOT NULL,
    outcome_status VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_outcomes_actual_cost_non_negative CHECK (actual_cost >= 0),
    CONSTRAINT chk_outcomes_actual_delay_non_negative CHECK (actual_delay_days >= 0),
    CONSTRAINT chk_outcomes_actual_service_level_range CHECK (actual_service_level >= 0.0 AND actual_service_level <= 1.0),
    CONSTRAINT chk_outcomes_actual_risk_range CHECK (actual_risk >= 0.0 AND actual_risk <= 1.0),
    CONSTRAINT chk_outcomes_actual_qty_non_negative CHECK (actual_quantity_delivered >= 0),
    CONSTRAINT chk_outcomes_status_enum CHECK (outcome_status IN ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILED', 'CANCELLED'))
);

CREATE INDEX idx_outcomes_decision_id ON outcomes(decision_id);

-- 10. FEEDBACK TABLE
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    outcome_id INTEGER NOT NULL REFERENCES outcomes(id) ON DELETE CASCADE,
    optimization_run_id INTEGER NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL,
    cost_error NUMERIC(14, 2) NOT NULL DEFAULT 0.00,
    delay_error FLOAT NOT NULL DEFAULT 0.0,
    risk_error FLOAT NOT NULL DEFAULT 0.0,
    prediction_error FLOAT NOT NULL DEFAULT 0.0,
    weight_adjustment FLOAT NOT NULL DEFAULT 0.0,
    learning_signal TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_feedback_type_enum CHECK (feedback_type IN ('COST_VARIANCE', 'DELAY_VARIANCE', 'PREDICTION_ERROR', 'OPTIMIZATION_ERROR', 'SERVICE_LEVEL_ERROR'))
);

CREATE INDEX idx_feedback_outcome_id ON feedback(outcome_id);
CREATE INDEX idx_feedback_optimization_run_id ON feedback(optimization_run_id);
