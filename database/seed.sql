-- ============================================================================
-- SUPPLY PRESCRIPT — SEED DATASET (PostgreSQL 15+)
-- Closed-Loop Supply Chain Operations Dataset
-- Includes Core Demonstration Scenario: Microchip X1 Disruption & Mitigation
-- ============================================================================

-- 1. SEED PRODUCTS (8+ Products across diverse categories)
INSERT INTO products (id, product_code, product_name, product_category, unit_cost, criticality_level) VALUES
(1, 'P001', 'Microchip X1 High-Density MCU', 'Electronics', 125.50, 'CRITICAL'),
(2, 'P002', 'Automotive Lithium Cell Module', 'Energy Storage', 85.00, 'CRITICAL'),
(3, 'P003', 'Precision Aluminum Enclosure', 'Mechanical Hardware', 34.20, 'HIGH'),
(4, 'P004', 'Industrial Optic Sensor Module', 'Sensors', 48.75, 'HIGH'),
(5, 'P005', 'High-Temp Copper Wiring Harness', 'Electrical Cables', 18.50, 'MEDIUM'),
(6, 'P006', 'Silicone Thermal Interface Pad', 'Thermal Materials', 4.80, 'LOW'),
(7, 'P007', 'Titanium Fastener Fast-Pack 100', 'Fasteners', 12.30, 'LOW'),
(8, 'P008', 'Heavy-Duty Actuator Motor 24V', 'Electromechanical', 210.00, 'HIGH')
ON CONFLICT (product_code) DO NOTHING;

-- 2. SEED SUPPLIERS (8+ Global Suppliers)
INSERT INTO suppliers (id, supplier_code, supplier_name, supplier_country, reliability_score, supplier_capacity_units, average_lead_time_days, risk_level) VALUES
(1, 'SUP-001', 'Pacific Silicon Semiconductor Ltd', 'Taiwan', 0.88, 15000, 14.0, 'HIGH'),
(2, 'SUP-002', 'Shenzhen Powercell Battery Corp', 'China', 0.91, 25000, 18.0, 'MEDIUM'),
(3, 'SUP-003', 'Kyoto Precision Dynamics', 'Japan', 0.98, 8000, 7.0, 'LOW'),
(4, 'SUP-004', 'EuroSensors AG', 'Germany', 0.95, 12000, 10.0, 'LOW'),
(5, 'SUP-005', 'Austin Advanced Components LLC', 'USA', 0.94, 20000, 5.0, 'LOW'),
(6, 'SUP-006', 'Vanguard Thermal Solutions', 'South Korea', 0.85, 30000, 15.0, 'MEDIUM'),
(7, 'SUP-007', 'Alpine Fastener & Tooling SA', 'Switzerland', 0.97, 50000, 6.0, 'LOW'),
(8, 'SUP-008', 'Veritas Motion Technologies', 'Canada', 0.79, 6000, 21.0, 'CRITICAL')
ON CONFLICT (supplier_code) DO NOTHING;

-- 3. SEED INVENTORY (Multi-warehouse stock positions)
INSERT INTO inventory (product_id, warehouse_code, inventory_units, reserved_units, available_units, reorder_point, safety_stock_units, inventory_date) VALUES
(1, 'WH-NORTH-01', 2500, 500, 2000, 1200, 600, '2026-10-01'),
(1, 'WH-SOUTH-02', 1800, 300, 1500, 1000, 500, '2026-10-01'),
(2, 'WH-NORTH-01', 4000, 1000, 3000, 2000, 1000, '2026-10-01'),
(3, 'WH-CENTRAL-01', 6000, 1200, 4800, 2500, 1200, '2026-10-01'),
(4, 'WH-NORTH-01', 3200, 600, 2600, 1500, 700, '2026-10-01'),
(5, 'WH-CENTRAL-01', 8500, 1500, 7000, 3000, 1500, '2026-10-01'),
(6, 'WH-SOUTH-02', 15000, 2000, 13000, 5000, 2500, '2026-10-01'),
(7, 'WH-CENTRAL-01', 22000, 3000, 19000, 8000, 4000, '2026-10-01'),
(8, 'WH-NORTH-01', 1200, 400, 800, 700, 350, '2026-10-01')
ON CONFLICT (product_id, warehouse_code, inventory_date) DO NOTHING;

-- 4. SEED SUPPLY EVENTS (20+ Events spanning operations)
INSERT INTO supply_events (id, event_id, product_id, supplier_id, event_date, demand_units, supplier_capacity_units, planned_lead_time_days, actual_lead_time_days, delay_occurred, delay_days, transport_cost, unit_cost, secondary_supplier_premium, air_freight_cost, service_risk_score) VALUES
(1, 'EVT-2026-001', 1, 1, '2026-09-15', 3000, 15000, 14.0, 16.0, true, 2.0, 4500.00, 125.50, 12550.00, 15000.00, 0.82),
(2, 'EVT-2026-002', 2, 2, '2026-09-16', 5000, 25000, 18.0, 18.0, false, 0.0, 6200.00, 85.00, 4250.00, 21000.00, 0.35),
(3, 'EVT-2026-003', 3, 3, '2026-09-18', 2000, 8000, 7.0, 7.0, false, 0.0, 1800.00, 34.20, 2000.00, 8500.00, 0.12),
(4, 'EVT-2026-004', 4, 4, '2026-09-20', 1500, 12000, 10.0, 11.0, true, 1.0, 2100.00, 48.75, 3600.00, 9200.00, 0.28),
(5, 'EVT-2026-005', 5, 5, '2026-09-22', 4000, 20000, 5.0, 5.0, false, 0.0, 1400.00, 18.50, 1800.00, 6000.00, 0.15),
(6, 'EVT-2026-006', 6, 6, '2026-09-25', 10000, 30000, 15.0, 20.0, true, 5.0, 3200.00, 4.80, 2400.00, 11000.00, 0.65),
(7, 'EVT-2026-007', 7, 7, '2026-09-26', 15000, 50000, 6.0, 6.0, false, 0.0, 2200.00, 12.30, 3000.00, 7500.00, 0.08),
(8, 'EVT-2026-008', 8, 8, '2026-09-28', 800, 6000, 21.0, 33.0, true, 12.0, 5800.00, 210.00, 16800.00, 28000.00, 0.92),
(9, 'EVT-2026-009', 1, 1, '2026-09-29', 2500, 15000, 14.0, 15.0, true, 1.0, 4200.00, 125.50, 10000.00, 14000.00, 0.74),
(10, 'EVT-2026-010', 2, 2, '2026-10-01', 4500, 25000, 18.0, 18.0, false, 0.0, 5900.00, 85.00, 3800.00, 19500.00, 0.25),
(11, 'EVT-2026-011', 3, 5, '2026-10-02', 3000, 20000, 5.0, 5.0, false, 0.0, 1900.00, 34.20, 2500.00, 9000.00, 0.10),
(12, 'EVT-2026-012', 4, 4, '2026-10-03', 2200, 12000, 10.0, 10.0, false, 0.0, 2600.00, 48.75, 4200.00, 10500.00, 0.18),
(13, 'EVT-2026-013', 5, 5, '2026-10-04', 6000, 20000, 5.0, 6.0, true, 1.0, 1700.00, 18.50, 2200.00, 7200.00, 0.32),
(14, 'EVT-2026-014', 6, 6, '2026-10-05', 8000, 30000, 15.0, 16.0, true, 1.0, 2900.00, 4.80, 1900.00, 9500.00, 0.44),
(15, 'EVT-2026-015', 7, 7, '2026-10-05', 12000, 50000, 6.0, 6.0, false, 0.0, 1950.00, 12.30, 2400.00, 6800.00, 0.06),
(16, 'EVT-2026-016', 8, 3, '2026-10-06', 600, 8000, 7.0, 7.0, false, 0.0, 3100.00, 210.00, 6300.00, 18000.00, 0.14),
(17, 'EVT-2026-017', 1, 1, '2026-10-06', 3500, 15000, 14.0, NULL, false, 0.0, 4800.00, 125.50, 14000.00, 16500.00, 0.85),
(18, 'EVT-2026-018', 2, 2, '2026-10-07', 5500, 25000, 18.0, NULL, false, 0.0, 6700.00, 85.00, 4700.00, 23000.00, 0.40),
(19, 'EVT-2026-019', 3, 3, '2026-10-07', 2500, 8000, 7.0, NULL, false, 0.0, 2050.00, 34.20, 2200.00, 9200.00, 0.11),
(20, 'EVT-2026-020', 8, 8, '2026-10-07', 950, 6000, 21.0, NULL, false, 0.0, 6400.00, 210.00, 19950.00, 32000.00, 0.89)
ON CONFLICT (event_id) DO NOTHING;

-- 5. SEED PREDICTIONS (Demo scenario + Operational events)
INSERT INTO predictions (id, event_id, model_name, model_version, prediction_type, delay_probability, predicted_delay_days, risk_score, risk_level, prediction_timestamp) VALUES
(1, 1, 'XGBoost Delay Classifier', 'v1.0', 'DELAY', 0.87, 14.0, 0.82, 'HIGH', '2026-09-15 08:30:00+00'),
(2, 6, 'XGBoost Delay Classifier', 'v1.0', 'DELAY', 0.65, 5.0, 0.60, 'MEDIUM', '2026-09-25 09:00:00+00'),
(3, 8, 'XGBoost Delay Classifier', 'v1.0', 'DELAY', 0.93, 12.0, 0.91, 'CRITICAL', '2026-09-28 10:15:00+00'),
(4, 17, 'XGBoost Delay Classifier', 'v1.0', 'DELAY', 0.85, 14.0, 0.80, 'HIGH', '2026-10-06 14:00:00+00'),
(5, 20, 'XGBoost Delay Classifier', 'v1.0', 'DELAY', 0.89, 10.0, 0.88, 'CRITICAL', '2026-10-07 11:30:00+00')
ON CONFLICT (id) DO NOTHING;

-- 6. SEED OPTIMIZATION RUNS
INSERT INTO optimization_runs (id, event_id, prediction_id, optimization_run_id, objective_type, optimization_status, total_cost_baseline, recommended_cost, service_level_target, risk_weight, cost_weight, time_weight, solver_name, solver_status, execution_time_ms, created_at) VALUES
(1, 1, 1, 'OPT-RUN-2026-001', 'BALANCED', 'COMPLETED', 381000.00, 396000.00, 0.98, 0.40, 0.40, 0.20, 'PULP_CBC', 'OPTIMAL', 142.3, '2026-09-15 09:00:00+00'),
(2, 8, 3, 'OPT-RUN-2026-008', 'MIN_RISK', 'COMPLETED', 173800.00, 201800.00, 0.99, 0.60, 0.20, 0.20, 'PULP_CBC', 'OPTIMAL', 115.8, '2026-09-28 11:00:00+00')
ON CONFLICT (optimization_run_id) DO NOTHING;

-- 7. SEED RECOMMENDATIONS (Including ALT_A, ALT_B, ALT_C for Demo Scenario)
INSERT INTO recommendations (id, optimization_run_id, recommendation_code, strategy_type, strategy_description, estimated_cost, estimated_delay_days, estimated_risk, service_level, cost_difference, risk_difference, rank, is_recommended, created_at) VALUES
(1, 1, 'ALT_A', 'AIR_FREIGHT', 'Expedite primary supplier shipment via priority air freight charter', 15000.00, 2.0, 0.08, 0.99, 15000.00, -0.74, 1, true, '2026-09-15 09:01:00+00'),
(2, 1, 'ALT_B', 'SECONDARY_SUPPLIER', 'Split procurement 60/40 with secondary qualified domestic supplier at 10% premium', 22000.00, 4.0, 0.15, 0.95, 22000.00, -0.67, 2, false, '2026-09-15 09:01:00+00'),
(3, 1, 'ALT_C', 'DELAY_LAUNCH', 'Reschedule assembly launch timeline by 14 days without logistics expediting spend', 0.00, 14.0, 0.85, 0.70, 0.00, 0.03, 3, false, '2026-09-15 09:01:00+00'),
(4, 2, 'ALT_A', 'SECONDARY_SUPPLIER', 'Emergency redirect of 800 motor units to Japanese precision supplier', 28000.00, 3.0, 0.10, 0.98, 28000.00, -0.81, 1, true, '2026-09-28 11:02:00+00'),
(5, 2, 'ALT_B', 'AIR_FREIGHT', 'Air freight remaining stock from Canadian central distribution center', 32000.00, 2.0, 0.12, 0.97, 32000.00, -0.79, 2, false, '2026-09-28 11:02:00+00')
ON CONFLICT (optimization_run_id, recommendation_code) DO NOTHING;

-- 8. SEED DECISIONS (Manager selects ALT_A for Microchip X1)
INSERT INTO decisions (id, decision_id, optimization_run_id, recommendation_id, decision_maker, decision_status, decision_reason, selected_at, execution_status, executed_at, created_at) VALUES
(1, 'DEC-2026-001', 1, 1, 'Sarah Chen (VP Global Supply Chain)', 'APPROVED', 'Selected Air Freight (ALT_A) to prevent factory line shutdown on Tier-1 automotive customer order.', '2026-09-15 10:30:00+00', 'COMPLETED', '2026-09-15 11:00:00+00', '2026-09-15 10:30:00+00'),
(2, 'DEC-2026-008', 2, 4, 'Marcus Vance (Procurement Lead)', 'APPROVED', 'Approved secondary supplier redirect to avoid 12-day actuator bottleneck.', '2026-09-28 13:00:00+00', 'COMPLETED', '2026-09-28 13:30:00+00', '2026-09-28 13:00:00+00')
ON CONFLICT (decision_id) DO NOTHING;

-- 9. SEED OUTCOMES (Actual Air Freight Cost = $18,000, Actual Delay = 2 days)
INSERT INTO outcomes (id, decision_id, actual_cost, actual_delay_days, actual_service_level, actual_risk, actual_quantity_delivered, cost_variance, delay_variance, service_level_variance, outcome_status, evaluation_date, notes, created_at) VALUES
(1, 1, 18000.00, 2.0, 0.99, 0.05, 3000, 3000.00, 0.0, 0.0, 'SUCCESS', '2026-09-18', 'Air freight shipment landed at regional airport on schedule; unexpected peak jet fuel surcharge added $3,000 to freight baseline.', '2026-09-18 16:00:00+00'),
(2, 2, 29500.00, 3.5, 0.97, 0.08, 800, 1500.00, 0.5, -0.01, 'SUCCESS', '2026-10-02', 'Secondary supplier met delivery requirements with minor handling charge increase.', '2026-10-02 17:00:00+00')
ON CONFLICT (decision_id) DO NOTHING;

-- 10. SEED FEEDBACK (Cost variance = +$3,000 closes learning loop)
INSERT INTO feedback (id, outcome_id, optimization_run_id, feedback_type, cost_error, delay_error, risk_error, prediction_error, weight_adjustment, learning_signal, created_at) VALUES
(1, 1, 1, 'COST_VARIANCE', 3000.00, 0.0, -0.03, 0.0, 0.05, 'Incorporate +20% jet fuel seasonal rate multiplier into future Air Freight cost parameter models for APAC corridors.', '2026-09-19 09:00:00+00'),
(2, 2, 2, 'COST_VARIANCE', 1500.00, 0.5, -0.02, 0.0, 0.02, 'Update handling fee calibration for Kyoto precision parts expedited customs clearance.', '2026-10-03 10:00:00+00')
ON CONFLICT (id) DO NOTHING;

-- Reset Sequences for PostgreSQL serial columns
SELECT setval('products_id_seq', (SELECT MAX(id) FROM products));
SELECT setval('suppliers_id_seq', (SELECT MAX(id) FROM suppliers));
SELECT setval('inventory_id_seq', (SELECT MAX(id) FROM inventory));
SELECT setval('supply_events_id_seq', (SELECT MAX(id) FROM supply_events));
SELECT setval('predictions_id_seq', (SELECT MAX(id) FROM predictions));
SELECT setval('optimization_runs_id_seq', (SELECT MAX(id) FROM optimization_runs));
SELECT setval('recommendations_id_seq', (SELECT MAX(id) FROM recommendations));
SELECT setval('decisions_id_seq', (SELECT MAX(id) FROM decisions));
SELECT setval('outcomes_id_seq', (SELECT MAX(id) FROM outcomes));
SELECT setval('feedback_id_seq', (SELECT MAX(id) FROM feedback));
