# Supply Prescript — REST API Documentation
**Base URL:** `http://127.0.0.1:8000/api`  
**Interactive Swagger UI:** `http://127.0.0.1:8000/docs`  
**ReDoc Specification:** `http://127.0.0.1:8000/redoc`  

---

## 1. Authentication & Protocols
All endpoints consume and produce `application/json`. Errors return standard HTTP status codes (400, 404, 422, 500) with descriptive JSON envelopes.

---

## 2. API Endpoint Directory

### 🏥 Health & Diagnostics
- `GET /api/health`
  - Returns service status and database connectivity.
  - Example Response:
    ```json
    {
      "status": "healthy",
      "database": {"status": "connected"},
      "version": "1.0.0"
    }
    ```

### 📦 Products Catalog
- `GET /api/products`: List managed SKUs.
- `GET /api/products/{product_id}`: Retrieve single product details.
- `POST /api/products`: Create a new product.
- `PUT /api/products/{product_id}`: Update product specifications.

### 🏭 Suppliers Directory
- `GET /api/suppliers`: List all approved primary and secondary suppliers.
- `GET /api/suppliers/{supplier_id}`: Retrieve single supplier profile.
- `POST /api/suppliers`: Register a new supplier.
- `PUT /api/suppliers/{supplier_id}`: Update supplier capacity or reliability.

### 🏢 Warehouse Inventory
- `GET /api/inventory`: List inventory stock snapshots across warehouses.
- `GET /api/inventory/{product_id}`: Retrieve inventory for a specific SKU.
- `POST /api/inventory`: Record a new warehouse inventory snapshot.
- `PUT /api/inventory/{inventory_id}`: Adjust stock counts or reservations.

### 🚚 Shipments & Supply Events
- `GET /api/shipments`: List operational shipment orders.
- `GET /api/shipments/{event_id}`: Retrieve shipment by event_id (e.g. `EVT-001`).
- `POST /api/shipments`: Register a new replenishment order.

### 🔮 Delay Predictions
- `POST /api/predictions/{event_id}`: Trigger ML inference for an event.
- `GET /api/predictions/{event_id}`: Retrieve existing delay prediction.
- `GET /api/predictions`: List historical prediction runs.

### ⚠️ Operational Risk Assessment
- `POST /api/risks/{event_id}`: Compute 5-dimensional operational risk scores.
- `GET /api/risks/{event_id}`: Retrieve risk assessment by event ID.
- `GET /api/risks`: List risk assessments.

### ⚙️ Prescriptive Optimization
- `POST /api/optimization/{event_id}?objective_type=MIN_COST`: Solve recovery allocation with PuLP CBC.
- `GET /api/optimization/{run_id}`: Retrieve optimization run by run ID.
- `GET /api/optimization`: List optimization runs.

### ⚖️ Recommendations
- `GET /api/recommendations/{run_id}`: Retrieve ranked alternatives, cost comparisons, and explanations.

### 🎯 Decisions & Governance
- `POST /api/decisions`: Record manager approval or override with reasoning.
- `GET /api/decisions`: List all operational decisions.
- `GET /api/decisions/{decision_id}`: Retrieve decision by ID.
- `PUT /api/decisions/{decision_id}`: Update decision state or execution progress (`NOT_STARTED` -> `IN_PROGRESS` -> `COMPLETED`).

### 📈 Post-Execution Outcomes
- `POST /api/outcomes`: Record actual post-execution performance and calculate variances.
- `GET /api/outcomes`: List all evaluated outcomes.
- `GET /api/outcomes/{decision_id}`: Retrieve outcome evaluation for a decision.

### 🔄 Closed-Loop Feedback & Learning
- `POST /api/feedback/{outcome_id}`: Process outcome into feedback and calibrate cost multipliers.
- `GET /api/feedback`: List historical feedback signals and weight adjustments.
- `GET /api/feedback/metrics`: Retrieve strategy-wise accuracy metrics.

