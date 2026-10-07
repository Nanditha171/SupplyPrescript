import joblib
import pandas as pd

MODEL_PATH = "models/supply_delay_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_delay(shipment_data):

    df = pd.DataFrame([shipment_data])

    # Remove columns that were not used during training
    df = df.drop(
        columns=["shipment_id", "delay", "delay_days"],
        errors="ignore"
    )

    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    result = {
        "prediction": "DELAY" if prediction == 1 else "NO DELAY",
        "delay_probability": round(float(probability), 4)
    }

    return result


if __name__ == "__main__":

    sample_shipment = {
        "supplier": "Supplier_1",
        "product_category": "Electronics",
        "origin": "India",
        "destination": "USA",
        "order_quantity": 500,
        "inventory_level": 200,
        "lead_time": 15,
        "historical_lead_time": 12,
        "shipping_mode": "Air",
        "distance_km": 1200,
        "transportation_cost": 5000,
        "previous_delays": 2,
        "weather_condition": "Clear",
        "demand_level": "High",
        "supplier_reliability": 0.85,
        "warehouse_capacity": 5000,
        "fuel_price": 3.5,
        "customs_delay": 1,
        "priority": "High"
    }

    result = predict_delay(sample_shipment)

    print("\nSHIPMENT DELAY PREDICTION")
    print("-------------------------")
    print("Prediction:", result["prediction"])
    print("Delay Probability:", result["delay_probability"])