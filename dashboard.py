import json
from pathlib import Path

import requests
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SupplyPrescript Dashboard",
    page_icon="📦",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def fetch_prediction_history():
    try:
        response = requests.get(
            f"{API_URL}/predictions/history",
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("predictions", []), None

    except requests.RequestException as exc:
        return [], str(exc)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📦 SupplyPrescript")
st.subheader("Supply Chain Delay Prediction Dashboard")

st.write(
    "Predict shipment delays, review risk probabilities, "
    "and view prediction records saved in PostgreSQL."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard Overview",
        "Predict Shipment Delay",
        "Prediction History",
        "Prescriptive Recommendations",
    ],
)


# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    records, error = fetch_prediction_history()

    if error:
        st.error(
            "Cannot connect to the API. "
            "Start the FastAPI server in another terminal."
        )
        st.caption(f"Technical details: {error}")

    else:
        total_records = len(records)

        delayed_records = sum(
            1
            for record in records
            if record.get("prediction") == "DELAY"
        )

        on_time_records = sum(
            1
            for record in records
            if record.get("prediction") == "NO DELAY"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction Records",
            total_records,
        )

        col2.metric(
            "Predicted Delays",
            delayed_records,
        )

        col3.metric(
            "Predicted No Delays",
            on_time_records,
        )

        st.divider()

        st.subheader("Recent Predictions")

        if records:
            df = pd.DataFrame(records)

            preferred_columns = [
                "id",
                "supplier_id",
                "product_id",
                "category",
                "prediction",
                "delay_probability",
                "created_at",
            ]

            available_columns = [
                column
                for column in preferred_columns
                if column in df.columns
            ]

            st.dataframe(
                df[available_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The API returns up to 100 recent prediction records."
            )

        else:
            st.info("No prediction records are available yet.")


# --------------------------------------------------
# PREDICT SHIPMENT DELAY
# --------------------------------------------------

elif page == "Predict Shipment Delay":

    st.header("Predict Shipment Delay")

    st.write(
        "Enter the shipment details below. "
        "The prediction will be sent to the trained ML model "
        "through FastAPI and saved in PostgreSQL."
    )

    with st.form("shipment_prediction_form"):

        st.subheader("Shipment Information")

        col1, col2 = st.columns(2)

        with col1:
            supplier_id = st.text_input(
                "Supplier ID",
                value="SUP001",
            )

            supplier_country = st.text_input(
                "Supplier Country",
                value="India",
            )

            product_id = st.text_input(
                "Product ID",
                value="PROD001",
            )

            category = st.selectbox(
                "Product Category",
                [
                    "Electronics",
                    "Clothing",
                    "Food",
                    "Automotive",
                    "Healthcare",
                    "Other",
                ],
            )

        with col2:
            demand_units = st.number_input(
                "Demand Units",
                min_value=0.0,
                value=100.0,
            )

            inventory_units = st.number_input(
                "Inventory Units",
                min_value=0.0,
                value=50.0,
            )

            supplier_capacity_units = st.number_input(
                "Supplier Capacity Units",
                min_value=0.0,
                value=200.0,
            )

            supplier_reliability = st.number_input(
                "Supplier Reliability",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
            )

            planned_lead_time_days = st.number_input(
                "Planned Lead Time (Days)",
                min_value=0.0,
                value=7.0,
            )

        st.subheader("Cost Information")

        col3, col4 = st.columns(2)

        with col3:
            unit_cost_usd = st.number_input(
                "Unit Cost (USD)",
                min_value=0.0,
                value=10.0,
            )

            transport_cost_usd = st.number_input(
                "Transport Cost (USD)",
                min_value=0.0,
                value=5.0,
            )

            secondary_supplier_premium_pct = st.number_input(
                "Secondary Supplier Premium (%)",
                min_value=0.0,
                value=10.0,
            )

            secondary_cost_difference = st.number_input(
                "Secondary Cost Difference",
                value=2.0,
            )

        with col4:
            air_freight_premium = st.number_input(
                "Air Freight Premium",
                min_value=0.0,
                value=15.0,
            )

            cost_per_demand_unit = st.number_input(
                "Cost per Demand Unit",
                min_value=0.0,
                value=10.0,
            )

        st.subheader("Inventory and Risk Information")

        col5, col6 = st.columns(2)

        with col5:
            inventory_coverage_ratio = st.number_input(
                "Inventory Coverage Ratio",
                min_value=0.0,
                value=0.5,
            )

            inventory_gap_units = st.number_input(
                "Inventory Gap Units",
                value=50.0,
            )

            capacity_utilization = st.number_input(
                "Capacity Utilization",
                min_value=0.0,
                value=0.7,
            )

            supplier_risk_score = st.number_input(
                "Supplier Risk Score",
                value=0.3,
            )

            demand_pressure = st.number_input(
                "Demand Pressure",
                value=0.5,
            )

        with col6:
            previous_delay_days = st.number_input(
                "Previous Delay Days",
                min_value=0.0,
                value=2.0,
            )

            previous_delay_flag = st.selectbox(
                "Previous Delay?",
                [0, 1],
            )

            shortage_flag = st.selectbox(
                "Shortage?",
                [0, 1],
                index=1,
            )

            rolling_average_delay = st.number_input(
                "Rolling Average Delay",
                min_value=0.0,
                value=1.0,
            )

            rolling_average_demand = st.number_input(
                "Rolling Average Demand",
                min_value=0.0,
                value=100.0,
            )

            rolling_average_inventory = st.number_input(
                "Rolling Average Inventory",
                min_value=0.0,
                value=50.0,
            )

        st.subheader("Outlier Flags")

        col7, col8 = st.columns(2)

        with col7:
            demand_outlier_flag = st.selectbox(
                "Demand Outlier",
                [0, 1],
            )

            inventory_outlier_flag = st.selectbox(
                "Inventory Outlier",
                [0, 1],
            )

        with col8:
            cost_outlier_flag = st.selectbox(
                "Cost Outlier",
                [0, 1],
            )

            lead_time_outlier_flag = st.selectbox(
                "Lead Time Outlier",
                [0, 1],
            )

        submitted = st.form_submit_button(
            "Predict Shipment Delay",
            use_container_width=True,
        )

    if submitted:

        if not supplier_id.strip() or not product_id.strip():
            st.error(
                "Supplier ID and Product ID are required."
            )

        else:
            shipment_data = {
                "demand_units": demand_units,
                "inventory_units": inventory_units,
                "supplier_capacity_units": supplier_capacity_units,
                "supplier_reliability": supplier_reliability,
                "planned_lead_time_days": planned_lead_time_days,
                "unit_cost_usd": unit_cost_usd,
                "transport_cost_usd": transport_cost_usd,
                "secondary_supplier_premium_pct": (
                    secondary_supplier_premium_pct
                ),
                "inventory_coverage_ratio": inventory_coverage_ratio,
                "inventory_gap_units": inventory_gap_units,
                "capacity_utilization": capacity_utilization,
                "cost_per_demand_unit": cost_per_demand_unit,
                "secondary_cost_difference": secondary_cost_difference,
                "air_freight_premium": air_freight_premium,
                "supplier_risk_score": supplier_risk_score,
                "demand_pressure": demand_pressure,
                "shortage_flag": shortage_flag,
                "previous_delay_days": previous_delay_days,
                "previous_delay_flag": previous_delay_flag,
                "rolling_average_delay": rolling_average_delay,
                "rolling_average_demand": rolling_average_demand,
                "rolling_average_inventory": rolling_average_inventory,
                "demand_outlier_flag": demand_outlier_flag,
                "inventory_outlier_flag": inventory_outlier_flag,
                "cost_outlier_flag": cost_outlier_flag,
                "lead_time_outlier_flag": lead_time_outlier_flag,
                "supplier_id": supplier_id.strip(),
                "supplier_country": supplier_country.strip(),
                "product_id": product_id.strip(),
                "category": category,
            }

            try:
                with st.spinner("Generating shipment prediction..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=shipment_data,
                        timeout=60,
                    )

                    response.raise_for_status()
                    result = response.json()

                st.success("Prediction completed successfully!")

                st.subheader("Prediction Result")

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Prediction",
                    result["prediction"],
                )

                result_col2.metric(
                    "Delay Probability",
                    f'{result["delay_probability_percent"]:.2f}%',
                )

                probability = result["delay_probability"]

                st.progress(
                    max(0.0, min(1.0, float(probability)))
                )

                st.write(
                    f'Database record ID: {result["prediction_id"]}'
                )

                if result.get("saved_to_database"):
                    st.success(
                        "Prediction saved to PostgreSQL."
                    )

                st.info(
                    "This is a model estimate, not a guarantee "
                    "of whether a shipment will be delayed."
                )

            except requests.RequestException as exc:
                st.error(
                    "Could not complete the prediction. "
                    "Make sure the FastAPI server is running."
                )

                if exc.response is not None:
                    st.code(
                        f"HTTP {exc.response.status_code}\n"
                        f"{exc.response.text}"
                    )
                else:
                    st.caption(str(exc))


# --------------------------------------------------
# PREDICTION HISTORY PAGE
# --------------------------------------------------

elif page == "Prediction History":

    st.header("Prediction History")

    if st.button("Refresh History"):
        st.rerun()

    records, error = fetch_prediction_history()

    if error:
        st.error(
            "Unable to retrieve history. "
            "Make sure the FastAPI server is running."
        )
        st.caption(f"Technical details: {error}")

    elif records:
        df = pd.DataFrame(records)

        st.metric(
            "Records Returned",
            len(df),
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="supply_prescript_prediction_history.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No prediction history is available yet. "
            "Submit a shipment prediction first."
        )


# --------------------------------------------------
# PRESCRIPTIVE RECOMMENDATIONS PAGE
# --------------------------------------------------

elif page == "Prescriptive Recommendations":

    st.header("Prescriptive Recommendations")

    st.write(
        "Compare supply-chain actions based on estimated cost, "
        "delivery time, and business constraints."
    )

    recommendation_path = (
        Path(__file__).resolve().parent
        / "reports"
        / "prescriptive_recommendations.json"
    )

    if not recommendation_path.exists():
        st.warning(
            "Recommendations file not found. "
            "Run the prescriptive solver first."
        )

    else:
        try:
            recommendation_data = json.loads(
                recommendation_path.read_text(encoding="utf-8")
            )

            budget = recommendation_data.get("budget", 0)

            max_delay = recommendation_data.get(
                "max_acceptable_delay_days", 0
            )

            alternatives = recommendation_data.get(
                "alternatives", []
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Available Budget",
                f"${budget:,.2f}",
            )

            col2.metric(
                "Maximum Acceptable Delay",
                f"{max_delay:.2f} days",
            )

            st.divider()

            st.subheader("Compare Alternative Actions")

            if alternatives:

                cards = st.columns(3)

                for card, option in zip(cards, alternatives):

                    with card:

                        st.markdown(
                            f"### {option['action']}"
                        )

                        if option.get("recommended", False):
                            st.success("⭐ RECOMMENDED")

                        elif option.get("feasible", False):
                            st.info("Feasible Alternative")

                        else:
                            st.error("Not Feasible")

                        cost = float(
                            option.get("cost", 0)
                        )

                        delay = float(
                            option.get(
                                "delay_days",
                                option.get("delay", 0),
                            )
                        )

                        st.metric(
                            "Estimated Cost",
                            f"${cost:,.2f}",
                        )

                        st.metric(
                            "Estimated Delay",
                            f"{delay:.2f} days",
                        )

                        st.caption(
                            option.get("reason", "")
                        )

                        st.caption(
                            "Days saved vs regular supply: "
                            f"{option.get('days_saved_vs_regular', 0):.2f}"
                        )

                        cost_saving = float(
                            option.get(
                                "cost_saving_vs_air_freight",
                                0,
                            )
                        )

                        st.caption(
                            "Cost difference vs air freight: "
                            f"${-cost_saving:,.2f}"
                        )

            else:
                st.warning(
                    "No alternative actions are available."
                )

            st.divider()

            if recommendation_data.get("status") == "success":

                st.success(
                    "Recommended Action: "
                    f"{recommendation_data['recommended_action']}"
                )

            else:

                st.error(
                    recommendation_data.get(
                        "message",
                        "No feasible action was found.",
                    )
                )

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            st.error(
                "Unable to load recommendation results."
            )

            st.caption(str(exc))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SupplyPrescript | Machine Learning + FastAPI + PostgreSQL"
)