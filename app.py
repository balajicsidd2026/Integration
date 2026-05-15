import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from catboost import Pool
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cargo Operations",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    cargo_model = joblib.load(
        "cargo_rejection_catboost_model.pkl"
    )

    dg_model = joblib.load(
        "dg_incident_model.pkl"
    )

    return cargo_model, dg_model

cargo_model, dg_model = load_models()

@st.cache_data
def load_reference_data():
    return pd.read_csv("dg_incident_test.csv")

reference_df = load_reference_data()

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "task1"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 96%;
}

header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

.main-title {
    font-size: 58px;
    font-weight: 800;
    color: #0b1f5e;
    margin: 0;
    line-height: 1.1;
}

.main-heading {
    font-size: 42px;
    font-weight: 700;
    color: #0b1f5e;
    margin-bottom: 4px;
}

.sub-heading {
    font-size: 17px;
    color: #5d6472;
    margin-bottom: 10px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #0b1f5e;
    margin-bottom: 10px;
}

div.stButton > button {
    width: 100%;
    height: 82px;
    border-radius: 18px;
    border: none;
    font-size: 19px;
    font-weight: 600;
    background-color:#0b1f5e;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}

div.stButton > button:hover {
    background-color:#0b1f6e;
    transform: translateY(-2px);
    color: white;
}

div[data-baseweb="select"] > div,
.stNumberInput > div > div > input {
    border-radius: 12px !important;
}

.stSlider {
    padding-top: 0.3rem;
}

hr {
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

[data-testid="stMetricValue"] {
    font-size: 34px;
}

.js-plotly-plot .plotly .modebar {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns(
    [1, 1],
    gap="large"
)

# =====================================================
# LEFT SIDE
# =====================================================

with header_left:

    st.markdown("""
    <div style="
        height:82px;
        display:flex;
        align-items:center;
    ">
        <div class='main-title'>
            Cargo Operations
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# RIGHT SIDE
# =====================================================

with header_right:

    nav1, nav2 = st.columns(
        2,
        gap="small"
    )

    with nav1:

        if st.button(
            "Cargo Acceptance Rejection Forecasting",
            use_container_width=True
        ):
            st.session_state.page = "task1"

    with nav2:

        if st.button(
            "Dangerous Goods Incident Prediction",
            use_container_width=True
        ):
            st.session_state.page = "task2"

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# TASK 1
# =========================================================

if st.session_state.page == "task1":

    st.markdown("""
    <div class='main-heading'>
    Cargo Acceptance Rejection Forecasting
    </div>

    <div class='sub-heading'>
    Predict cargo shipment acceptance or rejection
    using Machine Learning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # 3 COLUMN LAYOUT
    # =====================================================

    col1, col2, col3 = st.columns(3)

    # =====================================================
    # COLUMN 1
    # =====================================================

    with col1:

        shipment_date = st.date_input(
            "Shipment Date"
        )

        airline = st.selectbox(
            "Airline",
            ['Emirates', 'Qatar', 'Lufthansa', 'Air India']
        )

        SHC = st.selectbox(
            "SHC",
            ['DGR', 'PER', 'AVI', 'VAL', 'PIL', 'COL', 'HUM']
        )

        origin = st.selectbox(
            "Origin",
            ['HAM', 'DXB', 'MAA', 'OSL', 'AMS', 'JFK', 'MXP', 'FRA', 'PVG']
        )

        destination = st.selectbox(
            "Destination",
            ['JED']
        )

        packaging_condition = st.selectbox(
            "Packaging Condition",
            ['Good', 'Average', 'Damaged']
        )

    # =====================================================
    # COLUMN 2
    # =====================================================

    with col2:

        packaging_seal_status = st.selectbox(
            "Packaging Seal Status",
            ['Intact', 'Tampered', 'Broken', 'Missing']
        )

        shipper_company_name = st.selectbox(
            "Shipper Company",
            ['DB Schenker', 'FedEx', 'DHL', 'Maersk']
        )

        consignee_name = st.selectbox(
            "Consignee Name",
            ['Gulf Fresh Foods', 'Al Jazeera Trading','Blue Dart',]
        )

        shipper_type = st.selectbox(
            "Shipper Type",
            ['Corporate', 'Retail', 'Agent']
        )

        documentation_status = st.selectbox(
            "Documentation Status",
            ['Complete', 'Pending', 'Missing']
        )

        security_screening_status = st.selectbox(
            "Security Screening Status",
            ['Cleared', 'Pending', 'Failed']
        )

    # =====================================================
    # COLUMN 3
    # =====================================================

    with col3:

        xray_scan_result = st.selectbox(
            "X-Ray Scan Result",
            ['Clear', 'Suspicious']
        )

        shipment_priority = st.selectbox(
            "Shipment Priority",
            ['Normal', 'Express', 'VIP']
        )

        cargo_weight_kg = st.number_input(
            "Cargo Weight (kg)",
            min_value=0.0,
            value=132.00
        )

        damage_history_count = st.number_input(
            "Damage History Count",
            min_value=0,
            value=0
        )

        compliance_violation_count = st.number_input(
            "Compliance Violation Count",
            min_value=0,
            value=0
        )

        shipper_reliability_score = st.number_input(
            "Shipper Reliability Score",
            min_value=0,
            max_value=100,
            value=80
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "Predict Cargo Status",
        use_container_width=True
    ):

        month = shipment_date.month
        day = shipment_date.day
        day_of_week = shipment_date.weekday()

        is_weekend = 1 if day_of_week in [5,6] else 0

        input_data = pd.DataFrame({

            'airline': [airline],
            'SHC': [SHC],
            'origin': [origin],
            'destination': [destination],
            'packaging_condition': [packaging_condition],
            'packaging_seal_status': [packaging_seal_status],
            'shipper_company_name': [shipper_company_name],
            'consignee_name': [consignee_name],
            'shipper_type': [shipper_type],
            'documentation_status': [documentation_status],
            'security_screening_status': [security_screening_status],
            'xray_scan_result': [xray_scan_result],
            'shipment_priority': [shipment_priority],
            'cargo_weight_kg': [cargo_weight_kg],
            'damage_history_count': [damage_history_count],
            'compliance_violation_count': [compliance_violation_count],
            'shipper_reliability_score': [shipper_reliability_score],
            'month': [month],
            'day': [day],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend]
        })

        prediction_prob = cargo_model.predict_proba(
            input_data
        )[:,1][0]

        st.markdown("<hr>", unsafe_allow_html=True)

        if prediction_prob > 0.65:
            st.error("Prediction Result: Rejected Cargo")
        else:
            st.success("Prediction Result: Accepted Cargo")

        st.progress(float(prediction_prob))
        
        st.metric(
            "Probability of Rejection",
            f"{prediction_prob:.2%}"
        )


        # =================================================
        # FEATURE ANALYSIS
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='section-title'>
        Feature Contribution Analysis
        </div>
        """, unsafe_allow_html=True)

        explainer = shap.TreeExplainer(cargo_model)

        shap_values = explainer.shap_values(input_data)

        feature_impact = np.abs(shap_values[0])

        importance_df = pd.DataFrame({

            "Feature": input_data.columns,
            "Importance": feature_impact

        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=True
        ).tail(10)

        fig = go.Figure()

        fig.add_trace(go.Bar(

            x=importance_df["Importance"],

            y=importance_df["Feature"],

            orientation='h',

            text=[
                f"{round(x,2)}"
                for x in importance_df["Importance"]
            ],

            textposition='outside',

            marker=dict(
                color='#0b66c3'
            )
        ))

        fig.update_layout(

            height=550,

            plot_bgcolor='#f5f7fb',

            paper_bgcolor='#f5f7fb',

            margin=dict(
                l=140,
                r=40,
                t=20,
                b=20
            ),

            xaxis=dict(
                visible=False
            ),

            yaxis=dict(
                title=''
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

# =========================================================
# TASK 2
# =========================================================

elif st.session_state.page == "task2":

    st.markdown("""
    <div class='main-heading'>
    Dangerous Goods Incident Prediction
    </div>

    <div class='sub-heading'>
    AI-powered cargo safety risk assessment system
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):

        # =================================================
        # 3 COLUMN PERFECT ALIGNMENT
        # =================================================

        col1, col2, col3 = st.columns(3)

        # =================================================
        # COLUMN 1
        # =================================================

        with col1:

            shc_code = st.selectbox(
                "SHC Code",
                sorted(
                    reference_df["shc_code"]
                    .astype(str)
                    .unique()
                )
            )

            origin_destination = st.selectbox(
                "Route",
                sorted(
                    reference_df["origin_destination"]
                    .astype(str)
                    .unique()
                )
            )

            dg_class = st.selectbox(
                "DG Class",
                sorted(
                    reference_df["dg_class"]
                    .astype(float)
                    .unique()
                )
            )

            packaging_type = st.selectbox(
                "Packaging Type",
                sorted(
                    reference_df["packaging_type"]
                    .astype(str)
                    .unique()
                )
            )

            weather_condition = st.selectbox(
                "Weather Condition",
                sorted(
                    reference_df["weather_condition"]
                    .astype(str)
                    .unique()
                )
            )

        # =================================================
        # COLUMN 2
        # =================================================

        with col2:

            cargo_weight_kg = st.number_input(
                "Cargo Weight (kg)",
                min_value=0.0,
                value=15000.0
            )

            temperature_celsius = st.number_input(
                "Temperature (°C)",
                value=25.0
            )

            humidity_percentage = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0
            )

            shipment_hour = st.slider(
                "Shipment Hour",
                0,
                23,
                12
            )

            shipment_day_of_week = st.selectbox(
                "Shipment Day",
                options=[0,1,2,3,4,5,6],
                format_func=lambda x:
                ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
            )

        # =================================================
        # COLUMN 3
        # =================================================

        with col3:

            shipment_month = st.slider(
                "Shipment Month",
                1,
                12,
                5
            )

            handling_error_count = st.number_input(
                "Handling Error Count",
                min_value=0,
                value=0
            )

            previous_incident_count = st.number_input(
                "Previous Incident Count",
                min_value=0,
                value=0
            )

            safety_staff_count = st.number_input(
                "Safety Staff Count",
                min_value=0,
                value=10
            )

            doc_audit_result = st.radio(
                "Documentation Audit",
                options=[1,0],
                format_func=lambda x:
                "Pass" if x == 1 else "Fail"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submit = st.form_submit_button(
            "Run Risk Assessment",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submit:

        is_critical_mishandling = 1 if (
            int(handling_error_count) > 7
        ) else 0

        is_untrusted_shipper = 1 if (
            int(previous_incident_count) > 5
        ) else 0

        climate_shock_risk = 1 if (
            str(origin_destination) == "OSL-JED"
            and float(temperature_celsius) > 30
        ) else 0

        gas_handling_risk = 1 if (
            round(float(dg_class),1) == 2.1
            and int(handling_error_count) > 5
        ) else 0

        thermal_expansion_risk = 1 if (
            str(origin_destination) == "OSL-JED"
            and float(temperature_celsius) > 35
            and round(float(dg_class),1) == 3.0
        ) else 0

        shipper_hazard_combo = 1 if (
            int(previous_incident_count) > 5
            and str(shc_code) == "CAO"
        ) else 0

        input_dict = {

            "shc_code": [str(shc_code)],
            "origin_destination": [str(origin_destination)],
            "dg_class": [float(dg_class)],
            "packaging_type": [str(packaging_type)],
            "handling_error_count": [int(handling_error_count)],
            "previous_incident_count": [int(previous_incident_count)],
            "cargo_weight_kg": [float(cargo_weight_kg)],
            "temperature_celsius": [float(temperature_celsius)],
            "humidity_percentage": [float(humidity_percentage)],
            "weather_condition": [str(weather_condition)],
            "safety_staff_count": [int(safety_staff_count)],
            "doc_audit_result": [int(doc_audit_result)],
            "shipment_hour": [int(shipment_hour)],
            "shipment_day_of_week": [int(shipment_day_of_week)],
            "shipment_month": [int(shipment_month)],
            "is_critical_mishandling": [int(is_critical_mishandling)],
            "is_untrusted_shipper": [int(is_untrusted_shipper)],
            "climate_shock_risk": [int(climate_shock_risk)],
            "gas_handling_risk": [int(gas_handling_risk)],
            "thermal_expansion_risk": [int(thermal_expansion_risk)],
            "shipper_hazard_combo": [int(shipper_hazard_combo)]
        }

        input_data = pd.DataFrame(input_dict)

        prediction_pool = Pool(

            data=input_data,

            cat_features=[
                "shc_code",
                "origin_destination",
                "packaging_type",
                "weather_condition"
            ]
        )

        raw_prediction = dg_model.predict(
            prediction_pool
        )[0]

        risk_score = float(
            np.clip(raw_prediction, 0.0, 1.0)
        )

        if risk_score >= 0.75:
            risk_level = "HIGH"

        elif risk_score >= 0.45:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        st.markdown("<hr>", unsafe_allow_html=True)

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Risk Score",
                f"{risk_score:.1%}"
            )

        with metric2:
            st.metric(
                "Risk Level",
                risk_level
            )

        # with metric3:
        #     st.metric(
        #         "DG Class",
        #         dg_class
        #     )

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # PERFECTLY ALIGNED RESULT SECTION
        # =================================================

        result1, result2, result3 = st.columns(
            [1,1.2,1]
        )

        # =================================================
        # RISK GAUGE
        # =================================================

        with result1:

            st.markdown("""
            <div class='section-title'>
            Overall Risk Magnitude
            </div>
            """, unsafe_allow_html=True)

            gauge_fig = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=risk_score * 100,

                    number={
                        'suffix': "%"
                    },

                    gauge={

                        'axis': {
                            'range': [0,100]
                        },

                        'bar': {
                            'thickness': 0.32
                        },

                        'steps': [

                            {
                                'range': [0,45],
                                'color': "#A8E6A3"
                            },

                            {
                                'range': [45,75],
                                'color': "#FFD580"
                            },

                            {
                                'range': [75,100],
                                'color': "#FF8A80"
                            }
                        ]
                    }
                )
            )

            gauge_fig.update_layout(
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10
                )
            )

            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        # =================================================
        # FEATURE CONTRIBUTION
        # =================================================

        with result2:

            st.markdown("""
            <div class='section-title'>
            Feature Influence on Current Prediction
            </div>
            """, unsafe_allow_html=True)

            explainer = shap.TreeExplainer(dg_model)

            shap_values = explainer.shap_values(
                prediction_pool
            )

            contribution_df = pd.DataFrame({

                "Feature": input_data.columns,

                "Contribution": np.abs(
                    shap_values[0]
                )
            })

            total_contribution = (
                contribution_df["Contribution"]
                .sum()
            )

            contribution_df[
                "ContributionPercent"
            ] = (

                contribution_df["Contribution"]
                / total_contribution

            ) * 100

            contribution_df = contribution_df[
                contribution_df[
                    "ContributionPercent"
                ] > 0.5
            ]

            contribution_df = contribution_df.sort_values(
                by="Contribution",
                ascending=True
            )

            contribution_fig = px.bar(

                contribution_df,

                x="ContributionPercent",

                y="Feature",

                orientation="h",

                text=contribution_df[
                    "ContributionPercent"
                ].round(2).astype(str) + "%",

                height=430
            )

            contribution_fig.update_traces(
                textposition="outside"
            )

            contribution_fig.update_layout(

                template="simple_white",

                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),

                xaxis_title="",

                yaxis_title="",

                showlegend=False
            )

            st.plotly_chart(
                contribution_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        # =================================================
        # OPERATIONAL AUDIT
        # =================================================

        with result3:

            st.markdown("""
            <div class='section-title'>
            Operational Risk Audit
            </div>
            """, unsafe_allow_html=True)

            reasons = []

            if is_critical_mishandling:
                reasons.append(
                    "Critical mishandling threshold exceeded."
                )

            if gas_handling_risk:
                reasons.append(
                    "Gas handling operational instability detected."
                )

            if thermal_expansion_risk:
                reasons.append(
                    "Thermal expansion hazard detected."
                )

            if shipper_hazard_combo:
                reasons.append(
                    "High-risk shipper and CAO combination detected."
                )

            if climate_shock_risk:
                reasons.append(
                    "Climate shock route-temperature condition detected."
                )

            if is_untrusted_shipper:
                reasons.append(
                    "Historical shipper incidents exceed threshold."
                )

            if len(reasons) == 0:

                st.success(
                    "No major operational risk indicators detected."
                )

            else:

                for r in reasons:
                    st.warning(r)
