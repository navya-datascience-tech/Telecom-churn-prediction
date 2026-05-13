import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

.big-title {
    font-size: 50px;
    font-weight: bold;
    color: #00E5FF;
    text-align: center;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 22px;
    border-radius: 12px;
    height: 60px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    color: white;
}

.result-success {
    background-color: rgba(0,255,100,0.2);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    color: #00ff99;
    font-weight: bold;
}

.result-danger {
    background-color: rgba(255,0,0,0.2);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    color: #ff4d4d;
    font-weight: bold;
}

.metric-box {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD MODEL FILES
# -----------------------------------------
model = joblib.load("telecom_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------------------
# HEADER
# -----------------------------------------
st.markdown(
    '<p class="big-title">📡 AI Telecom Churn Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Predict Customer Retention Using Machine Learning Intelligence</p>',
    unsafe_allow_html=True
)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4149/4149673.png",
    width=120
)

st.sidebar.title("Project Info")

st.sidebar.info("""
### Developed By
Machine Learning Project

### Algorithm Used
Random Forest Classifier

### Features
- Customer Behavior Analysis
- Churn Risk Prediction
- Probability Score
- Interactive Dashboard
""")

# -----------------------------------------
# INPUT SECTION
# -----------------------------------------
st.markdown("## 📋 Customer Information")

col1, col2 = st.columns(2)

with col1:

    account_length = st.number_input(
        "📞 Account Length",
        min_value=0.0
    )

    voice_messages = st.number_input(
        "🎙 Voice Messages",
        min_value=0.0
    )

    intl_calls = st.number_input(
        "🌍 International Calls",
        min_value=0.0
    )

    intl_charge = st.number_input(
        "💰 International Charge",
        min_value=0.0
    )

    day_mins = st.number_input(
        "☀️ Day Minutes",
        min_value=0.0
    )

    day_calls = st.number_input(
        "📲 Day Calls",
        min_value=0.0
    )

    day_charge = st.number_input(
        "💵 Day Charge",
        min_value=0.0
    )

with col2:

    eve_mins = st.number_input(
        "🌆 Evening Minutes",
        min_value=0.0
    )

    eve_calls = st.number_input(
        "📞 Evening Calls",
        min_value=0.0
    )

    eve_charge = st.number_input(
        "💳 Evening Charge",
        min_value=0.0
    )

    night_calls = st.number_input(
        "🌙 Night Calls",
        min_value=0.0
    )

    night_charge = st.number_input(
        "💸 Night Charge",
        min_value=0.0
    )

    customer_calls = st.number_input(
        "☎️ Customer Service Calls",
        min_value=0.0
    )

    intl_plan = st.selectbox(
        "🌐 International Plan",
        ["yes", "no"]
    )

# -----------------------------------------
# DATA PREPARATION
# -----------------------------------------
intl_plan = 1 if intl_plan == "yes" else 0

input_data = pd.DataFrame({
    'account.length': [account_length],
    'voice.messages': [voice_messages],
    'intl.calls': [intl_calls],
    'intl.charge': [intl_charge],
    'day.mins': [day_mins],
    'day.calls': [day_calls],
    'day.charge': [day_charge],
    'eve.mins': [eve_mins],
    'eve.calls': [eve_calls],
    'eve.charge': [eve_charge],
    'night.calls': [night_calls],
    'night.charge': [night_charge],
    'customer.calls': [customer_calls],
    'intl.plan_yes': [intl_plan]
})

input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
)

input_scaled = scaler.transform(input_data)

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if st.button("🚀 Predict Customer Churn"):

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    st.markdown("## 📊 Prediction Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

    with col2:
        risk = "High" if probability > 0.5 else "Low"
        st.metric(
            label="Risk Level",
            value=risk
        )

    with col3:
        confidence = max(probability, 1 - probability)
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )

    st.progress(float(probability))

    st.markdown("### 🧠 Final Prediction")

    if prediction[0] == "yes":

        st.markdown(
            """
            <div class="result-danger">
            ⚠️ Customer Likely To Churn
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning("""
Recommended Actions:
- Offer retention discount
- Improve customer support
- Provide loyalty benefits
- Engage customer with offers
""")

    else:

        st.markdown(
            """
            <div class="result-success">
            ✅ Customer Likely To Stay
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("""
Customer shows strong retention behavior.
Continue providing quality service.
""")

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")

st.markdown("""
<center>
<h4>📡 AI Powered Telecom Analytics Dashboard</h4>
<p>Developed using Streamlit • Scikit-Learn • Machine Learning</p>
</center>
""", unsafe_allow_html=True)