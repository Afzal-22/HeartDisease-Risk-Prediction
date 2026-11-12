
import datetime

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FB;
        color: #243247;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EAF1F8;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(120deg, #16324F 0%, #087E8B 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 18px rgba(22, 50, 79, 0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        color: white;
    }

    /* Cards */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #DFE8F0;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(22, 50, 79, 0.05);
    }

    /* High risk */
    .result-high {
        background: #FFF0EF;
        border-left: 6px solid #E76F51;
        color: #A83232;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* Low risk */
    .result-low {
        background: #E5F6F0;
        border-left: 6px solid #2A9D8F;
        color: #176B5B;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        background-color: #087E8B;
        color: white;
        border: none;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #066570;
        color: white;
        border: none;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #DFE8F0;
        box-shadow: 0 2px 8px rgba(22, 50, 79, 0.04);
    }

    /* Headings */
    h1, h2, h3 {
        color: #16324F;
    }

    /* Fixed Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #F4F7FB;
        border-top: 1px solid #DFE8F0;
        color: #52677D;
        text-align: center;
        padding: 12px;
        font-size: 13px;
        z-index: 999;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# LOAD MODEL FILES
@st.cache_resource
def load_artifacts():
    model = joblib.load("Logistic_Regression_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()


# SESSION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []


# BUILD MODEL INPUT
def build_input(values: dict):

    raw_input = {
        "Age": values["age"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "MaxHR": values["max_hr"],
        "Oldpeak": values["oldpeak"],
        "Sex_" + values["sex"]: 1,
        "ChestPainType_" + values["chest_pain"]: 1,
        "RestingECG_" + values["resting_ecg"]: 1,
        "ExerciseAngina_" + values["exercise_angina"]: 1,
        "ST_Slope_" + values["st_slope"]: 1,
    }

    df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]


# RISK FLAGS
def risk_flags(v: dict):

    flags = []

    if v["resting_bp"] >= 140:
        flags.append(
            "Your blood pressure is on the high side (140 or above)."
        )

    if v["cholesterol"] >= 240:
        flags.append(
            "Your cholesterol is high (240 or above)."
        )

    if v["fasting_bs"] == 1:
        flags.append(
            "Your fasting blood sugar is above the normal level."
        )

    if v["exercise_angina"] == "Y":
        flags.append(
            "You get chest pain when exercising."
        )

    if v["oldpeak"] >= 2.0:
        flags.append(
            "Your heart shows noticeable strain during exercise."
        )

    if v["st_slope"] in ("Flat", "Down"):
        flags.append(
            "Your heart trace during exercise is not rising as expected."
        )

    if v["chest_pain"] == "TA":
        flags.append(
            "You have classic heart-type chest pain."
        )

    if v["max_hr"] < 0.6 * (220 - v["age"]):
        flags.append(
            "Your highest heart rate is low for your age."
        )

    return flags


# GAUGE CHART
def gauge(prob: float):

    pct = prob * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": "Estimated risk"},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#52677D"
                },
                "bar": {
                    "color": "#087E8B"
                },
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#DFE8F0",
                "steps": [
                    {"range": [0, 33], "color": "#B7E4D5"},
                    {"range": [33, 66], "color": "#FCE3B5"},
                    {"range": [66, 100], "color": "#F4B6AE"},
                ],
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#16324F"),
    )

    return fig


# SIDEBAR INPUTS
with st.sidebar:

    st.header("Your Health Details")

    st.caption(
        "Fill in what you know. Hover over the ? icons for help. "
        "Numbers from a recent check-up or lab report work best."
    )

    st.subheader("About You")

    age = st.slider(
        "Age (years)",
        18, 100, 40
    )

    sex = st.radio(
        "Sex",
        ["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female",
        horizontal=True,
    )

    st.subheader("Blood Pressure and Blood Tests")

    resting_bp = st.number_input(
        "Blood pressure, top number (mm Hg)",
        80, 200, 120,
        help="The higher number in a blood pressure reading, taken while resting."
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100, 600, 200,
        help="Found in a blood test report."
    )

    fasting_bs = st.selectbox(
        "Is your fasting blood sugar above 120 mg/dL?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Blood sugar measured after not eating for about 8 hours."
    )

    st.subheader("Symptoms and Heart Tests")

    chest_pain = st.selectbox(
        "What kind of chest pain do you get?",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Mild or unusual chest pain",
            "NAP": "Chest pain not caused by the heart",
            "TA": "Classic heart-type pain (pressure or tightness)",
            "ASY": "No chest pain",
        }[x],
        help="Pick the closest match."
    )

    exercise_angina = st.radio(
        "Do you get chest pain when exercising?",
        ["Y", "N"],
        index=1,
        format_func=lambda x: "Yes" if x == "Y" else "No",
        horizontal=True,
    )

    max_hr = st.slider(
        "Highest heart rate reached (beats per minute)",
        60, 220, 150,
        help="Taken during exercise or a stress test."
    )

    resting_ecg = st.selectbox(
        "Heart tracing (ECG) at rest",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "Slight abnormality",
            "LVH": "Thickened heart wall",
        }[x],
        help="From an ECG test done while resting."
    )

    oldpeak = st.slider(
        "Heart strain during exercise",
        0.0, 6.0, 1.0, 0.1,
        help="A number from a stress test report (called ST depression or oldpeak)."
    )

    st_slope = st.selectbox(
        "Shape of the heart tracing during exercise",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Going up",
            "Flat": "Flat",
            "Down": "Going down"
        }[x],
        help="From the same stress test report."
    )

    predict_clicked = st.button(
        "🔍 Check My Risk",
        type="primary"
    )


# COLLECT INPUT VALUES
values = dict(
    age=age,
    sex=sex,
    chest_pain=chest_pain,
    resting_bp=resting_bp,
    cholesterol=cholesterol,
    fasting_bs=fasting_bs,
    resting_ecg=resting_ecg,
    max_hr=max_hr,
    exercise_angina=exercise_angina,
    oldpeak=oldpeak,
    st_slope=st_slope,
)


# MAIN HEADER
st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Answer a few simple questions in the sidebar to check your "
    "estimated risk of heart disease."
)


# METRICS
c1, c2, c3, c4 = st.columns(4)

c1.metric("Age", age)
c2.metric("Blood Pressure", f"{resting_bp} mm Hg")
c3.metric("Cholesterol", f"{cholesterol} mg/dL")
c4.metric("Highest Heart Rate", max_hr)


# PREDICTION
if predict_clicked:

    input_df = build_input(values)

    scaled = scaler.transform(input_df)

    prediction = int(model.predict(scaled)[0])

    prob = None

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled)[0][1])

    # SAVE HISTORY
    st.session_state.history.append(
        {
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            **{
                k.replace("_", " ").title(): v
                for k, v in values.items()
            },
            "Risk %": round(prob * 100, 1)
            if prob is not None else None,
            "Result": "High Risk" if prediction == 1 else "Low Risk",
        }
    )

    st.write("")

    left, right = st.columns([3, 2])

    with left:

        if prediction == 1:

            st.markdown(
                """
                <div class="result-high">
                    ⚠️ High Risk of Heart Disease
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="result-low">
                    ✅ Low Risk of Heart Disease
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        st.subheader("Things to Be Aware Of")

        flags = risk_flags(values)

        if flags:

            for f in flags:
                st.warning(f)

        else:

            st.info(
                "No obvious warning values in the inputs provided."
            )

        report = pd.DataFrame([values]).assign(
            Prediction="High Risk" if prediction == 1 else "Low Risk",
            RiskPercent=round(prob * 100, 1)
            if prob is not None else None,
        )

        st.download_button(
            "⬇️ Download Report (CSV)",
            report.to_csv(index=False),
            file_name="heart_risk_report.csv",
            mime="text/csv",
        )

    with right:

        if prob is not None:

            if HAS_PLOTLY:

                st.plotly_chart(
                    gauge(prob),
                    use_container_width=True
                )

            else:

                st.metric(
                    "Estimated Risk",
                    f"{prob * 100:.0f}%"
                )

                st.progress(prob)

else:

    st.info(
        "👈 Fill in the patient details in the sidebar "
        "and press Check My Risk."
    )


# SESSION HISTORY
if st.session_state.history:

    st.divider()

    st.subheader("Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    col_download, col_clear, _ = st.columns([2, 2, 4])

    with col_download:

        st.download_button(
            "⬇️ Download Session History (CSV)",
            history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

    with col_clear:

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()


# FIXED FOOTER
st.markdown(
    """
    <div class="footer">
        ⚕️ It is not a medical diagnostic tool.
        Consult a healthcare professional for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
