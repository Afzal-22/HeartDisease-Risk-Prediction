import datetime

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(120deg, #e63946 0%, #8e2de2 100%);
        padding: 1.8rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .result-high {
        background: rgba(230,57,70,.12); border-left: 6px solid #e63946;
        padding: 1rem 1.3rem; border-radius: 10px; font-size: 1.25rem;
        font-weight: 600;
    }
    .result-low {
        background: rgba(42,157,143,.12); border-left: 6px solid #2a9d8f;
        padding: 1rem 1.3rem; border-radius: 10px; font-size: 1.25rem;
        font-weight: 600;
    }
    div.stButton > button {
        width: 100%; border-radius: 10px; height: 3rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

if "history" not in st.session_state:
    st.session_state.history = []

def build_input(values: dict) -> pd.DataFrame:
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

def risk_flags(v: dict):
    flags = []
    if v["resting_bp"] >= 140:
        flags.append("Your blood pressure is on the high side (140 or above).")
    if v["cholesterol"] >= 240:
        flags.append("Your cholesterol is high (240 or above).")
    if v["fasting_bs"] == 1:
        flags.append("Your fasting blood sugar is above the normal level.")
    if v["exercise_angina"] == "Y":
        flags.append("You get chest pain when exercising.")
    if v["oldpeak"] >= 2.0:
        flags.append("Your heart shows noticeable strain during exercise.")
    if v["st_slope"] in ("Flat", "Down"):
        flags.append("Your heart trace during exercise is not rising as expected.")
    if v["chest_pain"] == "TA":
        flags.append("You have classic heart-type chest pain.")
    if v["max_hr"] < 0.6 * (220 - v["age"]):
        flags.append("Your highest heart rate is low for your age.")
    return flags

def gauge(prob: float):
    pct = prob * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": "Estimated risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8e2de2"},
                "steps": [
                    {"range": [0, 33], "color": "#2a9d8f"},
                    {"range": [33, 66], "color": "#f4a261"},
                    {"range": [66, 100], "color": "#e63946"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig

with st.sidebar:
    st.header("Your health details")
    st.caption(
        "Fill in what you know. Hover over the ? icons for help. "
        "Numbers from a recent check-up or lab report work best."
    )

    st.subheader("About you")
    age = st.slider("Age (years)", 18, 100, 40)
    sex = st.radio(
        "Sex",
        ["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female",
        horizontal=True,
    )

    st.subheader("Blood pressure and blood tests")
    resting_bp = st.number_input(
        "Blood pressure, top number (mm Hg)",
        80, 200, 120,
        help="The higher number in a blood pressure reading, taken while resting. Around 120 is normal.",
    )
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100, 600, 200,
        help="Found in a blood test report. Below 200 is considered good.",
    )
    fasting_bs = st.selectbox(
        "Is your fasting blood sugar above 120 mg/dL?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Blood sugar measured after not eating for about 8 hours. Choose No if you are not sure.",
    )

    st.subheader("Symptoms and heart tests")
    chest_pain = st.selectbox(
        "What kind of chest pain do you get?",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Mild or unusual chest pain",
            "NAP": "Chest pain not caused by the heart",
            "TA": "Classic heart-type pain (pressure or tightness)",
            "ASY": "No chest pain",
        }[x],
        help="Pick the closest match. A doctor can tell you which type it is.",
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
        help="Taken during exercise or a stress test. A rough guide for the maximum is 220 minus your age.",
    )
    resting_ecg = st.selectbox(
        "Heart tracing (ECG) at rest",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "Slight abnormality",
            "LVH": "Thickened heart wall",
        }[x],
        help="From an ECG test done while resting. Choose Normal if you have not had one.",
    )
    oldpeak = st.slider(
        "Heart strain during exercise",
        0.0, 6.0, 1.0, 0.1,
        help="A number from a stress test report (called ST depression or oldpeak). 0 means no strain. Keep the default if you do not know.",
    )
    st_slope = st.selectbox(
        "Shape of the heart tracing during exercise",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {"Up": "Going up", "Flat": "Flat", "Down": "Going down"}[x],
        help="From the same stress test report (called ST slope). Choose Going up if you are not sure.",
    )

    predict_clicked = st.button("🔍 Check my risk", type="primary")

values = dict(
    age=age, sex=sex, chest_pain=chest_pain, resting_bp=resting_bp,
    cholesterol=cholesterol, fasting_bs=fasting_bs, resting_ecg=resting_ecg,
    max_hr=max_hr, exercise_angina=exercise_angina, oldpeak=oldpeak,
    st_slope=st_slope,
)

st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("Answer a few simple questions in the sidebar to check your risk of heart disease.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("Blood pressure", f"{resting_bp} mm Hg")
c3.metric("Cholesterol", f"{cholesterol} mg/dL")
c4.metric("Highest heart rate", max_hr)

if predict_clicked:
    input_df = build_input(values)
    scaled = scaler.transform(input_df)
    prediction = int(model.predict(scaled)[0])

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled)[0][1])

    st.session_state.history.append(
        {
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            **{k.replace("_", " ").title(): v for k, v in values.items()},
            "Risk %": round(prob * 100, 1) if prob is not None else None,
            "Result": "High Risk" if prediction == 1 else "Low Risk",
        }
    )

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        if prediction == 1:
            st.markdown(
                '<div class="result-high">⚠️ High Risk of Heart Disease</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-low">✅ Low Risk of Heart Disease</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.subheader("Things to be aware of")
        flags = risk_flags(values)
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.info("No obvious warning values in the inputs provided.")

        report = pd.DataFrame([values]).assign(
            Prediction="High Risk" if prediction == 1 else "Low Risk",
            RiskPercent=round(prob * 100, 1) if prob is not None else None,
        )
        st.download_button(
            "⬇️ Download report (CSV)",
            report.to_csv(index=False),
            file_name="heart_risk_report.csv",
            mime="text/csv",
        )

    with right:
        if prob is not None:
            if HAS_PLOTLY:
                st.plotly_chart(gauge(prob), use_container_width=True)
            else:
                st.metric("Estimated risk", f"{prob * 100:.0f}%")
                st.progress(prob)
else:
    st.info("👈 Fill in the patient details in the sidebar and press **Check my risk**.")

if st.session_state.history:
    st.divider()
    st.subheader("Session history")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df.iloc[::-1], use_container_width=True, hide_index=True
    )
    st.download_button(
        "⬇️ Download session history (CSV)",
        history_df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )