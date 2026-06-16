# CHRONIC KIDNEY DISEASE PREDICTION SYSTEM
# WITH SHAP + LIME EXPLAINABLE AI


# IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from fpdf import FPDF

# PAGE CONFIGURATION

st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
)

# LOAD TRAINED FILES

model = pickle.load(open("kidney_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

features = pickle.load(open("features.pkl", "rb"))

X_train = pickle.load(open("X_train.pkl", "rb"))

# SHAP EXPLAINER

shap_explainer = shap.Explainer(model)

# LIME EXPLAINER

lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=features,
    class_names=["CKD", "Not CKD"],
    mode="classification"
)

# TITLE

st.markdown(
    """
    <h1 style='text-align:center; color:green;'>
    NephroAI: Explainable Chronic Kidney Disease Prediction Platform
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter Patient Medical Details")

# SIDEBAR INPUTS

st.sidebar.header("Patient Information")

# NUMERICAL FEATURES

age = st.sidebar.number_input(
    "Age",
    min_value=2,
    max_value=90,
    value= 65
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure",
    min_value=50.0,
    max_value=200.0,
    value=90.0
)

specific_gravity = st.sidebar.number_input(
    "Specific Gravity",
    min_value=1.000,
    max_value=1.030,
    value=1.010,
    format="%.3f"
)

albumin = st.sidebar.number_input(
    "Albumin",
    min_value=0.0,
    max_value=5.0,
    value=4.0
)

sugar = st.sidebar.number_input(
    "Sugar",
    min_value=0.0,
    max_value=5.0,
    value=3.0
)

blood_glucose_random = st.sidebar.number_input(
    "Blood Glucose Random",
    min_value=50.0,
    max_value=500.0,
    value=250.0
)

blood_urea = st.sidebar.number_input(
    "Blood Urea",
    min_value=1.0,
    max_value=300.0,
    value=120.0
)

serum_creatinine = st.sidebar.number_input(
    "Serum Creatinine",
    min_value=0.1,
    max_value= 8.00,
    value= 1.8
)

sodium = st.sidebar.number_input(
    "Sodium",
    min_value=100.0,
    max_value=200.0,
    value=128.0
)

potassium = st.sidebar.number_input(
    "Potassium",
    min_value=1.0,
    max_value=10.0,
    value=5.8
)

haemoglobin = st.sidebar.number_input(
    "Haemoglobin",
    min_value=1.0,
    max_value=20.0,
    value=8.0
)

packed_cell_volume = st.sidebar.number_input(
    "Packed Cell Volume",
    min_value=1.0,
    max_value=60.0,
    value=30.0
)

white_blood_cell_count = st.sidebar.number_input(
    "White Blood Cell Count",
    min_value=1000.0,
    max_value=20000.0,
    value=1200.0
)

red_blood_cell_count = st.sidebar.number_input(
    "Red Blood Cell Count",
    min_value=1.0,
    max_value=10.0,
    value=3.2
)

# CATEGORICAL FEATURES

red_blood_cells = st.sidebar.selectbox(
    "Red Blood Cells",
    ["normal", "abnormal"]
)

pus_cell = st.sidebar.selectbox(
    "Pus Cell",
    ["normal", "abnormal"]
)

pus_cell_clumps = st.sidebar.selectbox(
    "Pus Cell Clumps",
    ["notpresent", "present"]
)

bacteria = st.sidebar.selectbox(
    "Bacteria",
    ["notpresent", "present"]
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    ["no", "yes"]
)

diabetes_mellitus = st.sidebar.selectbox(
    "Diabetes Mellitus",
    ["no", "yes"]
)

coronary_artery_disease = st.sidebar.selectbox(
    "Coronary Artery Disease",
    ["no", "yes"]
)

appetite = st.sidebar.selectbox(
    "Appetite",
    ["good", "poor"]
)

peda_edema = st.sidebar.selectbox(
    "Pedal Edema",
    ["no", "yes"]
)

aanemia = st.sidebar.selectbox(
    "Anemia",
    ["no", "yes"]
)

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Disease"):

    # ========================================================
    # ENCODING CATEGORICAL FEATURES
    # ========================================================

    rbc = 1 if red_blood_cells == "normal" else 0

    pc = 1 if pus_cell == "normal" else 0

    pcc = 1 if pus_cell_clumps == "present" else 0

    ba = 1 if bacteria == "present" else 0

    htn = 1 if hypertension == "yes" else 0

    dm = 1 if diabetes_mellitus == "yes" else 0

    cad = 1 if coronary_artery_disease == "yes" else 0

    appet = 1 if appetite == "poor" else 0

    pe = 1 if peda_edema == "yes" else 0

    ane = 1 if aanemia == "yes" else 0

    # ========================================================
    # CREATE Data Frame
    # ========================================================

    input_data = pd.DataFrame([[
    age,
    blood_pressure,
    specific_gravity,
    albumin,
    sugar,

    rbc,
    pc,
    pcc,
    ba,

    blood_glucose_random,
    blood_urea,
    serum_creatinine,
    sodium,
    potassium,
    haemoglobin,
    packed_cell_volume,
    white_blood_cell_count,
    red_blood_cell_count,

    htn,
    dm,
    cad,

    appet,
    pe,
    ane
        ]], columns=features)


    # ========================================================
    # FEATURE SCALING
    # ========================================================

    input_data_scaled = scaler.transform(input_data)

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(input_data_scaled)

    probability = model.predict_proba(input_data_scaled)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader("Prediction Result")

    if prediction[0] == 0:

        st.error("CKD Detected")

        confidence = probability[0][0] * 100

        st.write(f"Prediction Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

    else:

        st.success("No CKD Detected")

        confidence = probability[0][1] * 100

        st.write(f"Prediction Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

    # ========================================================
    # HEALTH PARAMETER CHART
    # ========================================================

    st.subheader("Important Health Parameters")

    chart_data = pd.DataFrame({

        "Feature": [
            "Blood Pressure",
            "Blood Urea",
            "Serum Creatinine",
            "Haemoglobin"
        ],

        "Value": [
            blood_pressure,
            blood_urea,
            serum_creatinine,
            haemoglobin
        ]
    })

    st.bar_chart(chart_data.set_index("Feature"))

    
    # =====================================
    # SHAP EXPLAINABILITY
    # =====================================

    st.subheader("SHAP Explainability")

    # Create SHAP Explainer
    explainer = shap.Explainer(
    model,
    feature_names= features
    )
        
    # Generate SHAP values
    shap_values = explainer(input_data_scaled)

    # Create Figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Explain predicted class automatically
    shap.plots.waterfall(
    shap_values[0, :, prediction[0]],
    max_display=10,
    show=False
    )

    # Show plot in Streamlit
    st.pyplot(fig)
    # ========================================================
    # LIME EXPLANATION
    # ========================================================

    st.subheader("LIME Explanation")

    lime_exp = lime_explainer.explain_instance(
        input_data_scaled[0],
        model.predict_proba,
        num_features=10
    )

    fig = lime_exp.as_pyplot_figure()

    st.pyplot(fig)

    # ========================================================
    # PDF REPORT
    # ========================================================

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10,
             txt="CKD Prediction Report",
             ln=True)

    pdf.ln(10)

    if prediction[0] == 0:

        pdf.cell(200, 10,
                 txt="Prediction: CKD Detected",
                 ln=True)

    else:

        pdf.cell(200, 10,
                 txt="Prediction: No CKD Detected",
                 ln=True)

    pdf.cell(200, 10,
             txt=f"Confidence: {confidence:.2f}%",
             ln=True)

    pdf.output("CKD_Report.pdf")

    # ========================================================
    # DOWNLOAD REPORT BUTTON
    # ========================================================

    with open("CKD_Report.pdf", "rb") as file:

        st.download_button(
            label="Download Report",
            data=file,
            file_name="CKD_Report.pdf",
            mime="application/pdf"
        )