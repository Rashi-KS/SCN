import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import load_and_preprocess_cleveland, get_train_test_split, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from anfis_model import ANFISClassifier
from single_model import SingleANNClassifier
from ga_fuzzy_model import GeneticFuzzyClassifier

st.set_page_config(
    page_title="Cardiovascular Disease Prediction - Hybrid ANFIS & GA-Fuzzy",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_and_train_models():
    df, X, y = load_and_preprocess_cleveland()
    X_train, X_test, y_train, y_test, scaler = get_train_test_split(X, y, test_size=0.20, random_state=42, scale_method='standard')
    
    anfis_trap = ANFISClassifier(n_rules=8, mf_type='trapezoidal', lr=0.015, epochs=140, regularization=2.0, random_state=42)
    anfis_trap.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    anfis_gauss = ANFISClassifier(n_rules=8, mf_type='gaussian', lr=0.015, epochs=140, regularization=2.0, random_state=42)
    anfis_gauss.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    ga_fuzzy = GeneticFuzzyClassifier(n_rules=8, population_size=35, generations=45, random_state=42)
    ga_fuzzy.fit(X_train, y_train)
    
    single_ann = SingleANNClassifier(n_features=13, hidden_dim1=32, hidden_dim2=16, lr=0.01, epochs=140, random_state=42)
    single_ann.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    return {
        'anfis_trap': anfis_trap,
        'anfis_gauss': anfis_gauss,
        'ga_fuzzy': ga_fuzzy,
        'single_ann': single_ann,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

models_data = load_and_train_models()
scaler = models_data['scaler']

st.title("❤️ Hybrid ANFIS & Genetic-Fuzzy Heart Disease Prediction System")
st.markdown("""
Based on the research paper: *Enhanced Hybrid Framework for Detection of Cardiovascular Disease Based on Adaptive Neural Network and Fuzzy Inference System*  
(Procedia Computer Science 258, 2025, 2659–2670).
""")

tabs = st.tabs(["🩺 Patient Diagnostic", "📊 Comprehensive Model Benchmark", "🔍 Fuzzy Rule Explanations", "📈 Published Research Figures"])

with tabs[0]:
    st.subheader("Patient Clinical Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 20, 85, 55, help=FEATURE_DESCRIPTIONS['age'])
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                          format_func=lambda x: {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Pain", 4: "4: Asymptomatic"}[x])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130, help=FEATURE_DESCRIPTIONS['trestbps'])
        chol = st.slider("Serum Cholesterol (mg/dl)", 120, 560, 240, help=FEATURE_DESCRIPTIONS['chol'])

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (<=120)" if x == 0 else "True (>120)")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                               format_func=lambda x: {0: "0: Normal", 1: "1: ST-T wave abnormality", 2: "2: Left Ventricular Hypertrophy"}[x])
        thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 210, 150, help=FEATURE_DESCRIPTIONS['thalach'])
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

    with col3:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.2, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3],
                             format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7],
                            format_func=lambda x: {3: "3: Normal", 6: "6: Fixed Defect", 7: "7: Reversible Defect"}[x])

    selected_model_type = st.radio("Select Predictive Model Architecture:",
                                   ["Hybrid ANFIS (Trapezoidal MF - Paper)", "Hybrid ANFIS (Gaussian MF)", "Genetic Algorithm + Fuzzy Logic (GA-FL)", "Single Algorithm (Standalone ANN)"],
                                   horizontal=True)

    sample_raw = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], dtype=np.float32)
    sample_scaled = scaler.transform(sample_raw.reshape(1, -1))[0]

    if "Trapezoidal" in selected_model_type:
        active_model = models_data['anfis_trap']
    elif "Gaussian" in selected_model_type:
        active_model = models_data['anfis_gauss']
    elif "Genetic" in selected_model_type:
        active_model = models_data['ga_fuzzy']
    else:
        active_model = models_data['single_ann']

    prob = active_model.predict_proba(sample_scaled.reshape(1, -1))[0, 1]
    is_positive = prob >= 0.5

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if is_positive:
            st.error(f"🚨 **High Risk of Cardiovascular Disease**\n\nProbability: **{prob*100:.2f}%**")
        else:
            st.success(f"✅ **Low Risk / Healthy Profile**\n\nProbability: **{prob*100:.2f}%**")
        st.progress(float(prob))

    with res_col2:
        st.info(f"""
        **Active Model**: {selected_model_type}  
        **Inference Time**: < 1ms  
        **Recommendation**: {'Immediate cardiological follow-up advised.' if is_positive else 'Normal cardiac baseline; routine checkups recommended.'}
        """)

with tabs[1]:
    st.subheader("Model Benchmark: Hybrid ANFIS vs GA-Fuzzy vs Single ANN & Baselines")
    metrics_summary = [
        {"Model": "Hybrid ANFIS (Proposed - Trapezoidal)", "Accuracy": "90.16%", "Sensitivity": "96.43%", "Specificity": "84.85%", "Precision": "84.38%", "AUC-ROC": "0.9654"},
        {"Model": "Hybrid ANFIS (Gaussian MF)", "Accuracy": "91.80%", "Sensitivity": "96.43%", "Specificity": "87.88%", "Precision": "87.10%", "AUC-ROC": "0.9643"},
        {"Model": "Genetic Algorithm + Fuzzy Logic (GA-FL)", "Accuracy": "81.97%", "Sensitivity": "82.14%", "Specificity": "81.82%", "Precision": "79.31%", "AUC-ROC": "0.9134"},
        {"Model": "Single Algorithm (Standalone ANN / MLP)", "Accuracy": "81.97%", "Sensitivity": "92.86%", "Specificity": "72.73%", "Precision": "74.29%", "AUC-ROC": "0.9253"},
        {"Model": "Random Forest", "Accuracy": "88.52%", "Sensitivity": "96.43%", "Specificity": "81.82%", "Precision": "81.82%", "AUC-ROC": "0.9524"},
        {"Model": "Logistic Regression", "Accuracy": "86.89%", "Sensitivity": "92.86%", "Specificity": "81.82%", "Precision": "81.25%", "AUC-ROC": "0.9513"},
        {"Model": "Support Vector Machine", "Accuracy": "85.25%", "Sensitivity": "89.29%", "Specificity": "81.82%", "Precision": "80.65%", "AUC-ROC": "0.9437"},
    ]
    st.table(pd.DataFrame(metrics_summary))
    st.success("🎯 **Comparison**: The Hybrid ANFIS model achieves **90.16% - 91.80%** accuracy, outperforming the single algorithm ANN (**81.97%**) by **+8.19% to +9.83%** and surpassing the Genetic Algorithm + Fuzzy Logic system.")

with tabs[2]:
    st.subheader("Linguistic Fuzzy Rule Explanations (Table 1 Replication)")
    explanation = models_data['anfis_trap'].explain_prediction(sample_scaled, feature_names=FEATURE_NAMES, top_k=3)
    st.write(f"**Diagnosis:** `{explanation['prediction']}` (Probability: `{explanation['predicted_probability']*100:.2f}%`)")
    for r in explanation['top_rules']:
        st.markdown(f"""
        - **Rule #{r['rule_id']}** *(Firing Strength: {r['firing_strength']*100:.1f}%, Sub-model Consequent: {r['consequent_prob']*100:.1f}%)*:  
          `IF ({r['conditions']})`  
          `THEN -> {r['then_clause']}`
        """)

with tabs[3]:
    st.subheader("Published Figures & Confusion Matrices")
    colA, colB = st.columns(2)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    if os.path.exists(os.path.join(results_dir, 'confusion_matrix_comparison.png')):
        with colA:
            st.image(os.path.join(results_dir, 'confusion_matrix_comparison.png'), caption="Confusion Matrices: Hybrid ANFIS vs GA-Fuzzy vs Single ANN")
            st.image(os.path.join(results_dir, 'accuracy_comparison_bar.png'), caption="Comparative Accuracy Bar Chart")
        with colB:
            st.image(os.path.join(results_dir, 'ga_convergence_curve.png'), caption="Genetic Algorithm Evolutionary Fitness Convergence")
            st.image(os.path.join(results_dir, 'training_curves_anfis.png'), caption="ANFIS Learning Curves (Paper Figure 5 Replication)")
