# Enhanced Hybrid Framework for Cardiovascular Disease Detection: ANFIS, Genetic-Fuzzy & Single ANN

This repository implements the hybrid neuro-fuzzy methodology and evolutionary fuzzy computing algorithms described in the research paper:

> **"Enhanced Hybrid Framework for Detection of Cardiovascular Disease Based on Adaptive Neural Network and Fuzzy Inference System"**  
> *Subham Kumar Padhy, Anjali Mohapatra, Sabyasachi Patra*  
> **Procedia Computer Science 258 (2025) 2659–2670**  
> International Conference on Machine Learning and Data Engineering (ICMLDE)

---

## 🌟 Executive Summary & Key Results

The framework implements and compares three principal intelligent architectures alongside conventional baselines on the **UCI Cleveland Dataset** (303 records, 14 clinical attributes):

1. **Hybrid ANFIS (Neuro-Fuzzy - Proposed in Paper)**:
   A 5-layer Adaptive Neuro-Fuzzy Inference System using parameter-adjusted trapezoidal/gaussian membership functions, rule firing normalization, and Takagi-Sugeno linear consequents.
2. **Genetic Algorithm + Fuzzy Logic (GA-FL / GA-FIS)**:
   An evolutionary fuzzy rule-based system (as studied in Aghamohammadi et al. Ref [5], Santhanam & Ephzibah Ref [8], and Reddy et al. Ref [6]) that optimizes fuzzy membership anchors, rule consequents, and feature salience via selection, crossover, and mutation over generations.
3. **Single Algorithm Model (Standalone ANN / MLP)**:
   A standalone Multi-Layer Perceptron neural network without fuzzy inference layers.
4. **Conventional Baselines**:
   Random Forest, Logistic Regression, Support Vector Machine, Decision Tree, and k-NN.

---

### 📊 Benchmark Performance (Held-Out Test Set 80:20 Split)

| Model Architecture | Accuracy (%) | Sensitivity / Recall (%) | Specificity (%) | Precision (%) | F1-Score (%) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid ANFIS (Proposed - Trapezoidal MF)** | **90.16%** | **96.43%** | **84.85%** | **84.38%** | **90.00%** | **0.9654** |
| **Hybrid ANFIS (Gaussian MF)** | **91.80%** | **96.43%** | **87.88%** | **87.10%** | **91.53%** | **0.9643** |
| **Genetic Algorithm + Fuzzy Logic (GA-FL)** | **81.97%** | **82.14%** | **81.82%** | **79.31%** | **80.70%** | **0.9134** |
| **Single Algorithm (Standalone ANN / MLP)** | **81.97%** | **92.86%** | **72.73%** | **74.29%** | **82.54%** | **0.9253** |
| **Random Forest** | 88.52% | 96.43% | 81.82% | 81.82% | 88.52% | 0.9524 |
| **Logistic Regression** | 86.89% | 92.86% | 81.82% | 81.25% | 86.67% | 0.9513 |
| **Support Vector Machine (RBF)** | 85.25% | 89.29% | 81.82% | 80.65% | 84.75% | 0.9437 |
| **k-Nearest Neighbors (k=5)** | 88.52% | 100.00% | 78.79% | 80.00% | 88.89% | 0.9232 |
| **Decision Tree (Max Depth 4)** | 78.69% | 82.14% | 75.76% | 74.19% | 77.97% | 0.8642 |

---

### 🔍 Confusion Matrix Breakdown (Held-Out Test Set: N = 61)

#### 1. Hybrid ANFIS (Proposed) — Accuracy: 90.16%
```text
                   Predicted Healthy (0)    Predicted Disease (1)
Actual Healthy (0):        28 (TN)                  5 (FP)
Actual Disease (1):         1 (FN)                 27 (TP)
```
* **Specificity**: 28 / 33 = **84.85%**
* **Sensitivity**: 27 / 28 = **96.43%** (Only 1 missed cardiovascular case!)

#### 2. Genetic Algorithm + Fuzzy Logic (GA-FL) — Accuracy: 81.97%
```text
                   Predicted Healthy (0)    Predicted Disease (1)
Actual Healthy (0):        27 (TN)                  6 (FP)
Actual Disease (1):         5 (FN)                 23 (TP)
```
* **Specificity**: 27 / 33 = **81.82%**
* **Sensitivity**: 23 / 28 = **82.14%**

#### 3. Single Algorithm (Standalone ANN) — Accuracy: 81.97%
```text
                   Predicted Healthy (0)    Predicted Disease (1)
Actual Healthy (0):        24 (TN)                  9 (FP)
Actual Disease (1):         2 (FN)                 26 (TP)
```
* **Specificity**: 24 / 33 = **72.73%**
* **Sensitivity**: 26 / 28 = **92.86%**

---

### 📈 10-Fold Cross-Validation Summary
* **Genetic Algorithm + Fuzzy Logic (GA-FL)**: **83.43%** (+/- 6.56%)
* **Hybrid ANFIS (Proposed)**: **81.83%** (+/- 6.49%)
* **Single Algorithm (Standalone ANN)**: **81.14%** (+/- 5.88%)

---

## 🗂️ Project File Structure

```text
anfis_heart_disease/
├── processed.cleveland.data      # UCI Cleveland dataset (303 records, 14 attributes)
├── data_preprocessing.py         # Missing-value imputation, feature scaling, 80:20 & 10-fold CV
├── anfis_model.py                # 5-Layer ANFIS classifier & fuzzy rule explanation engine
├── ga_fuzzy_model.py             # Genetic Algorithm + Fuzzy Inference System (GA-FL)
├── single_model.py               # Standalone ANN (MLP) single-algorithm model & baselines
├── experiment_runner.py          # Complete benchmarking, cross-validation & graph generation
├── app.py                        # Interactive Web Diagnostic App (Flask + TailwindCSS + API)
├── app_streamlit.py              # Streamlit Interactive Demonstration Dashboard
└── results/                      # Generated figures & metrics
    ├── accuracy_comparison_bar.png
    ├── confusion_matrix_comparison.png       # 3-Panel: ANFIS vs GA-Fuzzy vs Single ANN
    ├── ga_convergence_curve.png              # Fitness evolution over 60 generations
    ├── membership_functions.png
    ├── roc_comparison.png
    ├── training_curves_anfis.png
    ├── model_comparison_metrics.json
    └── membership_functions_metrics.json
```

---

## 🚀 How to Run

### 1. Run Complete Benchmark Experiments
```bash
python experiment_runner.py
```

### 2. Launch Interactive Web Diagnostic App (Flask)
```bash
python app.py
```
Open `http://127.0.0.1:5000` in any web browser.

### 3. Launch Streamlit Dashboard (Optional)
```bash
streamlit run app_streamlit.py
```
