import os
import tempfile
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from data_preprocessing import load_and_preprocess_cleveland, get_train_test_split, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from anfis_model import ANFISClassifier
from single_model import SingleANNClassifier
from ga_fuzzy_model import GeneticFuzzyClassifier

app = Flask(__name__)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Global cache for trained models
MODELS = {}

def init_models():
    print("Initializing and training models for web interface...")
    df, X, y = load_and_preprocess_cleveland()
    X_train, X_test, y_train, y_test, scaler = get_train_test_split(X, y, test_size=0.20, random_state=42, scale_method='standard')
    
    # 1. Hybrid ANFIS (Trapezoidal)
    anfis_trap = ANFISClassifier(n_rules=8, mf_type='trapezoidal', lr=0.015, epochs=140, regularization=2.0, random_state=42)
    anfis_trap.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # 2. Hybrid ANFIS (Gaussian)
    anfis_gauss = ANFISClassifier(n_rules=8, mf_type='gaussian', lr=0.015, epochs=140, regularization=2.0, random_state=42)
    anfis_gauss.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # 3. Genetic Algorithm + Fuzzy Logic (GA-FL)
    ga_fuzzy = GeneticFuzzyClassifier(n_rules=8, population_size=35, generations=45, random_state=42)
    ga_fuzzy.fit(X_train, y_train)
    
    # 4. Single Algorithm (Standalone ANN)
    single_ann = SingleANNClassifier(n_features=13, hidden_dim1=32, hidden_dim2=16, lr=0.01, epochs=140, random_state=42)
    single_ann.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    MODELS['anfis_trap'] = anfis_trap
    MODELS['anfis_gauss'] = anfis_gauss
    MODELS['ga_fuzzy'] = ga_fuzzy
    MODELS['single_ann'] = single_ann
    MODELS['scaler'] = scaler
    print("All models ready (Hybrid ANFIS, GA-Fuzzy, Single ANN)!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid ANFIS & Genetic-Fuzzy Heart Disease Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-card { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); }
        .danger-gradient { background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%); }
        .safe-gradient { background: linear-gradient(135deg, #15803d 0%, #22c55e 100%); }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
    <!-- Navbar -->
    <header class="bg-indigo-900 text-white shadow-lg sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <i class="fa-solid fa-heart-pulse text-red-400 text-3xl"></i>
                <div>
                    <h1 class="text-xl font-bold tracking-tight">Neuro-Fuzzy & Genetic-Fuzzy Heart Disease Prediction</h1>
                    <p class="text-xs text-indigo-200">Procedia CS (2025) · Hybrid ANFIS vs GA-Fuzzy vs Single ANN</p>
                </div>
            </div>
            <div class="flex space-x-2">
                <span class="px-3 py-1 bg-indigo-800 rounded-full text-xs font-semibold text-emerald-300 border border-indigo-700">
                    <i class="fa-solid fa-circle text-xs mr-1 animate-pulse"></i> 4 Models Active
                </span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">
        <!-- Highlights Banner -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <p class="text-xs font-semibold text-slate-500 uppercase">Hybrid ANFIS Accuracy</p>
                <p class="text-2xl font-black text-indigo-600 mt-1">90.16% – 91.80%</p>
                <p class="text-xs text-emerald-600 mt-1"><i class="fa-solid fa-arrow-up"></i> +8.19% vs Single ANN</p>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <p class="text-xs font-semibold text-slate-500 uppercase">GA + Fuzzy Logic (10-Fold)</p>
                <p class="text-2xl font-black text-emerald-600 mt-1">83.43%</p>
                <p class="text-xs text-slate-500 mt-1">Evolutionary rule tuning</p>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <p class="text-xs font-semibold text-slate-500 uppercase">Single Algorithm (ANN)</p>
                <p class="text-2xl font-black text-orange-500 mt-1">81.97%</p>
                <p class="text-xs text-slate-500 mt-1">Pure Multi-Layer Perceptron</p>
            </div>
            <div class="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
                <p class="text-xs font-semibold text-slate-500 uppercase">ANFIS Sensitivity</p>
                <p class="text-2xl font-black text-blue-600 mt-1">96.43%</p>
                <p class="text-xs text-slate-500 mt-1">High detection of true cases</p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left 2 Cols: Input Form -->
            <div class="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <div class="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
                    <div>
                        <h2 class="text-lg font-bold text-slate-900">Patient Clinical Parameters</h2>
                        <p class="text-xs text-slate-500">Cleveland Clinical Heart Disease Protocol</p>
                    </div>
                    <div class="flex items-center space-x-2">
                        <label class="text-xs font-semibold text-slate-600">Model Engine:</label>
                        <select id="model_type" onchange="predictRisk()" class="text-xs bg-slate-100 border border-slate-300 rounded-lg px-2.5 py-1.5 font-medium focus:ring-2 focus:ring-indigo-500">
                            <option value="anfis_trap">Hybrid ANFIS (Trapezoidal MF - Paper)</option>
                            <option value="anfis_gauss">Hybrid ANFIS (Gaussian MF)</option>
                            <option value="ga_fuzzy">Genetic Algorithm + Fuzzy Logic (GA-FL)</option>
                            <option value="single_ann">Single Algorithm (Standalone ANN)</option>
                        </select>
                    </div>
                </div>

                <form id="diagnostic-form" class="space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                        <!-- Age -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Age (years)</span>
                                <span id="age_val" class="text-indigo-600 font-bold">55</span>
                            </div>
                            <input type="range" id="age" min="25" max="80" value="55" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer" oninput="updateVal('age')">
                        </div>

                        <!-- Sex -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Sex</label>
                            <select id="sex" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="1">Male (1)</option>
                                <option value="0">Female (0)</option>
                            </select>
                        </div>

                        <!-- Chest Pain -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Chest Pain Type (cp)</label>
                            <select id="cp" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="4" selected>4: Asymptomatic (High Risk)</option>
                                <option value="1">1: Typical Angina</option>
                                <option value="2">2: Atypical Angina</option>
                                <option value="3">3: Non-Anginal Pain</option>
                            </select>
                        </div>

                        <!-- Blood Pressure -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Resting Blood Pressure (mm Hg)</span>
                                <span id="trestbps_val" class="text-indigo-600 font-bold">140</span>
                            </div>
                            <input type="range" id="trestbps" min="90" max="200" value="140" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer" oninput="updateVal('trestbps')">
                        </div>

                        <!-- Cholesterol -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Serum Cholesterol (mg/dl)</span>
                                <span id="chol_val" class="text-indigo-600 font-bold">250</span>
                            </div>
                            <input type="range" id="chol" min="120" max="500" value="250" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer" oninput="updateVal('chol')">
                        </div>

                        <!-- Fasting Sugar -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Fasting Blood Sugar > 120 mg/dl (fbs)</label>
                            <select id="fbs" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="0">False (&le; 120 mg/dl)</option>
                                <option value="1">True (> 120 mg/dl)</option>
                            </select>
                        </div>

                        <!-- Resting ECG -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Resting ECG Results</label>
                            <select id="restecg" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="0">0: Normal</option>
                                <option value="1">1: ST-T wave abnormality</option>
                                <option value="2" selected>2: Left ventricular hypertrophy</option>
                            </select>
                        </div>

                        <!-- Max HR -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Maximum Heart Rate (thalach)</span>
                                <span id="thalach_val" class="text-indigo-600 font-bold">130</span>
                            </div>
                            <input type="range" id="thalach" min="70" max="210" value="130" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer" oninput="updateVal('thalach')">
                        </div>

                        <!-- Exercise Angina -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Exercise Induced Angina (exang)</label>
                            <select id="exang" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="1" selected>Yes (1)</option>
                                <option value="0">No (0)</option>
                            </select>
                        </div>

                        <!-- Oldpeak -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>ST Depression (oldpeak)</span>
                                <span id="oldpeak_val" class="text-indigo-600 font-bold">2.0</span>
                            </div>
                            <input type="range" id="oldpeak" min="0" max="6" step="0.1" value="2.0" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer" oninput="updateVal('oldpeak')">
                        </div>

                        <!-- Slope -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">ST Slope</label>
                            <select id="slope" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="1">1: Upsloping</option>
                                <option value="2" selected>2: Flat</option>
                                <option value="3">3: Downsloping</option>
                            </select>
                        </div>

                        <!-- CA -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Fluoroscopy Vessels (ca)</label>
                            <select id="ca" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="0">0</option>
                                <option value="1">1</option>
                                <option value="2" selected>2</option>
                                <option value="3">3</option>
                            </select>
                        </div>

                        <!-- Thal -->
                        <div>
                            <label class="block text-xs font-semibold mb-1">Thalassemia (thal)</label>
                            <select id="thal" onchange="predictRisk()" class="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5">
                                <option value="3">3: Normal</option>
                                <option value="6">6: Fixed defect</option>
                                <option value="7" selected>7: Reversible defect</option>
                            </select>
                        </div>
                    </div>

                    <div class="pt-4 flex space-x-3">
                        <button type="button" onclick="predictRisk()" class="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-xl shadow-md transition flex items-center justify-center space-x-2">
                            <i class="fa-solid fa-stethoscope"></i>
                            <span>Evaluate Risk with Selected Model</span>
                        </button>
                        <button type="button" onclick="loadSampleCase('high')" class="bg-red-50 hover:bg-red-100 text-red-700 font-medium text-xs px-3 py-2 rounded-xl border border-red-200">
                            High Risk Preset
                        </button>
                        <button type="button" onclick="loadSampleCase('low')" class="bg-emerald-50 hover:bg-emerald-100 text-emerald-700 font-medium text-xs px-3 py-2 rounded-xl border border-emerald-200">
                            Healthy Preset
                        </button>
                    </div>
                </form>
            </div>

            <!-- Right Col: Output & Explanations -->
            <div class="space-y-6">
                <!-- Result Card -->
                <div id="result-box" class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                    <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wide mb-3">Diagnostic Assessment</h3>
                    <div id="risk-badge" class="p-4 rounded-xl text-white text-center mb-4 danger-gradient">
                        <p class="text-xs uppercase font-bold tracking-widest opacity-80" id="risk-status">Cardiovascular Disease Detected</p>
                        <p class="text-4xl font-black my-1" id="risk-prob">94.8%</p>
                        <p class="text-xs opacity-90" id="risk-detail">High likelihood of arterial stenosis</p>
                    </div>

                    <div class="w-full bg-slate-100 rounded-full h-3 mb-4 overflow-hidden">
                        <div id="prob-bar" class="bg-red-500 h-3 rounded-full transition-all duration-500" style="width: 94.8%"></div>
                    </div>

                    <div class="text-xs text-slate-600 space-y-1.5 pt-2 border-t border-slate-100">
                        <div class="flex justify-between">
                            <span class="font-medium">Active Model:</span>
                            <span id="res-model-name" class="font-bold text-indigo-700">Hybrid ANFIS</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">Validation Status:</span>
                            <span class="text-emerald-600 font-bold"><i class="fa-solid fa-check-circle"></i> Validated (80:20 Split)</span>
                        </div>
                    </div>
                </div>

                <!-- Fuzzy Rule Explanation Box -->
                <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="text-sm font-bold text-slate-800 flex items-center">
                            <i class="fa-solid fa-brain text-indigo-600 mr-2"></i>
                            Fuzzy Linguistic Rule Explanation
                        </h3>
                        <span class="text-xs text-slate-400 font-mono">Table 1</span>
                    </div>
                    <p class="text-xs text-slate-500 mb-3">
                        Active fuzzy rules showing premise activation degrees and local risk consequence:
                    </p>
                    <div id="rules-container" class="space-y-3">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Research Plots Gallery -->
        <div class="mt-10 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h2 class="text-lg font-bold text-slate-900 mb-4">Empirical Research Figures & Confusion Matrices</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h4 class="text-xs font-bold text-slate-600 mb-2 uppercase">1. Model Accuracy Comparison (Table 2 & Improvements)</h4>
                    <img src="/results/accuracy_comparison_bar.png" class="rounded-xl border border-slate-200 w-full" alt="Accuracy Bar">
                </div>
                <div>
                    <h4 class="text-xs font-bold text-slate-600 mb-2 uppercase">2. Confusion Matrices: Hybrid ANFIS vs GA-Fuzzy vs Single ANN</h4>
                    <img src="/results/confusion_matrix_comparison.png" class="rounded-xl border border-slate-200 w-full" alt="Confusion Matrix">
                </div>
                <div>
                    <h4 class="text-xs font-bold text-slate-600 mb-2 uppercase">3. Genetic Algorithm Optimization Convergence</h4>
                    <img src="/results/ga_convergence_curve.png" class="rounded-xl border border-slate-200 w-full" alt="GA Convergence">
                </div>
                <div>
                    <h4 class="text-xs font-bold text-slate-600 mb-2 uppercase">4. ROC Curves Comparison (AUC)</h4>
                    <img src="/results/roc_comparison.png" class="rounded-xl border border-slate-200 w-full" alt="ROC Comparison">
                </div>
            </div>
        </div>
    </main>

    <script>
        function updateVal(id) {
            document.getElementById(id + '_val').innerText = document.getElementById(id).value;
            predictRisk();
        }

        function loadSampleCase(type) {
            if (type === 'high') {
                document.getElementById('age').value = 67;
                document.getElementById('sex').value = 1;
                document.getElementById('cp').value = 4;
                document.getElementById('trestbps').value = 160;
                document.getElementById('chol').value = 286;
                document.getElementById('fbs').value = 0;
                document.getElementById('restecg').value = 2;
                document.getElementById('thalach').value = 108;
                document.getElementById('exang').value = 1;
                document.getElementById('oldpeak').value = 1.5;
                document.getElementById('slope').value = 2;
                document.getElementById('ca').value = 3;
                document.getElementById('thal').value = 3;
            } else {
                document.getElementById('age').value = 41;
                document.getElementById('sex').value = 0;
                document.getElementById('cp').value = 2;
                document.getElementById('trestbps').value = 130;
                document.getElementById('chol').value = 204;
                document.getElementById('fbs').value = 0;
                document.getElementById('restecg').value = 0;
                document.getElementById('thalach').value = 172;
                document.getElementById('exang').value = 0;
                document.getElementById('oldpeak').value = 0.2;
                document.getElementById('slope').value = 1;
                document.getElementById('ca').value = 0;
                document.getElementById('thal').value = 3;
            }
            ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'].forEach(id => {
                document.getElementById(id + '_val').innerText = document.getElementById(id).value;
            });
            predictRisk();
        }

        async function predictRisk() {
            const payload = {
                model_type: document.getElementById('model_type').value,
                features: [
                    parseFloat(document.getElementById('age').value),
                    parseFloat(document.getElementById('sex').value),
                    parseFloat(document.getElementById('cp').value),
                    parseFloat(document.getElementById('trestbps').value),
                    parseFloat(document.getElementById('chol').value),
                    parseFloat(document.getElementById('fbs').value),
                    parseFloat(document.getElementById('restecg').value),
                    parseFloat(document.getElementById('thalach').value),
                    parseFloat(document.getElementById('exang').value),
                    parseFloat(document.getElementById('oldpeak').value),
                    parseFloat(document.getElementById('slope').value),
                    parseFloat(document.getElementById('ca').value),
                    parseFloat(document.getElementById('thal').value)
                ]
            };

            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();

            const probPct = (data.probability * 100).toFixed(1) + '%';
            document.getElementById('risk-prob').innerText = probPct;
            document.getElementById('prob-bar').style.width = probPct;
            document.getElementById('res-model-name').innerText = data.model_name;

            const badge = document.getElementById('risk-badge');
            const status = document.getElementById('risk-status');
            const detail = document.getElementById('risk-detail');
            const bar = document.getElementById('prob-bar');

            if (data.prediction === 1) {
                badge.className = "p-4 rounded-xl text-white text-center mb-4 danger-gradient";
                status.innerText = "Cardiovascular Disease Detected";
                detail.innerText = "Elevated risk profile; cardiological screening advised";
                bar.className = "bg-red-500 h-3 rounded-full transition-all duration-500";
            } else {
                badge.className = "p-4 rounded-xl text-white text-center mb-4 safe-gradient";
                status.innerText = "Low Risk / Healthy Profile";
                detail.innerText = "Normal cardiac parameters within expected baseline";
                bar.className = "bg-emerald-500 h-3 rounded-full transition-all duration-500";
            }

            const container = document.getElementById('rules-container');
            container.innerHTML = '';
            if (data.top_rules && data.top_rules.length > 0) {
                data.top_rules.forEach(r => {
                    const el = document.createElement('div');
                    el.className = "p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs";
                    el.innerHTML = `
                        <div class="flex justify-between font-bold text-slate-700 mb-1">
                            <span>Rule #${r.rule_id}</span>
                            <span class="text-indigo-600">Firing: ${(r.firing_strength * 100).toFixed(1)}%</span>
                        </div>
                        <p class="text-slate-600 font-mono text-[11px] mb-1"><b>IF</b> (${r.conditions})</p>
                        <p class="font-semibold ${r.then_clause.includes('Elevated') ? 'text-red-600' : 'text-emerald-600'}"><b>THEN &rarr;</b> ${r.then_clause} (Sub-risk: ${(r.consequent_prob * 100).toFixed(1)}%)</p>
                    `;
                    container.appendChild(el);
                });
            } else {
                container.innerHTML = '<p class="text-xs text-slate-400 italic">Rule explanations not available for single algorithm ANN.</p>';
            }
        }

        window.onload = () => { predictRisk(); };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.json
    model_type = data.get('model_type', 'anfis_trap')
    raw_features = np.array(data.get('features'), dtype=np.float32)
    
    scaler = MODELS['scaler']
    features_scaled = scaler.transform(raw_features.reshape(1, -1))[0]
    
    if model_type == 'anfis_trap':
        model = MODELS['anfis_trap']
        name = "Hybrid ANFIS (Trapezoidal MF)"
    elif model_type == 'anfis_gauss':
        model = MODELS['anfis_gauss']
        name = "Hybrid ANFIS (Gaussian MF)"
    elif model_type == 'ga_fuzzy':
        model = MODELS['ga_fuzzy']
        name = "Genetic Algorithm + Fuzzy Logic (GA-FL)"
    else:
        model = MODELS['single_ann']
        name = "Single Algorithm (Standalone ANN)"
        
    prob = float(model.predict_proba(features_scaled.reshape(1, -1))[0, 1])
    pred = int(prob >= 0.5)
    
    top_rules = []
    if hasattr(model, 'explain_prediction'):
        explanation = model.explain_prediction(features_scaled, feature_names=FEATURE_NAMES, top_k=2)
        top_rules = explanation['top_rules']
        
    return jsonify({
        'model_name': name,
        'probability': prob,
        'prediction': pred,
        'top_rules': top_rules
    })

if __name__ == '__main__':
    init_models()
    port = 5000
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
