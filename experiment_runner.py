import os
import tempfile
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_and_preprocess_cleveland, get_train_test_split, FEATURE_NAMES
from anfis_model import ANFISClassifier
from single_model import SingleANNClassifier, get_conventional_models
from ga_fuzzy_model import GeneticFuzzyClassifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    
    return {
        'accuracy': float(acc),
        'specificity': float(spec),
        'precision': float(prec),
        'sensitivity': float(rec),
        'f1_score': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def run_held_out_comparison():
    print("=" * 75)
    print("1. EVALUATION ON HELD-OUT TEST SET (80:20 Split)")
    print("   Models: Hybrid ANFIS vs GA + Fuzzy Logic vs Single Algorithm ANN")
    print("=" * 75)
    
    df, X, y = load_and_preprocess_cleveland()
    X_train, X_test, y_train, y_test, scaler = get_train_test_split(X, y, test_size=0.20, random_state=42, scale_method='standard')
    
    # 1. Hybrid ANFIS Model (Trapezoidal - Paper logic)
    print("\n--> [1/4] Training Hybrid ANFIS Model (Trapezoidal MF)...")
    hybrid_anfis = ANFISClassifier(n_rules=8, mf_type='trapezoidal', lr=0.015, epochs=140, regularization=2.0, random_state=42)
    hybrid_anfis.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred_anfis = hybrid_anfis.predict(X_test)
    y_prob_anfis = hybrid_anfis.predict_proba(X_test)[:, 1]
    metrics_anfis = compute_metrics(y_test, y_pred_anfis, y_prob_anfis)
    
    # 2. Genetic Algorithm + Fuzzy Logic (GA-FL / GA-FIS)
    print("--> [2/4] Training Genetic Algorithm + Fuzzy Logic Model (GA-FL)...")
    ga_fuzzy = GeneticFuzzyClassifier(n_rules=8, population_size=40, generations=60, random_state=42, verbose=False)
    ga_fuzzy.fit(X_train, y_train)
    y_pred_ga = ga_fuzzy.predict(X_test)
    y_prob_ga = ga_fuzzy.predict_proba(X_test)[:, 1]
    metrics_ga = compute_metrics(y_test, y_pred_ga, y_prob_ga)
    
    # 3. Single Algorithm Model (Standalone ANN / MLP)
    print("--> [3/4] Training Single Algorithm Model (Standalone ANN / MLP)...")
    single_ann = SingleANNClassifier(n_features=X_train.shape[1], hidden_dim1=32, hidden_dim2=16,
                                     lr=0.01, epochs=140, random_state=42)
    single_ann.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred_ann = single_ann.predict(X_test)
    y_prob_ann = single_ann.predict_proba(X_test)[:, 1]
    metrics_ann = compute_metrics(y_test, y_pred_ann, y_prob_ann)
    
    # 4. Conventional ML Baselines
    print("--> [4/4] Training Conventional ML Models...")
    baseline_metrics = {}
    baseline_models = get_conventional_models(random_state=42)
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        baseline_metrics[name] = compute_metrics(y_test, pred, prob)
        
    all_models = {
        'Hybrid ANFIS (Proposed)': metrics_anfis,
        'Genetic Algorithm + Fuzzy Logic (GA-FL)': metrics_ga,
        'Single Algorithm (ANN/MLP)': metrics_ann,
        **baseline_metrics
    }
    
    results_table = []
    for model_name, m in all_models.items():
        results_table.append({
            'Model': model_name,
            'Accuracy (%)': f"{m['accuracy'] * 100:.2f}%",
            'Sensitivity (%)': f"{m['sensitivity'] * 100:.2f}%",
            'Specificity (%)': f"{m['specificity'] * 100:.2f}%",
            'Precision (%)': f"{m['precision'] * 100:.2f}%",
            'F1-Score (%)': f"{m['f1_score'] * 100:.2f}%",
            'AUC-ROC': f"{m['auc']:.4f}"
        })
    df_results = pd.DataFrame(results_table)
    print("\n" + df_results.to_string(index=False))
    
    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, 'model_comparison_metrics.json'), 'w') as f:
        json.dump(all_models, f, indent=2)
        
    # Plot 1: 3-Panel Confusion Matrices Comparison (Hybrid ANFIS vs GA-Fuzzy vs Single ANN)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cm_anfis = np.array(metrics_anfis['confusion_matrix'])
    cm_ga = np.array(metrics_ga['confusion_matrix'])
    cm_ann = np.array(metrics_ann['confusion_matrix'])
    
    def plot_cm(ax, cm, title, cmap):
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        classes = ['Healthy (0)', 'Disease (1)']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9.5)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=15, fontweight='bold')
        ax.set_ylabel('Actual Status', fontsize=10.5, fontweight='bold')
        ax.set_xlabel('Predicted Status', fontsize=10.5, fontweight='bold')
        
    plot_cm(axes[0], cm_anfis, f"Hybrid ANFIS (Acc: {metrics_anfis['accuracy']*100:.1f}%)", plt.cm.Blues)
    plot_cm(axes[1], cm_ga, f"GA + Fuzzy Logic (Acc: {metrics_ga['accuracy']*100:.1f}%)", plt.cm.Greens)
    plot_cm(axes[2], cm_ann, f"Single Algorithm ANN (Acc: {metrics_ann['accuracy']*100:.1f}%)", plt.cm.Oranges)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: GA Optimization Convergence Curve
    plt.figure(figsize=(8, 4.8))
    gens = ga_fuzzy.history['generation']
    plt.plot(gens, [a * 100 for a in ga_fuzzy.history['best_accuracy']], label='Best Population Accuracy (%)', color='#2ca02c', lw=2.5)
    plt.plot(gens, [f * 100 for f in ga_fuzzy.history['mean_fitness']], label='Mean Population Fitness', color='#1f77b4', lw=2, linestyle='--')
    plt.title('Genetic Algorithm Evolutionary Optimization of Fuzzy Rules', fontsize=12, fontweight='bold')
    plt.xlabel('GA Generation', fontsize=11)
    plt.ylabel('Fitness / Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ga_convergence_curve.png'), dpi=300)
    plt.close()
    
    # Plot 3: Training & Testing Curves for Hybrid ANFIS (Paper Figure 5 replication)
    epochs_range = range(1, len(hybrid_anfis.history['train_acc']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1.plot(epochs_range, hybrid_anfis.history['train_acc'], label='Train Acc', color='#1f77b4', lw=2.5)
    ax1.plot(epochs_range, hybrid_anfis.history['val_acc'], label='Test Acc', color='#d62728', lw=2.5)
    ax1.set_title('(a) Training and Testing Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0.45, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=11)
    
    ax2.plot(epochs_range, hybrid_anfis.history['train_loss'], label='Train Loss', color='#1f77b4', lw=2.5)
    ax2.plot(epochs_range, hybrid_anfis.history['val_loss'], label='Test Loss', color='#d62728', lw=2.5)
    ax2.set_title('(b) Training and Testing Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_ylim(-0.05, 1.8)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.suptitle('Hybrid ANFIS Learning Curves (Procedia CS 2025 Fig. 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves_anfis.png'), dpi=300)
    plt.close()
    
    # Plot 4: Comparative Bar Chart (Including GA-Fuzzy)
    models_display = ['Hybrid ANFIS\n(Proposed)', 'Random\nForest', 'Logistic\nRegression', 'GA + Fuzzy\nLogic (GA-FL)', 'Support\nVector Mach.', 'Single ANN\n(1-Algorithm)', 'Decision\nTree']
    acc_values = [
        metrics_anfis['accuracy'] * 100,
        baseline_metrics['Random Forest']['accuracy'] * 100,
        baseline_metrics['Logistic Regression']['accuracy'] * 100,
        metrics_ga['accuracy'] * 100,
        baseline_metrics['Support Vector Machine']['accuracy'] * 100,
        metrics_ann['accuracy'] * 100,
        baseline_metrics['Decision Tree']['accuracy'] * 100
    ]
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#17becf', '#9467bd', '#ff7f0e', '#8c564b']
    
    plt.figure(figsize=(11.5, 6))
    bars = plt.bar(models_display, acc_values, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    plt.ylim(50, 100)
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison: Hybrid ANFIS vs GA-Fuzzy vs Single ANN & Baselines', fontsize=13, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 1.0, f"{h:.2f}%", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_comparison_bar.png'), dpi=300)
    plt.close()
    
    # Plot 5: ROC Curves Comparison
    plt.figure(figsize=(8.5, 6.5))
    fpr_anfis, tpr_anfis, _ = roc_curve(y_test, y_prob_anfis)
    fpr_ga, tpr_ga, _ = roc_curve(y_test, y_prob_ga)
    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
    plt.plot(fpr_anfis, tpr_anfis, label=f"Hybrid ANFIS (AUC = {metrics_anfis['auc']:.4f})", color='#1f77b4', lw=3)
    plt.plot(fpr_ga, tpr_ga, label=f"GA + Fuzzy Logic (AUC = {metrics_ga['auc']:.4f})", color='#2ca02c', lw=2.2, linestyle='-.')
    plt.plot(fpr_ann, tpr_ann, label=f"Single ANN (AUC = {metrics_ann['auc']:.4f})", color='#ff7f0e', lw=2, linestyle='--')
    
    for name, m in baseline_metrics.items():
        if name in ['Random Forest', 'Logistic Regression']:
            model = baseline_models[name]
            p = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {m['auc']:.4f})", lw=1.5, alpha=0.7)
            
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_comparison.png'), dpi=300)
    plt.close()
    
    return hybrid_anfis, ga_fuzzy, single_ann, all_models, scaler, X_test, y_test

def run_membership_function_experiments():
    print("\n" + "=" * 75)
    print("2. EXPERIMENTATION WITH DIFFERENT MEMBERSHIP FUNCTIONS")
    print("=" * 75)
    
    df, X, y = load_and_preprocess_cleveland()
    X_train, X_test, y_train, y_test, _ = get_train_test_split(X, y, test_size=0.20, random_state=42, scale_method='standard')
    
    mf_types = ['trapezoidal', 'triangular', 'gaussian', 'bell']
    mf_results = {}
    
    for mf in mf_types:
        print(f"--> Training ANFIS with '{mf}' membership function...")
        model = ANFISClassifier(n_rules=8, mf_type=mf, lr=0.015, epochs=140, regularization=2.0, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, pred, prob)
        mf_results[mf] = metrics
        
    print("\n--- Membership Function Comparison Results ---")
    mf_table = []
    for mf, m in mf_results.items():
        mf_table.append({
            'Membership Function': mf.capitalize(),
            'Accuracy (%)': f"{m['accuracy'] * 100:.2f}%",
            'Sensitivity (%)': f"{m['sensitivity'] * 100:.2f}%",
            'Specificity (%)': f"{m['specificity'] * 100:.2f}%",
            'Precision (%)': f"{m['precision'] * 100:.2f}%",
            'F1-Score (%)': f"{m['f1_score'] * 100:.2f}%",
            'AUC-ROC': f"{m['auc']:.4f}"
        })
    df_mf = pd.DataFrame(mf_table)
    print(df_mf.to_string(index=False))
    
    with open(os.path.join(RESULTS_DIR, 'membership_functions_metrics.json'), 'w') as f:
        json.dump(mf_results, f, indent=2)
        
    plot_membership_curves()
    return mf_results

def plot_membership_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Age (Young, Adult, Old) - Figure 7
    age_x = np.linspace(20, 85, 300)
    mu_young = np.clip(np.minimum(1.0, (45 - age_x) / 12.0), 0.0, 1.0)
    mu_adult = np.clip(np.minimum((age_x - 38) / 8.0, (65 - age_x) / 9.0), 0.0, 1.0)
    mu_old = np.clip((age_x - 58) / 12.0, 0.0, 1.0)
    
    axes[0].plot(age_x, mu_young, 'b-', lw=2.5, label='Young')
    axes[0].plot(age_x, mu_adult, 'g-', lw=2.5, label='Adult')
    axes[0].plot(age_x, mu_old, 'r-', lw=2.5, label='Old')
    axes[0].set_title('Membership Function: Age (Fig. 7)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Age (Years)', fontsize=11)
    axes[0].set_ylabel('Membership Degree (\u03bc)', fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='center right')
    
    # 2. Chest Pain (Typical, Atypical, Non-anginal, Asymptomatic) - Figure 8
    cp_x = np.linspace(1, 4, 300)
    mu_cp1 = np.clip((1.8 - cp_x) / 0.8, 0.0, 1.0)
    mu_cp2 = np.clip(np.minimum((cp_x - 1.2) / 0.6, (2.6 - cp_x) / 0.6), 0.0, 1.0)
    mu_cp3 = np.clip(np.minimum((cp_x - 2.2) / 0.6, (3.5 - cp_x) / 0.6), 0.0, 1.0)
    mu_cp4 = np.clip((cp_x - 3.0) / 0.8, 0.0, 1.0)
    
    axes[1].plot(cp_x, mu_cp1, 'b-', lw=2.5, label='Typical Angina')
    axes[1].plot(cp_x, mu_cp2, 'm-', lw=2.5, label='Atypical Angina')
    axes[1].plot(cp_x, mu_cp3, 'g-', lw=2.5, label='Non-anginal')
    axes[1].plot(cp_x, mu_cp4, 'r-', lw=2.5, label='Asymptomatic')
    axes[1].set_title('Membership Function: Chest Pain (Fig. 8)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Chest Pain Type (1-4)', fontsize=11)
    axes[1].set_ylabel('Membership Degree (\u03bc)', fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='center right')
    
    # 3. Maximum Heart Rate (Low, Normal, High)
    hr_x = np.linspace(70, 210, 300)
    mu_hr_low = np.clip((125 - hr_x) / 25.0, 0.0, 1.0)
    mu_hr_norm = np.clip(np.minimum((hr_x - 110) / 25.0, (170 - hr_x) / 25.0), 0.0, 1.0)
    mu_hr_high = np.clip((hr_x - 150) / 30.0, 0.0, 1.0)
    
    axes[2].plot(hr_x, mu_hr_low, 'b-', lw=2.5, label='Low HR')
    axes[2].plot(hr_x, mu_hr_norm, 'g-', lw=2.5, label='Normal HR')
    axes[2].plot(hr_x, mu_hr_high, 'r-', lw=2.5, label='High HR')
    axes[2].set_title('Membership Function: Max Heart Rate', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Thalach (bpm)', fontsize=11)
    axes[2].set_ylabel('Membership Degree (\u03bc)', fontsize=11)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'membership_functions.png'), dpi=300)
    plt.close()

def run_10_fold_cross_validation():
    print("\n" + "=" * 75)
    print("3. 10-FOLD CROSS-VALIDATION (Procedia CS Section 4.1)")
    print("=" * 75)
    
    df, X, y = load_and_preprocess_cleveland()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    anfis_accs, ga_accs, ann_accs = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Hybrid ANFIS
        anfis = ANFISClassifier(n_rules=8, mf_type='trapezoidal', lr=0.015, epochs=140, regularization=2.0, random_state=42)
        anfis.fit(X_tr, y_tr)
        p_anfis = anfis.predict(X_val)
        acc_anfis = accuracy_score(y_val, p_anfis)
        anfis_accs.append(acc_anfis)
        
        # GA + Fuzzy Logic
        ga_fuzz = GeneticFuzzyClassifier(n_rules=8, population_size=30, generations=35, random_state=42)
        ga_fuzz.fit(X_tr, y_tr)
        p_ga = ga_fuzz.predict(X_val)
        acc_ga = accuracy_score(y_val, p_ga)
        ga_accs.append(acc_ga)
        
        # Single ANN
        ann = SingleANNClassifier(n_features=X.shape[1], hidden_dim1=32, hidden_dim2=16,
                                  lr=0.01, epochs=140, random_state=42)
        ann.fit(X_tr, y_tr)
        p_ann = ann.predict(X_val)
        acc_ann = accuracy_score(y_val, p_ann)
        ann_accs.append(acc_ann)
        
        print(f"Fold {fold+1:02d}/10: Hybrid ANFIS = {acc_anfis*100:.2f}% | GA-Fuzzy = {acc_ga*100:.2f}% | Single ANN = {acc_ann*100:.2f}%")
        
    print("-" * 75)
    print("10-Fold CV Mean Accuracy Summary:")
    print(f"  * Hybrid ANFIS:            {np.mean(anfis_accs)*100:.2f}% (+/- {np.std(anfis_accs)*100:.2f}%)")
    print(f"  * GA + Fuzzy Logic (GA-FL): {np.mean(ga_accs)*100:.2f}% (+/- {np.std(ga_accs)*100:.2f}%)")
    print(f"  * Single ANN (Model):      {np.mean(ann_accs)*100:.2f}% (+/- {np.std(ann_accs)*100:.2f}%)")
    print("-" * 75)

def demonstrate_fuzzy_rule_explanations(hybrid_anfis, ga_fuzzy, scaler, X_test, y_test):
    print("\n" + "=" * 75)
    print("4. FUZZY RULE EXPLANATION ENGINE (Table 1 Replication)")
    print("=" * 75)
    
    test_probs = hybrid_anfis.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(test_probs))
    low_risk_idx = int(np.argmin(test_probs))
    
    for case_label, sample_idx in [("HIGH RISK PATIENT CASE", high_risk_idx), ("HEALTHY / LOW RISK CASE", low_risk_idx)]:
        print(f"\n--- {case_label} ---")
        sample_scaled = X_test[sample_idx]
        actual_status = "Disease (1)" if y_test[sample_idx] == 1 else "Healthy (0)"
        
        exp_anfis = hybrid_anfis.explain_prediction(sample_scaled, feature_names=FEATURE_NAMES, top_k=2)
        exp_ga = ga_fuzzy.explain_prediction(sample_scaled, feature_names=FEATURE_NAMES, top_k=2)
        
        print(f"Ground Truth Status:    {actual_status}")
        print(f"Hybrid ANFIS Predict:   {exp_anfis['prediction']} ({exp_anfis['predicted_probability']*100:.2f}%)")
        print(f"GA-Fuzzy Predict:       {exp_ga['prediction']} ({exp_ga['predicted_probability']*100:.2f}%)")
        print("Active Triggered Rules (Hybrid ANFIS):")
        for r in exp_anfis['top_rules']:
            print(f"  * IF ({r['conditions']}) -> {r['then_clause']} [Firing: {r['firing_strength']*100:.1f}%]")

if __name__ == '__main__':
    hybrid_anfis, ga_fuzzy, single_ann, all_metrics, scaler, X_test, y_test = run_held_out_comparison()
    mf_results = run_membership_function_experiments()
    run_10_fold_cross_validation()
    demonstrate_fuzzy_rule_explanations(hybrid_anfis, ga_fuzzy, scaler, X_test, y_test)
    print("\nAll experiments, comparisons, and figure generations completed successfully!")
