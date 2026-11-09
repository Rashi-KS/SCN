import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET_NAME = 'target'
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male, 0 = female)',
    'cp': 'Chest Pain Type (1: typical angina, 2: atypical angina, 3: non-anginal, 4: asymptomatic)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)',
    'restecg': 'Resting ECG results (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)',
    'thalach': 'Maximum Heart Rate Achieved (bpm)',
    'exang': 'Exercise Induced Angina (1 = yes, 0 = no)',
    'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
    'slope': 'Slope of the Peak Exercise ST Segment (1: up, 2: flat, 3: down)',
    'ca': 'Number of Major Vessels (0-3) Colored by Fluoroscopy',
    'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)'
}

def load_and_preprocess_cleveland(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'processed.cleveland.data')
    
    # Read raw dataset
    df = pd.read_csv(data_path, header=None, names=ALL_COLUMNS, na_values='?')
    
    # Handle missing values: median for numerical / mode for categorical
    for col in FEATURE_NAMES:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Binarize target: 0 = Healthy / No CVD, >0 = CVD Present (1)
    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)
    
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[TARGET_NAME].values.astype(np.float32)
    
    return df, X, y

def get_train_test_split(X, y, test_size=0.20, random_state=42, scale_method='standard'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if scale_method == 'standard':
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    else:
        scaler = None
        
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
