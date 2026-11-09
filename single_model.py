import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class StandaloneANNNet(nn.Module):
    """
    Standard Single Algorithm: Multi-Layer Perceptron (Artificial Neural Network)
    without fuzzy membership or rule inference layers.
    """
    def __init__(self, n_features=13, hidden_dim1=32, hidden_dim2=16):
        super(StandaloneANNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

class SingleANNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the Single Algorithm (ANN) model.
    """
    def __init__(self, n_features=13, hidden_dim1=32, hidden_dim2=16,
                 lr=0.01, epochs=140, weight_decay=1e-4, random_state=42, verbose=False):
        self.n_features = n_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = {'train_loss': [], 'train_acc': []}
        
    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        self.model = StandaloneANNNet(
            n_features=n_features,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        self.history = {'train_loss': [], 'train_acc': []}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            self.history['val_loss'] = []
            self.history['val_acc'] = []
            
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
            probs = torch.sigmoid(logits).detach().numpy()
            preds = (probs >= 0.5).astype(int)
            acc = np.mean(preds == y)
            self.history['train_loss'].append(loss.item())
            self.history['train_acc'].append(acc)
            
            if eval_set is not None:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(X_val_tensor)
                    v_loss = criterion(v_logits, y_val_tensor).item()
                    v_probs = torch.sigmoid(v_logits).numpy()
                    v_preds = (v_probs >= 0.5).astype(int)
                    v_acc = np.mean(v_preds == y_val)
                    self.history['val_loss'].append(v_loss)
                    self.history['val_acc'].append(v_acc)
                self.model.train()
                
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            p1 = torch.sigmoid(logits).numpy()
            p0 = 1.0 - p1
        return np.column_stack([p0, p1])
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def get_conventional_models(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Support Vector Machine': SVC(probability=True, random_state=random_state, kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=4),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
