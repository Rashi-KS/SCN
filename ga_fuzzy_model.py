import os
import tempfile
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class GeneticFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """
    Genetic Algorithm + Fuzzy Inference System (GA-FL / GA-FIS) Classifier:
    
    Combines evolutionary computation (Genetic Algorithms) with Fuzzy Logic
    inference for cardiovascular disease diagnosis, as studied in:
    - Aghamohammadi et al. (ICCS 2019 / Procedia CS 2025 Ref [5])
    - Santhanam & Ephzibah (2015, Ref [8])
    - Reddy et al. (Evolutionary Intelligence 2020, Ref [6])
    
    The Genetic Algorithm optimizes:
    1. Fuzzy membership centers & spreads across clinical risk features.
    2. Rule activation weights and consequent class determinations.
    3. Feature salience weights to suppress noise.
    """
    def __init__(self, n_rules=8, population_size=40, generations=60,
                 crossover_rate=0.8, mutation_rate=0.15, tournament_size=3,
                 elitism=2, random_state=42, verbose=False):
        self.n_rules = n_rules
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.random_state = random_state
        self.verbose = verbose
        
        self.best_chromosome = None
        self.best_fitness = -1.0
        self.history = {'generation': [], 'best_fitness': [], 'mean_fitness': []}
        self.centers = None
        self.sigmas = None
        
    def _initialize_population(self, X, y):
        """
        Encodes candidate fuzzy systems into real-valued chromosomes:
        Chromosome structure:
        - Part 1: Feature weights (n_features)
        - Part 2: Rule consequent risk scores (n_rules)
        - Part 3: Rule threshold / bias (1)
        """
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.n_features = n_features
        
        # Heuristic initialization of centers using KMeans
        km = KMeans(n_clusters=self.n_rules, random_state=self.random_state, n_init=10).fit(X)
        self.centers = km.cluster_centers_
        self.sigmas = np.array([np.std(X[km.labels_ == i], axis=0) + 0.35 for i in range(self.n_rules)])
        
        # Chromosome length: n_features (weights) + n_rules (consequents) + 1 (bias)
        self.chrom_len = n_features + self.n_rules + 1
        
        population = []
        for i in range(self.population_size):
            # Feature weights in [0.2, 1.5]
            feat_w = np.random.uniform(0.3, 1.2, size=n_features)
            # Rule consequents in [-2.0, 2.0]
            consequents = np.random.uniform(-1.5, 1.5, size=self.n_rules)
            # Bias in [-1.0, 1.0]
            bias = np.random.uniform(-0.5, 0.5, size=1)
            
            # Seed the first individual with cluster label proportions
            if i == 0:
                for r in range(self.n_rules):
                    mask = (km.labels_ == r)
                    if np.sum(mask) > 0:
                        consequents[r] = 2.0 * (np.mean(y[mask]) - 0.5)
                        
            chrom = np.concatenate([feat_w, consequents, bias])
            population.append(chrom)
            
        return np.array(population)
        
    def _evaluate_chromosome(self, chrom, X, y):
        """
        Computes classification fitness of a chromosome on the dataset.
        """
        probs = self._predict_proba_from_chrom(chrom, X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        # Add smooth cross-entropy reward to break ties and guide genetic gradient
        eps = 1e-7
        bce = -np.mean(y * np.log(np.clip(probs, eps, 1-eps)) + (1-y) * np.log(np.clip(1-probs, eps, 1-eps)))
        fitness = acc + 0.05 * (1.0 / (1.0 + bce))
        return float(fitness), float(acc)
        
    def _predict_proba_from_chrom(self, chrom, X):
        feat_w = chrom[:self.n_features]
        consequents = chrom[self.n_features : self.n_features + self.n_rules]
        bias = chrom[-1]
        
        # Weighted fuzzy membership distances
        # dists shape: [n_samples, n_rules]
        weighted_diff = (X[:, None, :] - self.centers[None, :, :]) * feat_w[None, None, :]
        dists = np.linalg.norm(weighted_diff, axis=2)
        s = np.mean(self.sigmas)
        
        # Trapezoidal / Gaussian fuzzy membership
        core_dist = np.maximum(0.0, dists - 0.35 * s)
        w = np.exp(-0.5 * (core_dist / (s + 1e-6))**2)
        w_norm = w / (np.sum(w, axis=1, keepdims=True) + 1e-7)
        
        # Defuzzification: y = sum(w_norm * consequent) + bias
        logits = np.sum(w_norm * consequents, axis=1) + bias
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
        
    def fit(self, X, y, eval_set=None):
        np.random.seed(self.random_state)
        population = self._initialize_population(X, y)
        self.classes_ = np.unique(y)
        
        self.history = {'generation': [], 'best_fitness': [], 'best_accuracy': [], 'mean_fitness': []}
        
        for gen in range(self.generations):
            # Evaluate fitness of all individuals
            fitness_scores = []
            acc_scores = []
            for chrom in population:
                fit_val, acc_val = self._evaluate_chromosome(chrom, X, y)
                fitness_scores.append(fit_val)
                acc_scores.append(acc_val)
                
            fitness_scores = np.array(fitness_scores)
            acc_scores = np.array(acc_scores)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = population[best_idx].copy()
                
            self.history['generation'].append(gen + 1)
            self.history['best_fitness'].append(float(np.max(fitness_scores)))
            self.history['best_accuracy'].append(float(np.max(acc_scores)))
            self.history['mean_fitness'].append(float(np.mean(fitness_scores)))
            
            if self.verbose and (gen + 1) % 15 == 0:
                print(f"Gen [{gen+1}/{self.generations}] Best Acc: {self.history['best_accuracy'][-1]*100:.2f}% | Mean Fit: {self.history['mean_fitness'][-1]:.4f}")
                
            # Selection: Elitism
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population = [population[i].copy() for i in sorted_indices[:self.elitism]]
            
            # Tournament Selection + Crossover + Mutation
            while len(new_population) < self.population_size:
                # Tournament 1
                tourn1 = np.random.choice(self.population_size, size=self.tournament_size, replace=False)
                parent1 = population[tourn1[np.argmax(fitness_scores[tourn1])]]
                # Tournament 2
                tourn2 = np.random.choice(self.population_size, size=self.tournament_size, replace=False)
                parent2 = population[tourn2[np.argmax(fitness_scores[tourn2])]]
                
                # Crossover (Arithmetic / SBX blend)
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.uniform(0.1, 0.9, size=self.chrom_len)
                    child1 = alpha * parent1 + (1.0 - alpha) * parent2
                    child2 = alpha * parent2 + (1.0 - alpha) * parent1
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                    
                # Mutation (Gaussian perturbation)
                for child in [child1, child2]:
                    if np.random.rand() < self.mutation_rate:
                        mut_mask = np.random.rand(self.chrom_len) < 0.25
                        noise = np.random.normal(0, 0.25, size=self.chrom_len)
                        child[mut_mask] += noise[mut_mask]
                    if len(new_population) < self.population_size:
                        new_population.append(child)
                        
            population = np.array(new_population)
            
        return self
        
    def predict_proba(self, X):
        return self._predict_proba_from_chrom(self.best_chromosome, X)
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
        
    def explain_prediction(self, sample_x, feature_names=None, top_k=2):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(sample_x))]
            
        sample_x_2d = np.array(sample_x).reshape(1, -1)
        prob = self.predict_proba(sample_x_2d)[0, 1]
        pred_label = "Cardiovascular Disease Detected" if prob >= 0.5 else "Healthy / No CVD"
        
        feat_w = self.best_chromosome[:self.n_features]
        consequents = self.best_chromosome[self.n_features : self.n_features + self.n_rules]
        bias = self.best_chromosome[-1]
        
        weighted_diff = (sample_x_2d[:, None, :] - self.centers[None, :, :]) * feat_w[None, None, :]
        dists = np.linalg.norm(weighted_diff, axis=2)[0]
        s = np.mean(self.sigmas)
        
        core_dist = np.maximum(0.0, dists - 0.35 * s)
        w = np.exp(-0.5 * (core_dist / (s + 1e-6))**2)
        w_norm = w / (np.sum(w) + 1e-7)
        
        top_rule_indices = np.argsort(w_norm)[::-1][:top_k]
        explanations = []
        for r_idx in top_rule_indices:
            w_i = w_norm[r_idx]
            local_score = consequents[r_idx] + bias
            local_prob = 1.0 / (1.0 + np.exp(-np.clip(local_score, -15, 15)))
            
            conditions = []
            for f_idx, feat in enumerate(feature_names[:6]):
                val = sample_x[f_idx]
                c_val = self.centers[r_idx, f_idx]
                s_val = self.sigmas[r_idx, f_idx]
                level = "High" if val > c_val + 0.3 * s_val else ("Low" if val < c_val - 0.3 * s_val else "Normal")
                conditions.append(f"{feat} is {level}")
                
            rule_then = "Elevated CVD Risk" if local_prob >= 0.5 else "Low CVD Risk"
            explanations.append({
                'rule_id': int(r_idx + 1),
                'firing_strength': float(w_i),
                'consequent_prob': float(local_prob),
                'conditions': " AND ".join(conditions),
                'then_clause': rule_then
            })
            
        return {
            'predicted_probability': float(prob),
            'prediction': pred_label,
            'top_rules': explanations
        }
