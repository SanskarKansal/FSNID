import numpy as np
import torch
from mine import mine


class firefly_selection:
    """Firefly algorithm for selecting exactly k informative features."""

    def __init__(self, L, C, n_fireflies=25, k=15, Tmax=10, gamma=1.0, alpha=0.05,
                 device=None):
        self.n_fireflies = max(1, int(n_fireflies))
        self.n_features = int(L.shape[1])
        self.k = min(max(1, int(k)), self.n_features)
        self.Tmax = max(1, int(Tmax))
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.L = L
        self.C = C
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.fireflies = self.generate_binary_population(self.n_fireflies, self.n_features, self.k)
        self.fitness = np.array([self.evaluate_classifier(firefly) for firefly in self.fireflies], dtype=float)

        sorted_indices = np.argsort(-self.fitness)
        self.fireflies = np.array(self.fireflies, dtype=int)[sorted_indices]
        self.fitness = self.fitness[sorted_indices]

    def generate_binary_population(self, x, max_length, target_sum):
        """Generate a population of binary feature masks with exactly target_sum active features."""
        if target_sum > max_length:
            target_sum = max_length

        arrays = []
        max_attempts = max(100, x * 25)
        attempts = 0

        while len(arrays) < x and attempts < max_attempts:
            arr = np.zeros(max_length, dtype=int)
            indices = np.random.choice(max_length, target_sum, replace=False)
            arr[indices] = 1
            arrays.append(arr)
            attempts += 1

        if len(arrays) < x:
            while len(arrays) < x:
                arr = np.zeros(max_length, dtype=int)
                indices = np.random.choice(max_length, target_sum, replace=False)
                arr[indices] = 1
                arrays.append(arr)

        return arrays

    def evaluate_classifier(self, feature_subset):
        """Estimate the MI-based quality of a feature subset."""
        selected_features = self.L[:, feature_subset.astype(bool)]
        if selected_features.size == 0:
            return -1e12

        mi_scores = mine(selected_features, self.C, 5).run()
        return float(np.mean(mi_scores, axis=1)[-1])

    def move_firefly(self, i, j):
        """Move firefly i toward firefly j, keeping the subset size fixed at k."""
        current = self.fireflies[i].copy()
        target = self.fireflies[j]

        distance = np.linalg.norm(current - target)
        beta = np.exp(-self.gamma * distance ** 2)

        candidate = current.copy()
        for idx in range(self.n_features):
            probability = beta * (target[idx] - current[idx]) + self.alpha * (np.random.rand() - 0.5)
            if np.random.rand() < max(0.0, min(1.0, probability)):
                candidate[idx] = 1 - candidate[idx]

        if candidate.sum() != self.k:
            if candidate.sum() > self.k:
                excess = np.where(candidate == 1)[0]
                remove_idx = np.random.choice(excess, size=candidate.sum() - self.k, replace=False)
                candidate[remove_idx] = 0
            else:
                missing = np.where(candidate == 0)[0]
                add_idx = np.random.choice(missing, size=self.k - candidate.sum(), replace=False)
                candidate[add_idx] = 1

        self.fireflies[i] = candidate

    def run_main(self):
        """Run the firefly optimization and return the best k-feature subset."""
        for t in range(1, self.Tmax + 1):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j and self.fitness[i] < self.fitness[j]:
                        self.move_firefly(i, j)
                        self.fitness[i] = self.evaluate_classifier(self.fireflies[i])

            sorted_indices = np.argsort(-self.fitness)
            self.fireflies = np.array(self.fireflies, dtype=int)[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            print(f"Iteration {t} completed. Best features: {self.fireflies[0]}")

        return self.fireflies[0]
