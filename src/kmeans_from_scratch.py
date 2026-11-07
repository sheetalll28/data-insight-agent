import numpy as np
import csv
import os

class KMeansFromScratch:
    """
    A K-Means Clustering algorithm built entirely from scratch using NumPy.
    Finds hidden patterns by mathematically grouping similar rows together.

    - K-Means++ initialisation (spreads the starting centroids out, which
      gives faster convergence and better optima than uniform random picks).
    - n_init restarts; the run with the lowest WCSS wins.
    - Vectorised squared-distance computation.
    - Empty clusters are re-seeded with the point farthest from its centroid.
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4, n_init=10, seed=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol # Tolerance for convergence
        self.n_init = n_init # How many times we restart to avoid Local Minima
        self.seed = seed
        self.centroids = None
        self.wcss = float('inf')
        self.n_iter_ = 0
        
    def fit(self, X):
        """
        Trains the clustering model on the dataset X.
        Runs multiple times to avoid getting stuck in a bad random start.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if self.k < 1 or self.k > X.shape[0]:
            raise ValueError(f"k={self.k} must be between 1 and the number of rows ({X.shape[0]})")

        rng = np.random.default_rng(self.seed)
        best_centroids = None
        best_labels = None
        best_wcss = float('inf')
        
        # We loop n_init times and keep the result with the lowest WCSS
        for init in range(self.n_init):
            # 1. Initialization: K-Means++
            self.centroids = self._kmeans_plus_plus(X, rng)
            
            for i in range(self.max_iters):
                # 2. Assignment Step
                distances = self._calculate_distances(X)
                labels = np.argmin(distances, axis=1)
                
                # 3. Update Step
                new_centroids = np.zeros_like(self.centroids)
                for j in range(self.k):
                    members = labels == j
                    if np.any(members):
                        new_centroids[j] = X[members].mean(axis=0)
                    else:
                        # Empty cluster: move it to the worst-served point
                        far = np.argmax(distances[np.arange(len(X)), labels])
                        new_centroids[j] = X[far]
                        
                # 4. Check for Convergence
                shift = np.sum(np.abs(new_centroids - self.centroids))
                self.centroids = new_centroids
                if shift < self.tol:
                    break

            # Final assignment against the final centroids
            labels = np.argmin(self._calculate_distances(X), axis=1)
            current_wcss = self._calculate_wcss(X, labels)
            
            # If this random start resulted in tighter clusters, save it!
            if current_wcss < best_wcss:
                best_wcss = current_wcss
                best_centroids = np.copy(self.centroids)
                best_labels = np.copy(labels)
                self.n_iter_ = i + 1
                
        self.centroids = best_centroids
        self.wcss = best_wcss
        return best_labels

    def predict(self, X):
        return np.argmin(self._calculate_distances(np.asarray(X, dtype=float)), axis=1)

    def _kmeans_plus_plus(self, X, rng):
        """
        K-Means++ seeding: the first centroid is a random point; each next one
        is drawn with probability proportional to its squared distance from
        the nearest centroid chosen so far.
        """
        n = X.shape[0]
        centroids = [X[rng.integers(n)]]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
        for _ in range(1, self.k):
            total = closest_sq.sum()
            if total == 0:   # all remaining points coincide with a centroid
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=closest_sq / total)
            centroids.append(X[idx])
            closest_sq = np.minimum(closest_sq, np.sum((X - X[idx]) ** 2, axis=1))
        return np.array(centroids, dtype=float)
        
    def _calculate_distances(self, X):
        """
        Calculates the Euclidean distance from every point to every centroid.
        This is the Pythagorean theorem (a^2 + b^2 = c^2) scaled up to N-dimensions!
        Vectorised via broadcasting: (n, 1, d) - (1, k, d) -> (n, k).
        """
        diff = X[:, None, :] - self.centroids[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
        
    def _calculate_wcss(self, X, labels):
        """
        Calculates the Within-Cluster Sum of Squares (WCSS).
        This adds up how "tight" each cluster is. 
        Used by the Elbow Method to choose K without overfitting.
        """
        return float(np.sum((X - self.centroids[labels]) ** 2))

def load_and_prep_data():
    """
    Loads the real dataset and preps it for clustering.
    We normalize everything to 0-1 so that one large feature doesn't dominate the distance calculations.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    messy_path = os.path.join(base_dir, 'data', 'processed', 'messy_data.csv')
    
    with open(messy_path, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        
    numeric_cols = [
        'Symptom Severity (1-10)', 'Mood Score (1-10)', 
        'Sleep Quality (1-10)', 'Physical Activity (hrs/week)'
    ]
    
    X = []
    for row in data:
        features = []
        for col in numeric_cols:
            if row[col].lstrip('-').replace('.', '', 1).isdigit():
                val = float(row[col])
                # Quick outlier clip so the clusters don't get skewed by the 999s
                val = max(1.0, min(10.0, val))
                features.append(val)
            else:
                features.append(5.0) # Impute missing with median
        X.append(features)
        
    X = np.array(X)
    
    # Min-Max Normalization (0.0 to 1.0)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
    
    return X_norm

def main():
    print("Starting Pattern Discovery (K-Means) Agent...")
    X = load_and_prep_data()
    print(f"Prepared {X.shape[0]} patients for clustering.")
    
    print("\nRunning The Elbow Method to find the mathematically perfect K...")
    print("-----------------------------------------------------------------")
    
    wcss_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        # n_init=5 for speed, but enough to avoid local minima
        kmeans = KMeansFromScratch(k=k, n_init=5) 
        kmeans.fit(X)
        wcss_scores.append(kmeans.wcss)
        print(f"K = {k:2d} | WCSS (Inertia) = {kmeans.wcss:.2f}")
        
    print("-----------------------------------------------------------------")
    print("\nLook for the 'Elbow' in the scores above!")
    print("Find the K where the WCSS stops dropping massively and starts leveling out.")
    print("That is the mathematically optimal number of clusters for this dataset (avoiding overfitting!).")

if __name__ == "__main__":
    main()
