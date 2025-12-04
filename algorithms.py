import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            distances = self._calc_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = X[np.random.choice(n_samples)]
            
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.centroids = new_centroids
                break
                
            self.centroids = new_centroids
            
        return self.labels

    def _calc_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)


class KMedoids:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.medoids = None
        self.labels = None
        self.medoid_indices = None

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        self.medoid_indices = np.random.choice(n_samples, self.k, replace=False)
        self.medoids = X[self.medoid_indices]
        
        for _ in range(self.max_iters):
            distances = self._calc_distances(X, self.medoids)
            self.labels = np.argmin(distances, axis=1)
            
            new_medoid_indices = np.copy(self.medoid_indices)
            changed = False
            for i in range(self.k):
                cluster_indices = np.where(self.labels == i)[0]
                if len(cluster_indices) == 0:
                    continue
                    
                cluster_points = X[cluster_indices]
                curr_min_cost = float('inf')
                best_idx = -1
                
                # Optimization: compute distance matrix for cluster only once
                # But sticking to simple logic for clarity
                for idx in cluster_indices:
                    cost = np.sum(np.linalg.norm(cluster_points - X[idx], axis=1))
                    if cost < curr_min_cost:
                        curr_min_cost = cost
                        best_idx = idx
                
                if best_idx != self.medoid_indices[i]:
                    new_medoid_indices[i] = best_idx
                    changed = True
            
            self.medoid_indices = new_medoid_indices
            self.medoids = X[self.medoid_indices]
            
            if not changed:
                break
                
        return self.labels

    def _calc_distances(self, X, medoids):
        return np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)


class AGNES:
    def __init__(self, k=3, linkage='single'):
        self.k = k
        self.linkage = linkage
        self.labels = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        clusters = [[i] for i in range(n_samples)]
        
        while len(clusters) > self.k:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            # Simple O(N^3) implementation (or O(N^2) if dists cached, but here we recalculate)
            # This is fine for small datasets in TP context
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._calculate_cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            if merge_i == -1:
                break
                
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
            
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, point_indices in enumerate(clusters):
            for idx in point_indices:
                self.labels[idx] = cluster_id
                
        return self.labels

    def _calculate_cluster_distance(self, X, cluster1_idx, cluster2_idx):
        points1 = X[cluster1_idx]
        points2 = X[cluster2_idx]
        dists = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)
        
        if self.linkage == 'single':
            return np.min(dists)
        elif self.linkage == 'complete':
            return np.max(dists)
        elif self.linkage == 'average':
            return np.mean(dists)
        else:
            raise ValueError("Unknown linkage")


class DIANA:
    def __init__(self, k=3):
        self.k = k
        self.labels = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        clusters = [[i for i in range(n_samples)]]
        
        while len(clusters) < self.k:
            max_diameter = -1
            split_idx = -1
            
            for i, cluster_indices in enumerate(clusters):
                if len(cluster_indices) < 2:
                    continue
                diameter = self._calculate_diameter(X, cluster_indices)
                if diameter > max_diameter:
                    max_diameter = diameter
                    split_idx = i
            
            if split_idx == -1:
                break
                
            parent_cluster = clusters[split_idx]
            splinter, remainder = self._split_cluster(X, parent_cluster)
            
            clusters.pop(split_idx)
            clusters.append(remainder)
            clusters.append(splinter)
            
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, point_indices in enumerate(clusters):
            for idx in point_indices:
                self.labels[idx] = cluster_id
                
        return self.labels

    def _calculate_diameter(self, X, indices):
        if len(indices) < 2:
            return 0
        points = X[indices]
        dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        return np.max(dists)

    def _split_cluster(self, X, indices):
        B = set(indices)
        A = set()
        
        max_avg_dist = -1
        initial_splinter = -1
        
        for idx in B:
            others = list(B - {idx})
            if not others: 
                continue
            dist = np.mean(np.linalg.norm(X[idx] - X[others], axis=1))
            if dist > max_avg_dist:
                max_avg_dist = dist
                initial_splinter = idx
                
        if initial_splinter != -1:
            B.remove(initial_splinter)
            A.add(initial_splinter)
        else:
            return list(A), list(B)
            
        while True:
            max_diff = -float('inf')
            move_candidate = -1
            
            for idx in B:
                dist_to_A = np.mean(np.linalg.norm(X[idx] - X[list(A)], axis=1))
                others_in_B = list(B - {idx})
                if not others_in_B:
                    dist_to_B = 0
                else:
                    dist_to_B = np.mean(np.linalg.norm(X[idx] - X[others_in_B], axis=1))
                
                val = dist_to_B - dist_to_A # Prefer closer to A, further from B?
                # Actually standard DIANA: maximize (avg_dist(i, B\{i}) - avg_dist(i, A))
                # Because we want to move point that is far from B and close to A.
                
                if val > max_diff:
                    max_diff = val
                    move_candidate = idx
            
            if max_diff > 0:
                B.remove(move_candidate)
                A.add(move_candidate)
            else:
                break
                
        return list(A), list(B)


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if visited[i]:
                continue
                
            visited[i] = True
            neighbors = self._region_query(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self.labels[i] = cluster_id
                self._expand_cluster(X, neighbors, cluster_id, visited)
                cluster_id += 1
                
        return self.labels

    def _expand_cluster(self, X, neighbors, cluster_id, visited):
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + [n for n in new_neighbors if n not in neighbors]
            
            if self.labels[neighbor_idx] == -1:
                 self.labels[neighbor_idx] = cluster_id
            i += 1

    def _region_query(self, X, idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(dists <= self.eps)[0])


# --- Preprocessing ---

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.stats_ = None

    def fit(self, X):
        # X is expected to be a numpy array or pandas DataFrame values
        if self.strategy == 'mean':
            self.stats_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.stats_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X_new = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X_new[mask, i] = self.stats_[i]
        return X_new

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Handle zero range to avoid division by zero
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0 # Avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --- Model Selection ---

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# --- Classification ---

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            # Euclidean distance
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get k nearest neighbor indices
            k_indices = np.argsort(distances)[:self.k]
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = labels[np.argmax(counts)]
            predictions.append(most_common)
        return np.array(predictions)


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # Add small epsilon to var to prevent division by zero
        var += 1e-9 
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# --- Metrics ---

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_idx[true]][class_to_idx[pred]] += 1
        
    return matrix, classes

def precision_score(y_true, y_pred, average='macro'):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    precisions = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        if (tp + fp) > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
            
    if average == 'macro':
        return np.mean(precisions)
    return precisions # or handle 'weighted', etc.

def recall_score(y_true, y_pred, average='macro'):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    recalls = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if (tp + fn) > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)
            
    if average == 'macro':
        return np.mean(recalls)
    return recalls

def f1_score(y_true, y_pred, average='macro'):
    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)
    
    f1s = []
    for p, r in zip(precisions, recalls):
        if (p + r) > 0:
            f1s.append(2 * p * r / (p + r))
        else:
            f1s.append(0.0)
            
    if average == 'macro':
        return np.mean(f1s)
    return f1s