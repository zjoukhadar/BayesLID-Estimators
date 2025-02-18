import numpy as np
from scipy.spatial.distance import cdist

def get_knn_distances(features, k):
    """
    Given a feature matrix (num_samples x feature_dim), compute pairwise
    Euclidean distances and return a list where each element is an array of the
    k nearest neighbor distances for that sample (excluding the self-distance).
    """
    # Convert to numpy array if tensor
    if hasattr(features, 'cpu'):
        features = features.cpu().numpy()
    distances = cdist(features, features, metric='euclidean')
    knn_list = []
    num_samples = distances.shape[0]
    for i in range(num_samples):
        # Remove self-distance (zero) and sort
        dists = np.delete(distances[i], i)
        sorted_dists = np.sort(dists)
        knn = sorted_dists[:k] if len(sorted_dists) >= k else sorted_dists
        knn_list.append(knn)
    return knn_list