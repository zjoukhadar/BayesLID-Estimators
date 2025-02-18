import numpy as np
from scipy.spatial.distance import cdist

def get_knn_distances(features, k):
    """
    Compute the k-nearest neighbor distances for each sample in the feature matrix.

    Parameters:
      features : array-like of shape (num_samples, feature_dim)
          The feature representations (e.g., from a CNN).
      k : int
          The number of nearest neighbors to retrieve (excluding the sample itself).

    Returns:
      A list of arrays, each containing the k sorted nearest neighbor distances for the corresponding sample.
    """
    # Ensure features are in numpy format.
    if hasattr(features, 'cpu'):
        features = features.cpu().numpy()
    # Compute pairwise Euclidean distances.
    distances = cdist(features, features, metric='euclidean')
    knn_list = []
    num_samples = distances.shape[0]
    for i in range(num_samples):
        # Exclude the self-distance (zero) then sort.
        dists = np.delete(distances[i], i)
        sorted_dists = np.sort(dists)
        knn = sorted_dists[:k] if len(sorted_dists) >= k else sorted_dists
        knn_list.append(knn)
    return knn_list