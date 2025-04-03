from sklearn.preprocessing import normalize
import sklearn
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import scipy


class DiffusionCondensation:
    def __init__(self,
                 k=5,
                 alpha=2,
                 epsilon_scale=0.99,
                 merge_threshold=1e-15,
                 min_clusters=5,
                 max_iterations=100,
                 t=1,
                 data_dependent_epsilon=True,
                 symmetric_kernel=False,
                 bandwidth_norm="max"):

        self.k = k
        self.alpha = alpha
        self.epsilon_scale = epsilon_scale
        self.merge_threshold = merge_threshold
        self.min_clusters = min_clusters
        self.max_iterations = max_iterations
        self.t = t
        self.data_dependent_epsilon = data_dependent_epsilon
        self.symmetric_kernel = symmetric_kernel
        self.bandwidth_norm = bandwidth_norm # can be "max" or "l1"
        self.cluster_function = None
        self.epsilon = None
        self.labels_ = None
        self.cluster_tree = None

    def diffusion_operator(self, data):
        """Create diffusion operator"""
        # Generate the k-NN graph adjacency matrix using Scikit-learn
        knn_graph_adjacency = kneighbors_graph(data, n_neighbors=self.k, mode='distance')
        knn_graph_adjacency = normalize(knn_graph_adjacency, norm=self.bandwidth_norm, axis=1, copy=False)

        knn_graph_adjacency = -knn_graph_adjacency.power(self.alpha)
        np.exp(knn_graph_adjacency.data, out=knn_graph_adjacency.data)

        if self.symmetric_kernel:
            knn_graph_adjacency = 0.5*(knn_graph_adjacency + knn_graph_adjacency.T)
        knn_graph_adjacency = normalize(knn_graph_adjacency, norm='l1', axis=1, copy=False)

        return knn_graph_adjacency

    def diffusion_condensation(self, data):
        """Run Diffusion Condensation"""
        if self.k >= data.shape[0]:
            self.k = data.shape[0] - 1

        P = self.diffusion_operator(data)
        for _ in np.arange(self.t):
            data = P @ data

        return data

    def merge_data_points(self, data):
        """
        Merges data points that are within the merge_threshold distance of each other.
        Keeps track of which points are merged into which.
        """
        distance_matrix = squareform(pdist(data))
        numpoints = distance_matrix.shape[0]
        cluster_mapping = {i: i for i in range(numpoints)}
        indices_to_keep = list(set(range(numpoints)))
        new_merges = False

        # Iterate over data labels starting with highest number to push labels to lower values
        for j in np.arange(numpoints - 1, 0, -1):
            min_dist = np.min(distance_matrix[j, :j])
            if min_dist < self.merge_threshold:
                new_merges = True
                # Merge with closest index
                cluster_mapping[j] = np.argmin(distance_matrix[j, :j])
                # Discard datapoint in future condensation iterations
                indices_to_keep.remove(j)
                for i in np.arange(numpoints - 1, j, -1):
                    # Relabel clusters whose label is the current relabeled node
                    if cluster_mapping[i] == j:
                        cluster_mapping[i] = cluster_mapping[j]
                    # Shift higher indices to lower value (for correct future indexing)
                    elif cluster_mapping[i] > j:
                        cluster_mapping[i] -= 1

        new_data = data[indices_to_keep]

        if new_merges:
            return new_data, cluster_mapping
        else:
            # No need to pass cluster mapping if no merges occur
            return new_data, None

    def fit(self, data):
        """
        Iteratively applies given_function to the data and merges close data points.
        Returns the final dataset and the history of merged data points.
        """
        # dist = pdist(data)
        # self.epsilon = 1e-15 - np.max(dist) ** 2 / np.log(np.min(dist) + 1e-10)
        num_clusters = data.shape[0]
        cluster_maps = [{"clusters": num_clusters,
                         "cluster_maps": {i: i for i in range(num_clusters)}}]

        iterations = 0
        while iterations < self.max_iterations and num_clusters > self.min_clusters:
            data = self.diffusion_condensation(data)  # Apply the given function
            data, mapping = self.merge_data_points(data)  # Merge close points
            if mapping is not None:
                cluster_maps.append({"clusters": data.shape[0],
                                     "cluster_maps": mapping})

            num_clusters = data.shape[0]
            iterations += 1

        self.cluster_tree = cluster_maps
        self.get_labels()

    def get_labels(self, cluster_level=None):
        """Retrieves coarsest cluster ids for datapoints"""
        labels = self.cluster_tree[0]["cluster_maps"]
        for map in self.cluster_tree[1:cluster_level]:
            temp = {}
            for datapoint_id, label in labels.items():
                temp[datapoint_id] = map["cluster_maps"][label]

            labels = temp

        self.labels_ = np.array([v for v in labels.values()])

    def predict(self):
        return None