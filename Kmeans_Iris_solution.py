import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_iris(path="Iris.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"])
    return df


class KMeans:
    def __init__(self, n_clusters=3, random_state=None):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids_ = None

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        choices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids_ = X[choices].astype(float)

    def _assign_labels(self, X):
        # Compute pairwise distances between points and centroids
        expanded = np.expand_dims(X, axis=1)
        distances = np.linalg.norm(expanded - self.centroids_, axis=-1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.empty_like(self.centroids_)
        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) == 0:
                # keep previous centroid if cluster is empty
                new_centroids[cluster] = self.centroids_[cluster]
            else:
                new_centroids[cluster] = cluster_points.mean(axis=0)
        return new_centroids

    def fit(self, X, max_iter=100):
        X = np.asarray(X, dtype=float)
        self._initialize_centroids(X)
        for _ in range(max_iter):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.allclose(new_centroids, self.centroids_):
                break
            self.centroids_ = new_centroids
        self.labels_ = labels
        return self


def main():
    iris = load_iris()
    X = iris.iloc[:, :-1].values
    print("Loaded Iris dataset:", iris.shape)
    print(iris["Species"].value_counts())

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    print("Centroids:")
    print(model.centroids_)
    unique, counts = np.unique(model.labels_, return_counts=True)
    print("Cluster sizes:", dict(zip(unique.tolist(), counts.tolist())))

    fig = go.Figure()
    colors = ["#DB4CB2", "#c9e9f6", "#7D3AC1"]
    names = ["Cluster 0", "Cluster 1", "Cluster 2"]
    for cluster in range(3):
        mask = model.labels_ == cluster
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker_color=colors[cluster],
                name=names[cluster],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=model.centroids_[:, 0],
            y=model.centroids_[:, 1],
            mode="markers",
            marker_color="#CAC9CD",
            marker_symbol="x",
            marker_size=12,
            name="Centroids",
        )
    )
    fig.update_layout(
        title="K-Means Clustering on Iris (first two features)",
        template="plotly_dark",
        width=900,
        height=500,
    )
    fig.show()


if __name__ == "__main__":
    main()
