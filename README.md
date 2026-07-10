# K-Means Clustering on Iris Dataset

This repository contains a custom implementation of the **K-Means Clustering** algorithm from scratch in Python, applied to the classic **Iris Dataset**. It includes a script for command-line execution and visualization, as well as a Jupyter Notebook for interactive exploration.

## 🚀 Overview

The K-Means algorithm partitions $n$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean (centroid). 

This project implements the algorithm without using scikit-learn's clustering module, relying instead on:
- **NumPy** for vectorization and distance computations.
- **Pandas** for dataset loading and preprocessing.
- **Plotly** for interactive dark-themed 2D scatter plots of the clusters.

---

## 📁 Repository Structure

*   [Kmeans_Iris_solution.py](file:///c:/Users/vigne/Desktop/Kmeans/Kmeans_Iris_solution.py): The main Python implementation containing the custom [KMeans](file:///c:/Users/vigne/Desktop/Kmeans/Kmeans_Iris_solution.py#L12) class, logic for running it on the Iris dataset, and rendering an interactive Plotly visualization.
*   `Kmean.ipynb`: Jupyter Notebook containing step-by-step walkthroughs and experiments.
*   `Iris.csv`: The Iris flower dataset containing 150 samples with four features:
    *   Sepal Length (cm)
    *   Sepal Width (cm)
    *   Petal Length (cm)
    *   Petal Width (cm)
    *   Species (target label, omitted during clustering)

---

## 🛠️ Implementation Details

The core algorithm is implemented in the [KMeans](file:///c:/Users/vigne/Desktop/Kmeans/Kmeans_Iris_solution.py#L12) class in [Kmeans_Iris_solution.py](file:///c:/Users/vigne/Desktop/Kmeans/Kmeans_Iris_solution.py):

*   **Initialization** (`_initialize_centroids`): Initial centroids are chosen randomly from the input data points using a specified `random_state` for reproducibility.
*   **Label Assignment** (`_assign_labels`): Computes pairwise Euclidean distances between each data point and the current centroids, assigning points to the nearest centroid.
*   **Centroid Update** (`_update_centroids`): Recomputes the centroids as the mean of all points assigned to each cluster.
*   **Convergence** (`fit`): The algorithm iterates until centroids do not change position (using `np.allclose`) or `max_iter` is reached.

---

## ⚙️ Setup & Requirements

To run this project locally, ensure you have Python installed, then install the required dependencies:

```bash
pip install numpy pandas plotly
```

If you wish to run the interactive Jupyter Notebook, also install:

```bash
pip install notebook
```

---

## 🏃 How to Run

Run the Python script directly from your terminal:

```bash
python Kmeans_Iris_solution.py
```

### Expected Output
Upon execution, the script will:
1. Load the Iris dataset and display basic statistics about the target species.
2. Initialize and fit the custom `KMeans` model with $K=3$.
3. Print the final centroids and the size of each cluster.
4. Launch an interactive Plotly graph showing the clusters and their centroids on a dark-themed canvas:
   * **x-axis**: Sepal Length
   * **y-axis**: Sepal Width
   * **Markers**: Cluster membership represented by distinct colors, with centroids marked by a gray `x`.
