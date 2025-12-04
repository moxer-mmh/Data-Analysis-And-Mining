# Data Analysis & Mining Application

A comprehensive Python application for data analysis, preprocessing, clustering, and classification. This project implements fundamental data mining algorithms from scratch, providing an interactive GUI for exploration and learning.

## 🚀 Features

### 1. Data Preprocessing (TP1)
*   **Data Loading:** Support for CSV files.
*   **Missing Value Imputation:** Replace missing values with Mean or Median.
*   **Normalization:** Min-Max Scaling (0-1 range).
*   **Standardization:** Z-Score Scaling (Mean 0, Std Dev 1).

### 2. Clustering (TP2, TP3, TP4)
Implementations of unsupervised learning algorithms:
*   **Partitioning:**
    *   **K-Means:** Centroid-based clustering.
    *   **K-Medoids:** Medoid-based clustering (more robust to outliers).
*   **Hierarchical:**
    *   **AGNES:** Agglomerative (Bottom-Up) with Single, Complete, or Average linkage.
    *   **DIANA:** Divisive (Top-Down).
*   **Density-Based:**
    *   **DBSCAN:** Clustering based on density (epsilon, min_samples).
*   **Visualization:** Interactive 2D scatter plots of clusters.

### 3. Classification (TP5, TP6)
Implementations of supervised learning algorithms:
*   **K-Nearest Neighbors (KNN):**
    *   Classification based on 'k' nearest neighbors.
    *   **Optimization:** Automated search for the optimal 'k' (1-10) with precision plotting.
*   **Naive Bayes:**
    *   Gaussian Naive Bayes implementation for continuous features.
*   **Evaluation Metrics:**
    *   Confusion Matrix.
    *   Accuracy, Precision, Recall, F1-Measure.
    *   Train/Test Split (customizable ratio).

## 🛠️ Installation

### Prerequisites
*   Python 3.13+
*   `uv` (Recommended for dependency management)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd TP-Data
    ```

2.  **Install dependencies:**

    Using `uv` (fastest & recommended):
    ```bash
    uv sync
    ```

    Alternatively, using `pip` with the project file:
    ```bash
    pip install .
    ```

## 🖥️ Usage

The easiest way to run the application is using the Makefile:

```bash
make start
```

Alternatively, you can use the shell script or run directly with `uv`:

```bash
# Using the script
./run_app.sh

# OR using uv
uv run python data_analysis_app.py
```

## 📂 Project Structure

*   `data_analysis_app.py`: Main entry point and GUI implementation (PyQt6).
*   `algorithms.py`: Core implementations of all ML algorithms (from scratch).
*   `data/`: Directory containing dataset benchmarks (e.g., diabetes.csv, iris.csv).
*   `docs/references/`: Course materials and practical session PDFs.
*   `notebooks/`: Jupyter notebooks for experiments and scratchpad code.

## 🧪 Practical Sessions Covered

*   **TP1:** Data Exploration & Preprocessing.
*   **TP2:** K-Means Clustering.
*   **TP3:** K-Medoids Clustering.
*   **TP4:** Clustering Comparison & Interfaces.
*   **TP5:** Supervised Learning (KNN).
*   **TP6:** Naive Bayes & Evaluation.

## 📝 Notes

This project avoids using high-level ML libraries (like scikit-learn) for the *core logic* of the algorithms, implementing them using `numpy` to demonstrate understanding of the underlying mechanics.