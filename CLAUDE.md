# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational data mining application implementing fundamental ML algorithms from scratch using NumPy. Built for university practical sessions (TP1-TP6) covering preprocessing, clustering, and classification with an interactive PyQt6 GUI.

## Development Commands

### Running the Application
```bash
make start              # Primary method
./run_app.sh            # Alternative via shell script
uv run python data_analysis_app.py  # Direct execution
```

### Dependency Management
```bash
uv sync                 # Install/sync dependencies (recommended)
pip install .           # Alternative using pip
```

**Note:** This project uses `uv` for fast dependency management (Python 3.13+).

## Architecture

### Two-Layer Design

**1. Algorithm Layer (`algorithms.py`)**
- Pure NumPy implementations of all ML algorithms (no scikit-learn for core logic)
- Self-contained classes following scikit-learn API conventions (fit/predict/transform)
- Organized into functional groups:
  - **Clustering:** KMeans, KMedoids, AGNES (agglomerative hierarchical), DIANA (divisive hierarchical), DBSCAN
  - **Preprocessing:** SimpleImputer (mean/median), MinMaxScaler, StandardScaler
  - **Classification:** KNN, GaussianNaiveBayes
  - **Utilities:** train_test_split, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

**2. GUI Layer (`data_analysis_app.py`)**
- PyQt6-based interface with custom "Midnight Aurora" theme
- Tab-based workflow matching the TP structure:
  - Data Loading & Preprocessing (TP1)
  - Clustering (TP2-4): Interactive parameter controls + 2D scatter plot visualization
  - Classification (TP5-6): Train/test split UI, model evaluation, KNN k-optimization plotting
- Custom UI components: `ModernCard`, `AccentButton`, `GlowEffect`, `MatplotlibCanvas`
- Color palette defined in `COLORS` dict at top of file

### Key Design Patterns

**Algorithm Independence:** Each algorithm in `algorithms.py` is completely self-contained with no cross-dependencies. When modifying an algorithm, you only need to edit that single class.

**GUI-Algorithm Separation:** The GUI imports all algorithms from `algorithms.py` but algorithms have zero GUI dependencies. This allows testing algorithms independently.

**Stateful UI Workflow:** The application maintains state across tabs:
- `self.df` stores the loaded DataFrame
- `self.preprocessed_data` stores transformed data
- `self.cluster_labels` stores clustering results
- Classification tab expects preprocessing to be completed first

## Dataset Structure

**Location:** `data/` directory

**Format:** CSV files with:
- Last column = target/label (for classification)
- All preceding columns = features
- May contain missing values (handled by SimpleImputer)

**Example datasets:**
- `iris.csv` - Classic 4-feature classification
- `diabetes.csv` - Medical diagnosis
- `heart.csv`, `horse-colic.csv` - Additional benchmarks

## Algorithm Implementation Notes

### Clustering Algorithms
- **KMeans/KMedoids:** Initialize with random centroids/medoids, iterate until convergence or max_iters
- **AGNES:** O(N³) bottom-up merging, supports single/complete/average linkage
- **DIANA:** Top-down splitting, finds cluster with max diameter and splits via splinter group
- **DBSCAN:** Density-based, labels noise as -1

### Distance Calculations
All clustering algorithms use **Euclidean distance** via `np.linalg.norm()`. Broadcasting pattern for vectorized distance matrices:
```python
distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
```

### Classification
- **KNN:** Majority voting among k nearest neighbors, no training phase (lazy learning)
- **Naive Bayes:** Gaussian PDF assumption, uses log probabilities to avoid underflow
- **Evaluation:** All metrics support multi-class classification with macro averaging

## GUI Development Guidelines

### Styling
- All colors reference the `COLORS` dictionary
- Use `ModernCard` for grouped sections, `AccentButton` for primary actions
- Follow the existing pattern for parameter controls (QSpinBox/QDoubleSpinBox in QFormLayout)

### Adding New Algorithms
1. Implement the algorithm class in `algorithms.py` with `fit()` method
2. Import in `data_analysis_app.py` line 18-24
3. Add UI controls in the appropriate tab's `init_*_tab()` method
4. Add execution logic in the corresponding button handler (e.g., `run_clustering()`)

### Matplotlib Integration
- Each tab with visualization uses `MatplotlibCanvas` wrapper around FigureCanvas
- Clustering uses 2D scatter plots (first 2 features if >2 dimensions)
- Classification uses line plots for KNN k-optimization
- Always call `canvas.draw()` after updating plot data

## Testing Workflow

**Manual Testing via GUI:**
1. Load dataset from `data/` directory
2. Apply preprocessing (imputation → normalization/standardization)
3. Run clustering with various parameters and observe visual clusters
4. Switch to classification tab, split data, train model, evaluate metrics

**No automated tests exist** - this is an educational project focused on algorithm implementation.
