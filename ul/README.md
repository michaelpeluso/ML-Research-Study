# Unsupervised Learning Analysis

## Overview

This project implements a comprehensive unsupervised learning analysis framework for evaluating clustering algorithms, dimensionality reduction techniques, and their impact on neural network performance. The implementation follows a systematic five-step approach:

1. **Clustering Analysis** - K-Means and Gaussian Mixture Models (GMM/EM) on original datasets
2. **Dimensionality Reduction** - PCA, ICA, and Random Projection techniques
3. **Clustering on Reduced Data** - Re-applying clustering algorithms to dimensionality-reduced feature spaces
4. **Neural Networks on Reduced Data** - Training neural networks on transformed datasets
5. **Neural Networks with Cluster Features** - Augmenting neural networks with cluster-derived features

The analysis is performed on two real-world datasets:

-   **US Accidents Dataset** (2M samples, 28 features) - Regression task predicting accident severity
-   **Hotel Bookings Dataset** (87K samples, 14 features) - Classification task predicting booking cancellations

## Project Structure

```
ul/
├── src/
│   ├── main.py                          # Main execution entry point
│   ├── clustering/                      # K-Means and GMM implementations
│   ├── dimensionality_reduction/        # PCA, ICA, and Random Projection
│   ├── neural_networks/                 # Neural network models and training
│   ├── core/                            # Orchestration for Steps 3-5
│   └── utils/                           # Data processing, plotting, logging
├── data/                                # Raw datasets (CSV files)
├── cache/                               # Cached processed data and models
├── figures/                             # Generated plots and visualizations
├── logs/                                # Execution logs and experiment reports
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Requirements

-   Python 3.12 or higher
-   Required packages listed in `requirements.txt`

### Key Dependencies

-   `scikit-learn` - Machine learning algorithms
-   `torch` - Neural network implementation
-   `pandas`, `numpy` - Data manipulation
-   `matplotlib`, `seaborn`, `plotly` - Visualization
-   `category_encoders` - Feature encoding

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd ul
    ```

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv

    # On Windows
    .venv\Scripts\activate

    # On macOS/Linux
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Verify data files are present:**
   Ensure the following files exist in the `data/` directory:
    - `US_Accidents_March23_2M_rows.csv`
    - `hotel_bookings.csv`

## Execution

### Running the Complete Analysis

Execute all five steps of the unsupervised learning pipeline:

```bash
cd src
python main.py
```

### Configuration Options

The main execution parameters are defined at the top of `src/main.py`:

```python
# Datasets to analyze
DATASETS = ['accidents', 'hotels']

# Component/cluster range to test
N_COMPONENTS_RANGE = list(range(2, 15))  # Test 2-14 clusters/components

# Random seed for reproducibility
SEED = 42

# Computational parameters
N_JOBS = 8                    # Parallel processing threads
STABILITY_RUNS = 5            # Runs for stability analysis
N_INIT = 10                   # K-Means initialization attempts
SILHOUETTE_SUBSAMPLE = 10000  # Samples for silhouette calculation
```

## Output

### Generated Artifacts

1. **Figures** (`figures/` directory):

    - Clustering evaluation plots (silhouette, elbow curves, BIC/AIC)
    - Dimensionality reduction comparisons (scree plots, reconstruction error)
    - Neural network learning curves and performance metrics
    - Comprehensive heatmaps and comparison visualizations

2. **Logs** (`logs/` directory):

    - Detailed execution reports for each step
    - Performance metrics and timing information
    - Hyperparameter configurations and results

3. **Cache** (`cache/` directory):

    - Processed datasets (speeds up subsequent runs)
    - Trained models and transformers
    - Intermediate results for each experiment step

4. **Summary Reports**:
    - `figures/<dataset>/ul_report_summary.csv` - Consolidated metrics
    - `figures/<dataset>/execution_report.txt` - Detailed execution logs

### Key Visualizations

The analysis generates publication-quality figures including:

-   Feature variance distributions
-   PCA scree plots with explained variance
-   Clustering quality heatmaps
-   Neural network convergence curves
-   2D projections of reduced feature spaces
-   Comprehensive metric comparison tables

Generate additional visualizations with

```python
python create_advanced_figures.py
```

## Expected Runtime

Execution time varies by hardware configuration:

-   **Accidents Dataset**: ~30-60 minutes (2M samples)
-   **Hotels Dataset**: ~10-20 minutes (87K samples)

Using cached data and pre-trained models can reduce subsequent runs to minutes.

## Reproducibility

All experiments use fixed random seeds (`SEED=42`) for reproducibility. The data processing pipeline applies:

-   Deterministic train/validation/test splits
-   Consistent feature scaling and encoding
-   Fixed initialization for clustering and neural networks
