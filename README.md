## Hand Gesture Classification

Supervised learning project for recognizing hand gestures from landmark-based features, comparing several classical ML models. The final production model is based on **XGBoost**.

---

## Project Structure

- `src/` – reusable Python modules
  - `preprocessing.py`: data preprocessing utilities (re-centering and normalization of landmarks).
  - `helpers.py`: shared helper functions for evaluation and visualization.
  - `mlflow_logger.py`: helper for logging runs, metrics, models, and plots.
- `notebooks/`
  - `hand_gestures_classification.ipynb`: main experimentation notebook where models are trained and evaluated.
  - `mlruns/`: experiment tracking outputs (created automatically when you track runs).
- `screenshoots/` – PNG screenshots for documentation/presentation.
- `README.md` – project overview and usage instructions.

---

## Model Comparison

| Run name                      | Model               | Accuracy | F1 Score | Precision | Recall  |
| ----------------------------- | ------------------- | -------- | -------- | --------- | ------- |
| xgb_default                   | XGBoost             | 0.98033  | 0.98035  | 0.98046   | 0.98033 |
| gridsearch_svc_C30_ovo_gamma2 | SVC (GridSearch)    | 0.97644  | 0.97648  | 0.97661   | 0.97644 |
| svc_poly_ovo_d10              | SVC (Poly, d=10)    | 0.97371  | 0.97378  | 0.97398   | 0.97371 |
| svc_poly_ovo_d15              | SVC (Poly, d=15)    | 0.97332  | 0.97339  | 0.97355   | 0.97332 |
| gridsearch_svc_C20_ovo_gamma1 | SVC (GridSearch)    | 0.97176  | 0.97179  | 0.97206   | 0.97176 |
| gridsearch_svc_C10_ovo_gamma1 | SVC (GridSearch)    | 0.96670  | 0.96675  | 0.96715   | 0.96670 |
| random_forest_default         | Random Forest       | 0.96436  | 0.96437  | 0.96450   | 0.96436 |
| ovr_logistic_cv               | Logistic Regression | 0.93437  | 0.93423  | 0.93503   | 0.93437 |
| svc_poly_ovo_d5               | SVC (Poly, d=5)     | 0.92795  | 0.92851  | 0.93215   | 0.92795 |
| svc_poly_ovo_d3               | SVC (Poly, d=3)     | 0.83739  | 0.83668  | 0.84916   | 0.83739 |


XGBoost achieved the best overall performance on both **accuracy** and **F1 score**.

---

## Why XGBoost Was Chosen

XGBoost was selected as the final model for several reasons:

- **Top performance across metrics**: It clearly outperforms the other evaluated models on accuracy, F1, precision, and recall (see table above).
- **Handles non-linear decision boundaries**: Hand-gesture features can be highly non-linear; gradient-boosted trees naturally capture these relationships better than linear models.
- **Robust to feature interactions**: XGBoost can automatically model complex feature interactions without manual feature engineering.
- **Regularization and robustness**: Built-in L1/L2 regularization and shrinkage help reduce overfitting compared with plain decision trees or un-regularized models.
- **Interpretability tools**: Feature importance from tree ensembles gives insight into which features drive the predictions.
- **Efficient training**: The implementation is optimized and scales well to larger datasets and parameter searches.

---

## How to Use This Project

- **Environment**: Create a Python environment with at least the following packages installed: `scikit-learn`, `xgboost`, `matplotlib`, `pandas`, `numpy`, and `jupyter` (see `requirements.txt`).
- **Reproducing experiments**: Open `notebooks/hand_gestures_classification.ipynb` in Jupyter, run the cells to preprocess data, train models, and evaluate them on the test set.
- **Comparing models**: Use the metrics printed in the notebook (and the table above) to compare different algorithms and configurations.

---

## End-to-End Usage Guide (From Clone to Results)

Below is a complete, reproducible flow you can follow after cloning the repository.

### 1. Clone the Repository

```bash
git clone <YOUR_REPO_URL> hand-gesture-classification
cd hand-gesture-classification
```

> Replace `<YOUR_REPO_URL>` with the actual Git URL of this project.

### 2. Create and Activate a Virtual Environment (Recommended)

Using `venv`:

```bash
python -m venv .venv
```

- **Windows (PowerShell)**:
  ```bash
  .venv\Scripts\Activate
  ```
- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter and Run the Notebook

From the project root:

```bash
jupyter notebook
```

Then, in the browser:

1. Open `notebooks/hand_gestures_classification.ipynb`.
2. Run the notebook cells in order:
   - Data loading and preprocessing (using functions from `src/preprocessing.py`).
   - Model training and evaluation (SVC, Random Forest, Logistic Regression, XGBoost, etc.).
   - Computing metrics and confusion matrices (using helpers in `src/helpers.py`).

---

## MLflow Tracking and UI (Optional)

This project can optionally use **MLflow** for experiment tracking through the helper `log_to_mlflow` function in `src/mlflow_logger.py`. When enabled, each run can log:

- **Parameters**: model hyperparameters and configuration.
- **Metrics**: accuracy, F1, precision, recall, etc.
- **Artifacts**: trained model object, confusion-matrix plot, and any extra files.
- **Metadata**: run name, experiment name, and tags.

As you execute runs with MLflow enabled in the notebook, the `notebooks/mlruns/` directory will be populated automatically.

### How to launch the MLflow UI

1. Make sure MLflow is installed (via `requirements.txt` or manually):

   ```bash
   pip install mlflow
   ```

2. From the project root, start the UI pointing to the tracking directory:

   ```bash
   mlflow ui
   ```

3. Open your browser at:

   ```text
   http://localhost:5000
   ```

There you can:

- Compare all recorded runs (e.g., `xgb_default`, `gridsearch_svc_C30_ovo_gamma2`, etc.).
- Inspect metrics (accuracy, F1, precision, recall).
- Download models and view the confusion-matrix plots.

This final section is the only place where MLflow is required; the rest of the workflow can be run without it if you only need local experiments.
