import mlflow
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn
from typing import Dict, List, Optional

import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Optional, Dict, List, Any

def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    model: Any = None,
    params: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    tags: Optional[Dict] = None,
    artifacts: Optional[List[str]] = None,
    model_path: str = "model",
    y_test=None,
    y_pred=None
):
    """
    Logs parameters, metrics, models, artifacts, and diagnostic plots to MLflow.
    """
    # 1. Clean up active runs
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # 2. Log Metadata (Params, Metrics, Tags)
        if params: mlflow.log_params(params)
        if metrics: mlflow.log_metrics(metrics)
        if tags: mlflow.set_tags(tags)

        # 3. Log Model with automatic flavor detection
        if model is not None:
            try:
                # Uses autolog/infer logic or defaults to sklearn
                # You can extend this logic for xgboost/pytorch as needed
                mlflow.sklearn.log_model(sk_model=model, artifact_path=model_path)
                print(f"Model logged: {model_path}")
            except Exception as e:
                print(f"Model logging failed: {e}")

        # 4. Log Visualization (Confusion Matrix)
        if y_test is not None and y_pred is not None:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation=45)
                ax.set_title(f"Confusion Matrix - {run_name}")
                
                # Log plot directly to the 'plots' directory in artifacts
                mlflow.log_figure(fig, "plots/confusion_matrix.png")
                plt.close(fig)
                print("Visualization logged: plots/confusion_matrix.png")
            except Exception as e:
                print(f"Visualization failed: {e}")

        # 5. Log Custom Artifacts
        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path, artifact_path="extra_files")

    print(f"Session Finished: {run_name} (ID: {run.info.run_id})")