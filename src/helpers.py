from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate common classification metrics and return as a dictionary.

    Args:
        y_true: true labels
        y_pred: predicted labels
        average: type of averaging for multi-class ('weighted', 'macro', 'micro')

    Returns:
        dict: keys are 'accuracy', 'f1', 'precision', 'recall'
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average)
    }
    return metrics

def show_metrics(model, x_test_prep, y_test):
    y_test_pred = model.predict(x_test_prep)
    metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average='weighted'),
            "precision": precision_score(y_test, y_test_pred, average='weighted'),
            "recall": recall_score(y_test, y_test_pred, average='weighted')
        }
    print(metrics)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax, xticks_rotation=90)
    ax.set_title(f"Confusion Matrix - {model.__class__.__name__}")