from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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