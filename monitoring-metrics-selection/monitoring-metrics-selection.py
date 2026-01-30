def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    n = len(y_true)
    if system_type == "classification":
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for ind in range(n):
            if y_true[ind] == 1 and y_pred[ind] == 1:
                tp += 1
            if y_true[ind] == 0 and y_pred[ind] == 0:
                tn += 1
            if y_true[ind] == 0 and y_pred[ind] == 1:
                fp += 1
            if y_true[ind] == 1 and y_pred[ind] == 0:
                fn += 1
        accuracy = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        return sorted([
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall)
        ])
    elif system_type == "regression":
        mae = sum([abs(y_true[ind] - y_pred[ind]) for ind in range(n)]) / n
        rmse = (sum([(y_true[ind] - y_pred[ind]) ** 2 for ind in range(n)]) / n) ** 0.5
        return sorted([
            ("mae", mae),
            ("rmse", rmse)
        ])
    elif system_type == "ranking":
        k = 3
        ranked = sorted(zip(y_pred, y_true), reverse=True)
        top_k = ranked[:k]
        relevant_in_top_k = sum(y for _, y in top_k)
        precision_at_k = relevant_in_top_k / k if k else 0.0
        total_relevant = sum(y_true)
        recall_at_k = relevant_in_top_k / total_relevant if total_relevant else 0.0
        return sorted([
            ("precision_at_3", precision_at_k),
            ("recall_at_3", recall_at_k)
        ])
    else:
        return []