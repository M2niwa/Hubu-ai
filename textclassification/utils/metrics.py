from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(pred, labels):
    """模型评估指标"""
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='weighted')
    return {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4)
    }