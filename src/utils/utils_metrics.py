import torch

from sklearn.metrics import f1_score, average_precision_score

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum()
    acc = correct / y_true.size(0)
    return acc.item()

def accuracy_for_binary(y_pred, y_true):
    # y_pred: logity [batch_size]
    # y_true: etykiety [batch_size] (0 lub 1)
    probs = torch.sigmoid(y_pred)
    predicted = (probs >= 0.5).long()
    correct = (predicted == y_true.long()).sum()
    acc = correct.float() / y_true.size(0)
    return acc.item()

def mean(lst):
    return sum(lst) / len(lst)


def prf(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    class_pred = predicted.bool()
    class_true = y_true.bool()
    tp = (class_pred & class_true).sum().item()
    fp = (class_pred & ~class_true).sum().item()
    fn = (~class_pred & class_true).sum().item()
    return tp, fp, fn
    
def calc_f1(running_metrics, phase):
    tp = sum(running_metrics[f'{phase}_tp']) / sum(running_metrics['batch_sizes'])
    fp = sum(running_metrics[f'{phase}_fp']) / sum(running_metrics['batch_sizes'])
    fn = sum(running_metrics[f'{phase}_fn']) / sum(running_metrics['batch_sizes'])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def get_val_score(epochs_metrics, phase, score_name):
    """
    Get the validation score based on the configuration.
    
    Args:
        epochs_metrics (dict): Dictionary containing metrics for each epoch.
        phase (str): The phase of training, e.g., 'val'.
        score_name (str): The name of the score to retrieve, e.g., 'acc'.
    
    Returns:
        float: The validation score.
    """
    return epochs_metrics[f'{phase}_{score_name}'][-1]


def mAP(y_pred, y_true):
    """
    y_true: torch.Tensor [N, num_labels] lub numpy array, multi-hot
    y_prob: tensor/numpy [N, num_labels], wartości prawdopodobieństwa (np. po sigmoid)
    """
    y_pred = (torch.sigmoid(y_pred) > 0.5).int()
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    mAP = average_precision_score(y_true, y_pred, average='macro')
    return mAP