# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.runner import DistEvalHook as BasicDistEvalHook


class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis] # (batch_size, 1)
    if len(labels.shape) == 3: 
        labels = labels.squeeze(-1)
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def top_k_accuracy_multilabel(scores, labels, topk=(1,)):
    """
    Multi-label top-k recall: measures average recall of ground truth labels.
    Args:
        scores (list[np.ndarray]): List of prediction scores, shape (num_classes,) each
        labels (list): Ground truth labels (int, list, or array)
        topk (tuple): Tuple of k values
    Returns:
        list: Average recall values for each k
    """
    num_samples = len(scores)
    maxk = max(topk)
    res = []
    for k in topk:
        total_recall = 0
        for i in range(num_samples):
            # Get top-k predictions
            score_array = np.array(scores[i])
            # top_k_indices = np.argsort(score_array)[-k:][::-1]
            # top_k_preds = set(top_k_indices)
            top_k_preds = set(np.where(score_array > 0.5)[0])
            if len(top_k_preds) == 0:
                top_k_preds = [0]

            # Normalize ground truth labels to set
            if isinstance(labels[i], (list, np.ndarray)):
                gt_set = set(labels[i])
            else:
                gt_set = {labels[i]}

            # Calculate recall for this sample
            if len(gt_set) > 0:
                recall = len(top_k_preds & gt_set) / len(gt_set)
                total_recall += recall

        avg_recall = total_recall / num_samples * 100.0
        res.append(avg_recall)
    return res


def compute_class_recall(scores, labels, num_classes, topk=(1,)):
    """
    Compute per-class recall and mean recall for multi-label classification.
    
    Args:
        scores (list[np.ndarray]): List of prediction scores for each sample.
                                   Each element is an array of shape (num_classes,)
        labels (list): Ground truth labels. Each element can be:
                      - int: single label
                      - list/array: multiple labels
        num_classes (int): Total number of classes
        topk (tuple): Tuple of k values to compute recall for
    
    Returns:
        dict: Dictionary containing:
            - 'overall_recall': recall over all samples
            - 'class_recall': dict mapping k -> array of per-class recall (num_classes,)
            - 'mean_recall': dict mapping k -> mean recall across classes
            - 'class_support': array of sample counts per class (num_classes,)
            - 'y_preds': list of predicted labels for each sample
    """
    num_samples = len(scores)
    y_preds = []
    
    # Initialize counters for each class
    class_tp = {k: np.zeros(num_classes) for k in topk}  # True positives
    class_support = np.zeros(num_classes)  # Total ground truth samples per class
    total_recall = 0
    for i in range(num_samples):
        score_array = np.array(scores[i])
        
        # Normalize ground truth labels to set
        if isinstance(labels[i], (list, np.ndarray)):
            gt_set = set(labels[i])
        else:
            gt_set = {labels[i]}
        
        # Update support count for each ground truth class
        for gt_label in gt_set:
            class_support[gt_label] += 1
        
        # Compute for each k
        for k in topk:
            top_k_preds = set(np.where(score_array > 0.5)[0])
            if len(top_k_preds) == 0:
                # top_k_preds = [0]
                top_k_preds = [np.argmax(score_array)]
            y_preds.append(list(top_k_preds))
            
            # Check which ground truth labels were predicted
            for gt_label in gt_set:
                if gt_label in top_k_preds:
                    class_tp[k][gt_label] += 1
            # Calculate recall for this sample
            if len(gt_set) > 0:
                recall = len(set(top_k_preds) & gt_set) / len(gt_set)
                total_recall += recall
    overall_recall = total_recall / num_samples * 100.0
    # Compute per-class recall and mean recall
    class_recall = {}
    mean_recall = {}
    
    for k in topk:
        # Compute recall for each class (avoid division by zero)
        recall_per_class = np.zeros(num_classes)
        for c in range(num_classes):
            if class_support[c] > 0:
                recall_per_class[c] = class_tp[k][c] / class_support[c] * 100.0
            else:
                recall_per_class[c] = np.nan  # No samples for this class
        
        class_recall[k] = recall_per_class
        
        # Compute mean recall (excluding classes with no samples)
        valid_recalls = recall_per_class[~np.isnan(recall_per_class)]
        mean_recall[k] = np.mean(valid_recalls) if len(valid_recalls) > 0 else 0.0
    
    return {
        'overall_recall': overall_recall,
        'mean_recall': mean_recall,
        'class_recall': class_recall,
        'class_support': class_support,
        'y_preds': y_preds
    }

def print_recall_results(results, topk=(1,), class_name_list=None):
    """
    Print recall results in a nicely formatted table.
    
    Args:
        results (dict): Results from compute_class_recall_detailed
        topk (tuple): K values to display
    """
    import sys
    
    class_names = results.get('class_names', None)
    if not class_names:
        class_names = class_name_list
    num_classes = len(results['class_support'])
    
    log_msg = []
    # Print header
    log_msg.append("\n" + "="*80)
    log_msg.append("MULTI-LABEL CLASSIFICATION RECALL RESULTS".center(80))
    log_msg.append("="*80)
    
    # Print overall metrics
    log_msg.append("\n┌─── Overall Metrics " + "─"*56 + "┐")
    log_msg.append("│" + " "*77 + "│")
    
    header = f"│  {'Metric':<30s}"
    for k in topk:
        header += f"  Top-{k:2d}  "
    header += " │"
    log_msg.append(header)
    log_msg.append("│" + "─"*77 + "│")
    
    # overall_recall
    row = f"|  {'Recall':<30s}"
    row += f"  {results['overall_recall']:6.2f}%"
    log_msg.append(row)

    # Mean Recall
    row = f"│  {'Mean Recall (macro avg)':<30s}"
    for k in topk:
        row += f"  {results['mean_recall'][k]:6.2f}%"
    log_msg.append(row)
    
    # Weighted Recall
    if 'weighted_recall' in results:
        row = f"│  {'Weighted Recall (micro avg)':<30s}"
        for k in topk:
            row += f"  {results['weighted_recall'][k]:6.2f}%"
        row += " │"
        log_msg.append(row)
    
    log_msg.append("│" + " "*77 + "│")
    log_msg.append("└" + "─"*77 + "┘")
    
    # Print per-class results
    log_msg.append("\n┌─── Per-Class Recall " + "─"*56 + "┐")
    log_msg.append("│" + " "*77 + "│")
    
    # Table header
    header = f"│  {'Class':<25s} {'Support':>8s}"
    for k in topk:
        header += f"  Top-{k:2d}  "
    header += " │"
    log_msg.append(header)
    log_msg.append("│" + "─"*77 + "│")
    
    # Sort classes by support (descending) for better readability
    class_indices = np.argsort(results['class_support'])[::-1]
    
    for idx in class_indices:
        support = int(results['class_support'][idx])
        
        # Skip classes with no samples (optional)
        if support == 0:
            continue
        
        # Get class name
        if class_names is not None and idx < len(class_names):
            class_name = class_names[idx]
        else:
            class_name = f"Class {idx}"
        
        # Truncate long class names
        if len(class_name) > 24:
            class_name = class_name[:21] + "..."
        
        row = f"│  {class_name:<25s} {support:8d}"
        
        for k in topk:
            recall = results['class_recall'][k][idx]
            if not np.isnan(recall):
                row += f"  {recall:6.2f}%"
            else:
                row += f"  {'N/A':>7s}"
        # row += " │"
        log_msg.append(row)
    
    log_msg.append("│" + " "*77 + "│")
    log_msg.append("└" + "─"*77 + "┘")
    
    # Print summary statistics
    log_msg.append("\n┌─── Summary Statistics " + "─"*53 + "┐")
    log_msg.append("│" + " "*77 + "│")
    
    total_samples = sum(results['class_support'])
    active_classes = np.sum(results['class_support'] > 0)
    
    log_msg.append(f"│  Total Samples: {int(total_samples):<20d}  Active Classes: {int(active_classes)}/{num_classes:<15d} │")
    
    # Best and worst performing classes for Top-1
    if topk and topk[0] in results['class_recall']:
        k = topk[0]
        valid_mask = ~np.isnan(results['class_recall'][k])
        if np.any(valid_mask):
            valid_recalls = results['class_recall'][k][valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            best_idx = valid_indices[np.argmax(valid_recalls)]
            worst_idx = valid_indices[np.argmin(valid_recalls)]
            
            best_name = class_names[best_idx] if class_names else f"Class {best_idx}"
            worst_name = class_names[worst_idx] if class_names else f"Class {worst_idx}"
            
            log_msg.append("│" + " "*77 + "│")
            log_msg.append(f"│  Best Class:  {best_name[:20]:<20s}  ({results['class_recall'][k][best_idx]:.2f}%)")
            log_msg.append(f"│  Worst Class: {worst_name[:20]:<20s}  ({results['class_recall'][k][worst_idx]:.2f}%)")
    
    log_msg.append("│" + " "*77 + "│")
    log_msg.append("└" + "─"*77 + "┘\n")
    log_msg = '\n'.join(log_msg)
    return log_msg


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def compute_multilabel_metrics(scores, labels, num_classes, topk=(1,)):
    """
    Compute comprehensive metrics for multi-label classification including recall, 
    precision, and F1 scores (overall, per-class, and mean).

    Args:
        scores (list[np.ndarray]): List of prediction scores for each sample.
                                    Each element is an array of shape (num_classes,)
        labels (list): Ground truth labels. Each element can be:
                        - int: single label
                        - list/array: multiple labels
        num_classes (int): Total number of classes
        topk (tuple): Tuple of k values to compute metrics for

    Returns:
        dict: Dictionary containing for each k in topk:
            - 'overall_recall': recall computed over all samples
            - 'overall_precision': precision computed over all samples
            - 'overall_f1': F1 score computed from overall recall and precision
            - 'class_recall': array of per-class recall (num_classes,)
            - 'class_precision': array of per-class precision (num_classes,)
            - 'class_f1': array of per-class F1 scores (num_classes,)
            - 'mean_recall': mean recall across classes (excluding NaN)
            - 'mean_precision': mean precision across classes (excluding NaN)
            - 'mean_f1': mean F1 score across classes (excluding NaN)
            - 'class_support': array of ground truth sample counts per class
            - 'class_predictions': array of prediction counts per class
            - 'y_preds': list of predicted labels for each sample
    """
    num_samples = len(scores)
    y_preds = {k: [] for k in topk}

    # Initialize counters for each class and k
    class_tp = {k: np.zeros(num_classes) for k in topk}  # True positives
    class_fp = {k: np.zeros(num_classes) for k in topk}  # False positives
    class_support = np.zeros(num_classes)  # Total ground truth samples per class

    # Overall metrics accumulators
    overall_tp = {k: 0 for k in topk}
    overall_fp = {k: 0 for k in topk}
    overall_fn = {k: 0 for k in topk}

    for i in range(num_samples):
        score_array = np.array(scores[i])
        
        # Normalize ground truth labels to set
        if isinstance(labels[i], (list, np.ndarray)):
            gt_set = set(labels[i])
        else:
            gt_set = {labels[i]}
        
        # Update support count for each ground truth class
        for gt_label in gt_set:
            class_support[gt_label] += 1
        
        # Compute for each k
        for k in topk:
            # Get top-k predictions (using threshold 0.5)
            top_k_preds = set(np.where(score_array > 0.5)[0])
            if len(top_k_preds) == 0:
                # Fallback to argmax if no predictions above threshold
                top_k_preds = {np.argmax(score_array)}
            
            y_preds[k].append(list(top_k_preds))
            
            # Calculate true positives, false positives, false negatives
            tp_set = top_k_preds & gt_set
            fp_set = top_k_preds - gt_set
            fn_set = gt_set - top_k_preds
            
            # Update overall counters
            overall_tp[k] += len(tp_set)
            overall_fp[k] += len(fp_set)
            overall_fn[k] += len(fn_set)
            
            # Update per-class counters
            for pred_tp in tp_set:
                class_tp[k][pred_tp] += 1
            
            for pred_fp in fp_set:
                class_fp[k][pred_fp] += 1

    # Prepare results
    results = {}

    for k in topk:
        # Compute overall metrics
        overall_recall = (overall_tp[k] / (overall_tp[k] + overall_fn[k]) * 100.0 
                            if (overall_tp[k] + overall_fn[k]) > 0 else 0.0)
        overall_precision = (overall_tp[k] / (overall_tp[k] + overall_fp[k]) * 100.0 
                            if (overall_tp[k] + overall_fp[k]) > 0 else 0.0)
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                        if (overall_precision + overall_recall) > 0 else 0.0)
        
        # Compute per-class metrics
        class_recall = np.zeros(num_classes)
        class_precision = np.zeros(num_classes)
        class_f1 = np.zeros(num_classes)
        class_predictions = class_tp[k] + class_fp[k]
        
        for c in range(num_classes):
            # Recall
            if class_support[c] > 0:
                class_recall[c] = class_tp[k][c] / class_support[c] * 100.0
            else:
                class_recall[c] = np.nan
            
            # Precision
            if class_predictions[c] > 0:
                class_precision[c] = class_tp[k][c] / class_predictions[c] * 100.0
            else:
                class_precision[c] = np.nan
            
            # F1 Score
            if not np.isnan(class_recall[c]) and not np.isnan(class_precision[c]):
                if (class_recall[c] + class_precision[c]) > 0:
                    class_f1[c] = (2 * class_precision[c] * class_recall[c] / 
                                    (class_precision[c] + class_recall[c]))
                else:
                    class_f1[c] = 0.0
            else:
                class_f1[c] = np.nan
        
        # Compute mean metrics (excluding NaN values)
        valid_recalls = class_recall[~np.isnan(class_recall)]
        valid_precisions = class_precision[~np.isnan(class_precision)]
        valid_f1s = class_f1[~np.isnan(class_f1)]
        
        mean_recall = np.mean(valid_recalls) if len(valid_recalls) > 0 else 0.0
        mean_precision = np.mean(valid_precisions) if len(valid_precisions) > 0 else 0.0
        mean_f1 = np.mean(valid_f1s) if len(valid_f1s) > 0 else 0.0
        
        results[k] = {
            'overall_recall': overall_recall,
            'overall_precision': overall_precision,
            'overall_f1': overall_f1,
            'class_recall': class_recall,
            'class_precision': class_precision,
            'class_f1': class_f1,
            'mean_recall': mean_recall,
            'mean_precision': mean_precision,
            'mean_f1': mean_f1,
            'class_support': class_support,
            'class_predictions': class_predictions,
            'y_preds': y_preds[k]
        }

    return results[1]


def print_metrics(metrics, label_map=None):
    log_msg = []
    log_msg.append(f"\nOverall Recall: {metrics['overall_recall']:.2f}%")
    log_msg.append(f"Overall Precision: {metrics['overall_precision']:.2f}%")
    log_msg.append(f"Overall F1: {metrics['overall_f1']:.2f}%")
    log_msg.append(f"\nMean Recall: {metrics['mean_recall']:.2f}%")
    log_msg.append(f"Mean Precision: {metrics['mean_precision']:.2f}%")
    log_msg.append(f"Mean F1: {metrics['mean_f1']:.2f}%")
    print(f"\nPer-class metrics:")
    for c in range(len(label_map)):
        print(f"  Class {c}-{label_map[c]}: Recall={metrics['class_recall'][c]:.2f}%, "
            f"Precision={metrics['class_precision'][c]:.2f}%, "
            f"F1={metrics['class_f1'][c]:.2f}%")
    print("\n")
    log_msg = '\n'.join(log_msg)
    return log_msg