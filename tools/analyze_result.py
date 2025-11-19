import pickle
import numpy as np


def load_results(file_path):
    """Load results from a pickle file.
    Args:
        file_path (str): The path of the pickle file.
    Returns:
        list: The loaded results.
    """
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results


def load_gt_labels(file_path):
    """Load ground truth labels from a pickle file.
    Args:
        file_path (str): The path of the pickle file.
    Returns:
        list: The loaded ground truth labels.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    split_ids = data['split']['val']
    annotations = data['annotations']
    anns = [x for x in annotations if x['filename'] in split_ids]
    for i in range(len(split_ids)):
        assert anns[i]['filename'] == split_ids[i]

    gt_labels = [ann['label'] for ann in anns]
    return gt_labels


def compute_acc(ann_file, pred_file, label_map):
    results = load_results(pred_file)
    gt_labels = load_gt_labels(ann_file)
    assert len(results) == len(gt_labels), "Results and ground truth labels length mismatch"

    # load label_map
    label_map = [x.strip() for x in open(label_map).readlines()]
    id_to_label = {i: label for i, label in enumerate(label_map)}
    # compute mean accuracy
    num_classes = len(label_map)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for pred, gt in zip(results, gt_labels):
        class_total[gt] += 1
        if np.argmax(pred) == gt:
            class_correct[gt] += 1
    class_acc = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    mean_acc = sum(class_acc) / num_classes
    print(f'Accuracy: {sum(class_correct) / sum(class_total):.3f}, Mean Accuracy: {mean_acc:.3f}')
    print("Accuracy of each class:")
    for i, acc in enumerate(class_acc):
        print(f'  {id_to_label[i]}: {acc:.3f} ({class_correct[i]}/{class_total[i]})')
    return gt_labels, results


if __name__ == '__main__':
    pred_file = 'work_dirs/stgcn++/stgcn++_dtc_v2_yolo11/j.py/best_pred.pkl'
    ann_file = 'data/DTC/dtc7.pkl'
    label_map = 'tools/data/label_map/dtc7.txt'
    compute_acc(ann_file, pred_file, label_map)
