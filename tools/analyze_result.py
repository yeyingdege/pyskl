import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_results(file_path):
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
    if isinstance(label_map, str):
        label_map = [x.strip() for x in open(label_map).readlines()]
    id_to_label = {i: label for i, label in enumerate(label_map)}
    # compute mean accuracy
    num_classes = len(label_map)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    y_preds = []
    for pred, gt in zip(results, gt_labels):
        class_total[gt] += 1
        y = np.argmax(pred)
        y_preds.append(y)
        if y == gt:
            class_correct[gt] += 1
    class_acc = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    mean_acc = sum(class_acc) / num_classes
    print(f'Accuracy: {sum(class_correct) / sum(class_total):.3f}, Mean Accuracy: {mean_acc:.3f}')
    print("Accuracy of each class:")
    for i, acc in enumerate(class_acc):
        print(f'  {id_to_label[i]}: {acc:.3f} ({class_correct[i]}/{class_total[i]})')
    return gt_labels, y_preds


def plot_confusion_matrix(cm, labels, title='Confusion Matrix (Normalized)', cmap=plt.cm.Blues, fmt=".2f"):
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        cm (np.ndarray): The confusion matrix (normalized or raw).
        labels (list): List of class names corresponding to the matrix indices.
        title (str): Title of the plot.
        cmap (matplotlib.colors.Colormap): Color map to use for the heatmap.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,              # Show the values in the cells
        fmt=fmt,                 # Format the numbers to two decimal places
        cmap=cmap,               # Use the specified color map
        linewidths=.5,           # Add lines between cells
        square=True,             # Make cells square
        cbar=True,               # Show color bar
        xticklabels=labels,      # Set x-axis labels (Predicted)
        yticklabels=labels       # Set y-axis labels (True)
    )

    # Adding labels and title
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(title.replace(" ", "_").lower() + '.png')
    plt.close()


def main(pred_file: str, ann_file: str, label_map: str):
    LABELS = [x.strip() for x in open(label_map).readlines()]
    gt_labels, y_preds = compute_acc(ann_file, pred_file, LABELS)

    NUM_CLASSES = len(LABELS)
    # Calculate the raw confusion matrix
    # cm[i, j] is the number of samples with true label i that were predicted as label j.
    cm = confusion_matrix(gt_labels, y_preds, labels=range(NUM_CLASSES))

    # Calculate the normalized confusion matrix (by row, to show recall/accuracy per class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 1. Plot the Normalized Confusion Matrix (Shows Recall)
    plot_confusion_matrix(
        cm_normalized,
        LABELS,
        title='Normalized Confusion Matrix (Recall per Class)',
        cmap=sns.light_palette("seagreen", as_cmap=True) # Use a custom palette
    )
    # 2. Plot the Raw Confusion Matrix (Optional, for counts)
    plot_confusion_matrix(
        cm,
        LABELS,
        title='Raw Confusion Matrix (Counts)',
        cmap=plt.cm.Oranges,
        fmt="d"
    )


if __name__ == '__main__':
    pred_file = 'work_dirs/stgcn++/stgcn++_dtc_v2_yolo11/j.py/best_pred.pkl'
    ann_file = 'data/DTC/dtc7.pkl'
    label_map = 'tools/data/label_map/dtc7.txt'
    main(pred_file, ann_file, label_map)
