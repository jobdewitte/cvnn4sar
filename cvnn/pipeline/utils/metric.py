import torch

def calculate_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    return correct

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area
    return iou

def calculate_p_r_f1(true_pos, false_pos, false_neg):

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0. else 0.

    return precision, recall, f1
    
