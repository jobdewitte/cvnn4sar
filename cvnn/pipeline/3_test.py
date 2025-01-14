import argparse
from pathlib import Path


from utils.dataset import create_detection_dataset
from utils.metric import calculate_iou, calculate_p_r_f1
from utils.architectures.cxmodels import complex_fcos_resnet_fpn
from utils.architectures.rlmodels import real_fcos_resnet_fpn
import torch
import numpy as np

import pickle
from tqdm import tqdm 

from torchvision.ops import boxes as box_ops

def parse_args():
    parser = argparse.ArgumentParser(description="performs testing of a complex model for SARFish dataset")

    parser.add_argument(
        '--dataset_file', type=str,
        help='path/to/dataset.json'
        )
    
    parser.add_argument(
        '--representation', choices=['complex', 'real_imag', 'amp_phase', 'amp_only', 'phase_only'], type=str,
        help='representation of the data'
        )
    
    parser.add_argument(
        '--drop_low', choices=[0,1], type=int,
        help='if 1: drop detections that have confidence = `LOW`, if 0: keep them.'
        )
    
    parser.add_argument(
        '--drop_empty', choices=[0,1], type=int,
        help='if 1: drop images that have no detections, if 0: keep them.'
        )
    
    parser.add_argument(
        '--backbone_name', choices=['resnet18', 'resnet34', 'resnet50'], type=str,
        help='backbone for feature extraction'
    )

    parser.add_argument(
        '--trained_model_path', type=str,
        help='path/to/saved/model'
        )
    
    parser.add_argument(
        '--tensorboard_folder', type=str,
        help='path/to/tensorboard/log'
    )

    parser.add_argument(
        '--model_to_use', type=str,
        help='which trained model to use as tested model'
        )
    
    parser.add_argument(
        '--score_thresholds', type = str,
        help='thresholds for confidence score of predictions'
        )
    
    parser.add_argument(
        '--iou_threshold', type=float,
        help='threshold for iou of groundtruth and prediction for true positive detection'
        )
    
    parser.add_argument(
        '--save', choices=["true", "false"], type=str,
        help='save test logs'
        )
    

    
    return parser.parse_args()


def main():
    
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_fn = Path(args.trained_model_path, f"{args.model_to_use}.pth")

    if args.representation == 'complex':
        model = complex_fcos_resnet_fpn(
            args.backbone_name, in_channels=2, num_classes=4,frozen_backbone=None, nms_thresh=0.7
            )
    
    elif args.representation in ['amp_phase', 'real_imag']:
        model = real_fcos_resnet_fpn(
            args.backbone_name, in_channels=4, num_classes=4, frozen_backbone=None,
            image_mean=[0,0,0,0], image_std=[1,1,1,1], nms_thresh=0.7
            )
    
    elif args.representation in ['amp_only', 'phase_only']:
        model = real_fcos_resnet_fpn(
            args.backbone_name, in_channels=2, num_classes=4, frozen_backbone=None,
            image_mean=[0,0], image_std=[1,1], nms_thresh=0.7
            )


    model.load_state_dict(torch.load(str(model_fn), map_location = device))
    model.to(device)
    model.eval()

    test_data = create_detection_dataset(
        args.dataset_file, args.representation, args.drop_low, args.drop_empty, partition='test'
        )
    
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size = 1, sampler = test_sampler, num_workers = 0
        )
    
    print('### dataset info ###')
    print(f'number of testing samples: {test_data.__len__()}')
    print(args.representation)
    print(args.dataset_file)

    score_thresholds = [float(th) for th in args.score_thresholds.split(',')]
    
    metrics_log = {'score_threshold':[], 'precision': [], 'recall': [], 'f1_score': [], 'confusion_matrix': []}

    for i,score_threshold in enumerate(score_thresholds):
        print(f'score threshold: {score_threshold}')
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        confusion_matrix = np.zeros((4,4))

        torch.no_grad()
        for test_id, (test_image, test_targets) in enumerate(tqdm(test_data_loader, desc=f'testing... ({i+1}/{len(score_thresholds)})')):
            
            prediction = model(test_image)
            
            predicted_boxes = prediction[0]['boxes']
            scores = prediction[0]['scores']
            predicted_classes= prediction[0]['labels']

            #perform non_class based nms
            keep = box_ops.nms(predicted_boxes, scores, 0.1)
            predicted_boxes = predicted_boxes[keep]
            predicted_classes = predicted_classes[keep]
            scores = scores[keep]

            predicted_boxes = predicted_boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predicted_classes = predicted_classes.cpu().detach().numpy()
            
            groundtruth_boxes = test_targets['boxes'].squeeze(0).cpu().detach().numpy()
            groundtruth_classes = test_targets['labels'].squeeze(0).cpu().detach().numpy()

            num_detect = len(predicted_boxes) 
            
            if num_detect == 0:
                total_false_negatives += len(groundtruth_boxes)
                continue
            
            #Only output scores larger than threshold
            is_good_score = scores > score_threshold
            predicted_boxes = predicted_boxes[is_good_score, :]
            predicted_classes = predicted_classes[is_good_score]
            scores = scores[is_good_score]
            image_true_positives = 0
            image_false_positives = 0
            image_false_negatives = 0

            matched_gt = set()
            matched_pred = set()

            for gt_id, gt_box in enumerate(groundtruth_boxes):
                max_iou_for_gt = 0
                best_pred_id = None
                for pr_id, pr_box in enumerate(predicted_boxes):
                    iou_for_pr_gt = calculate_iou(pr_box, gt_box)
                    if iou_for_pr_gt > max_iou_for_gt:
                        max_iou_for_gt = iou_for_pr_gt
                        best_pred_id = pr_id

                if max_iou_for_gt > args.iou_threshold:
                    image_true_positives += 1
                    matched_gt.add(gt_id)
                    matched_pred.add(best_pred_id)

                    pred_class = predicted_classes[best_pred_id]
                    gt_class = groundtruth_classes[gt_id]
                    confusion_matrix[pred_class, gt_class] += 1
        
                    
            image_false_positives = len(predicted_boxes) - len(matched_pred)
            image_false_negatives = len(groundtruth_boxes) - len(matched_gt)
            
            total_true_positives += image_true_positives
            total_false_positives += image_false_positives
            total_false_negatives += image_false_negatives

        print(total_true_positives, total_false_positives, total_false_negatives)
        precision, recall, f1_score = calculate_p_r_f1(total_true_positives, total_false_positives, total_false_negatives)
        print(f'precision: {round(precision, 3)}')
        print(f'recall: {round(recall, 3)}')
        print(f'f1: {round(f1_score, 3)}')
        metrics_log['score_threshold'].append(score_threshold)
        metrics_log['precision'].append(precision)
        metrics_log['recall'].append(recall)
        metrics_log['f1_score'].append(f1_score)
        metrics_log['confusion_matrix'].append(confusion_matrix)

    if args.save=="true":
        with open(Path(args.tensorboard_folder, f'{str(args.iou_threshold).replace(".","_")}.pkl'), 'wb') as file:
            pickle.dump(metrics_log, file)
            
if __name__ == "__main__":
    main()