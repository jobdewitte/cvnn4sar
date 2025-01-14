import os
import argparse
import datetime
from pathlib import Path
import torch
import torch.utils.tensorboard
import csv
from tqdm import tqdm

from utils.dataset import create_detection_dataset
from utils.architectures.cxmodels import complex_fcos_resnet_fpn
from utils.architectures.rlmodels import real_fcos_resnet_fpn


def collate_fn(batch):
    return tuple(zip(*batch))

def print_allocated_memory():
    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))

def parse_args():
    parser = argparse.ArgumentParser(description="performs training of a complex model for SARFish dataset")

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
        '--frozen_backbone', type=str,
        help='path/to/pretrained/backbone or `None`'
    )

    parser.add_argument(
        '--num_epochs', type=int,
        help='number of training epochs'
    )

    parser.add_argument(
        '--batch_size', type=int,
        help='number of images per batch'
    )

    parser.add_argument(
        '--save', choices=["true", "false"], type=str,
        help='save model and training logs'
    )


    return parser.parse_args()

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('device:', device)

    if not Path(args.trained_model_path).exists():
        Path(args.trained_model_path).mkdir(parents=True, exist_ok=True)

    if not Path(args.tensorboard_folder).exists():
        Path(args.tensorboard_folder).mkdir(parents=True, exist_ok=True)
    
    train_data = create_detection_dataset(
        args.dataset_file, args.representation, bool(args.drop_low), bool(args.drop_empty), partition='train'
        )
    
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size, sampler = train_sampler, 
        num_workers = 0, collate_fn = collate_fn
        )
 
    val_data = create_detection_dataset(
        args.dataset_file, args.representation, bool(args.drop_low), bool(args.drop_empty), partition='validation'
        )
    
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    val_data_loader = torch.utils.data.DataLoader(
        val_data, batch_size = args.batch_size, sampler = val_sampler, 
        num_workers = 0, collate_fn = collate_fn
        )

    frozen_backbone = args.frozen_backbone if not args.frozen_backbone == 'None' else None

    if args.representation == 'complex':
        model = complex_fcos_resnet_fpn(
            args.backbone_name, in_channels=2, num_classes=4,frozen_backbone=frozen_backbone, nms_thresh=0.5
            )
        
    
    elif args.representation in ['amp_phase', 'real_imag']:
        model = real_fcos_resnet_fpn(
            args.backbone_name, in_channels=4, num_classes=4, frozen_backbone=frozen_backbone,
            image_mean=[0,0,0,0], image_std=[1,1,1,1], nms_thresh=0.5
            )
    
    elif args.representation in ['amp_only', 'phase_only']:
        model = real_fcos_resnet_fpn(
            args.backbone_name, in_channels=2, num_classes=4, frozen_backbone=frozen_backbone,
            image_mean=[0,0], image_std=[1,1], nms_thresh=0.5
            )

        
    print('### info ###')
    print(args.representation)
    print(args.dataset_file)
    print(f'number of training samples: {train_data.__len__()}')
    print(f'number of validation samples: {val_data.__len__()}')
    model.to(device)

    train_num_batch = len(train_data_loader)
    val_num_batch = len(val_data_loader)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr = 1e-5, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_num_batch * args.num_epochs
    )
    
    tb_id = 0
    val_tb_id = 0

    val_best_loss = float('inf')

    train_losses_log = [['epoch_id', 'batch_id', 'train_loss']]
    val_losses_log =[['epoch_id', 'batch_id', 'val_loss']]

    for epoch_id in range(args.num_epochs):
        
        if frozen_backbone is None:
            checkpoint_path = Path(
                args.trained_model_path, f"without_fb_checkpoint.pth"
            )
        else:
            checkpoint_path = Path(
                args.trained_model_path, f"with_fb_checkpoint.pth"
            )

        # TRAIN
        #
        model.train()
        
        train_all_loss = 0
        # DEBUG: (images, targets) = next(iter(train_data_loader))
        for batch_id, (train_images, train_targets) in enumerate(tqdm(train_data_loader, desc='training...')):
            
            train_loss_dict = model(train_images, train_targets)
            train_losses = sum(loss for loss in train_loss_dict.values())
            train_one_loss = train_losses.item()
            train_losses_log.append([epoch_id, batch_id, train_one_loss])
            train_all_loss += train_one_loss
            #
            # NOTE: losses.item() is loss_dict['classification'].item() + loss_dict['bbox_regression'].item() 
            #               + loss_dict['bbox_ctrness'].item()
            #
            optimizer.zero_grad()
            train_losses.backward()
            optimizer.step()
            lr_scheduler.step()

        # VALIDATION
        #
        model.train()  # should really be model.eval(), but to cater for the FCOS.forward() if self.training statement
        with torch.no_grad():
            val_all_loss = 0
            
            for val_id, (val_images, val_targets) in enumerate(tqdm(val_data_loader, desc='validating...')):

                val_loss_dict = model(val_images, val_targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_one_loss = val_losses.item()
                val_losses_log.append([epoch_id, val_id, val_one_loss])
                val_all_loss += val_one_loss

        #
        # SAVE MODEL
        #

        print(f'average train loss: {round(train_all_loss / train_data.__len__(), 6)}')
        print(f'average validation loss: {round(val_all_loss / val_data.__len__(), 6)}')
        
        if val_all_loss < val_best_loss:
            best_epoch = epoch_id
            if args.save=="true":
                torch.save(model.state_dict(), str(checkpoint_path))
                print(f'saved current best model: epoch {best_epoch}')
                val_best_loss = val_all_loss

    print(f'best_model: epoch {best_epoch}')
    #

    # Save the tensorboard
    if args.save == "true":
        with open(Path(args.tensorboard_folder, 'train_loss_log.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train_losses_log)

        with open(Path(args.tensorboard_folder, 'val_loss_log.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(val_losses_log)
    
    

    
    print("Training complete!")


if __name__ == "__main__":
    main()