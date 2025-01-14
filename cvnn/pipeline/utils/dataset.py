from pathlib import Path
from typing import Tuple, List, Union, Dict

import numpy as np
import json 

import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, dataset_file: str, representation: str, drop_low: bool, partition: str):
        self.dataset_file = dataset_file
        self.representation = representation
        self.drop_low = drop_low
        self.partition = partition

        self.transform = None
        self.data = self.load_data()


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")


    def load_data(self):
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)
        
        data = [img for img in data if img['partition'] == self.partition]

        if self.drop_low:
            data = [img for img in data if img['confidence'] != 'LOW']
         
        return data
    
    def view_as_real_imag(self, img):
        real = np.real(img)
        imag = np.imag(img)
        return np.concatenate([real, imag], axis=0)

    
    def view_as_amp_phase(self, img):
        amp = np.abs(img)
        phase = np.angle(img)
        return np.concatenate([amp, phase], axis=0)
    
    def amp_only(self, img):
        amp = np.abs(img)
        return amp
    
    def phase_only(self, img):
        phase = np.angle(img)
        return phase
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_info = self.data[index]
        img_path = Path(img_info['image_folder'], img_info['filename'])
        label = img_info['class']

        
        image = np.load(img_path)

        if self.representation == 'real_imag':
            image = self.view_as_real_imag(image)

        elif self.representation == 'amp_phase':
            image = self.view_as_amp_phase(image)
        
        elif self.representation == 'amp_only':
            image = self.amp_only(image)
        
        elif self.representation == 'phase_only':
            image = self.phase_only(image)
            
        image = torch.from_numpy(image)
        label = torch.tensor(label)
        
        image = image.to(self.device)
        label = label.to(self.device)

        return image, label
    

class DetectionDataset(Dataset):
    def __init__(self, dataset_file: str, representation: str, drop_low: bool, drop_empty: bool, partition: str, ):
        self.dataset_file = dataset_file
        self.representation = representation
        self.drop_low = drop_low
        self.drop_empty = drop_empty
        self.partition = partition

        self.data = self.load_data()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

    def load_data(self):
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        data = [img for img in data if img['partition'] == self.partition]

        if self.drop_low:
            for img in data:
                img['targets'] = [target for target in img['targets'] if target['confidence'] != 'LOW']

        if self.drop_empty:
            data = [img for img in data if len(img['targets']) != 0]
        
        return data

    def rescale(self, img):
        if 'SNAP_data' in self.dataset_file.split('/'):
            img = np.stack((img[0]/0.291, img[1]/0.676))
            amps = np.abs(img)
            clipped_amps = np.minimum(amps, 1)
            
            img = img * clipped_amps / (amps + 1e-10)
            img * np.pi
            return img
        
        elif 'SARFish_data' in self.dataset_file.split('/'):
            img = np.stack((img[0]/86.683, img[1]/204.478))
            amps = np.abs(img)
            clipped_amps = np.minimum(amps, 1.)
            
            img = img * clipped_amps / (amps + 1e-10)
            img * np.pi
            return img
        
        else:
            return img

    
    def view_as_real_imag(self, img):
        real = np.real(img)
        imag = np.imag(img)
        return np.concatenate([real, imag], axis=0)

    
    def view_as_amp_phase(self, img):
        amp = np.abs(img)
        phase = np.angle(img)
        return np.concatenate([amp, phase], axis=0)

    def amp_only(self, img):
        amp = np.abs(img)
        return amp
    
    def phase_only(self, img):
        phase = np.angle(img)
        return phase
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_info = self.data[index]
        img_path = Path(img_info['image_folder'], img_info['filename'])
        img_targets = img_info['targets']

        
        image = np.load(img_path)
        image = self.rescale(image)

        if self.representation == 'real_imag':
            image = self.view_as_real_imag(image)

        elif self.representation == 'amp_phase':
            image = self.view_as_amp_phase(image)
        
        elif self.representation == 'amp_only':
            image = self.amp_only(image)
        
        elif self.representation == 'phase_only':
            image = self.phase_only(image)
            
        image = torch.from_numpy(image)
        
        targets = {}
        bboxes = torch.tensor([t['bbox'] for t in img_targets])
        labels = torch.tensor([t['class'] for t in img_targets])

        image = image.to(self.device)

        targets['boxes'] = bboxes.to(self.device)
        targets['labels'] = labels.to(self.device)

        return image, targets

def create_classification_dataset(dataset_file: str, representation: str, drop_low: bool, partition: str):
    return ClassificationDataset(dataset_file, representation, drop_low, partition)

def create_detection_dataset(dataset_file: str, representation: str, drop_low: bool, drop_empty: bool, partition: str):
    return DetectionDataset(dataset_file,  representation, drop_low, drop_empty, partition)

