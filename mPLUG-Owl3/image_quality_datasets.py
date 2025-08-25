"""
Image Quality Assessment Datasets
This module implements EVADataset and AVADataset classes for image quality assessment
with PyTorch DataLoader compatibility and proper transforms.
"""

import os
import csv
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EVADataset(Dataset):
    """EVA Dataset for image quality assessment"""
    
    def __init__(self, image_dir, label_file, pre_crop_size=272, image_size=224, transform=None):
        self.vmin = 1024.
        self.vmax = -1024.
        self.scores = []
        self.image_dir = image_dir
        self.image_size = image_size
        self.pre_crop_size = pre_crop_size
        self.label_dict = self._load_labels(label_file)
        self.image_ids = list(self.label_dict.keys())
        self.scores = np.array(self.scores)
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((pre_crop_size, pre_crop_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _load_labels(self, label_file):
        label_dict = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                image_id = row['image_id']
                score = float(row['score'])
                self.scores.append(score)
                label_dict[image_id] = score
                self.vmin = min(self.vmin, score)
                self.vmax = max(self.vmax, score)
        return label_dict

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        image_id = self.image_ids[idx]
        score = self.label_dict[image_id]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score, dtype=torch.float32)

    def get_all_samples(self):
        """获取所有样本的图片路径和分数"""
        image_paths = [os.path.join(self.image_dir, image_id + '.jpg') for image_id in self.image_ids]
        scores = [self.label_dict[image_id] for image_id in self.image_ids]
        return image_paths, scores


class AVADataset(Dataset):
    """AVA Dataset for image quality assessment"""
    
    def __init__(self, label_file, img_root, pre_crop_size=272, image_size=224, transform=None):
        """
        Args:
            label_file (str): 标签json文件路径
            img_root (str): 图片根目录
        """
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.img_root = img_root
        self.pre_crop_size = pre_crop_size
        self.image_size = image_size
        self.data = [x for x in data['files'] if 'image' in x and 'score' in x]
        self.image_ids = [os.path.basename(x['image']) for x in self.data]
        self.scores = np.array([float(x['score']) for x in self.data], dtype=np.float32)
        self.label_dict = {os.path.basename(x['image']): float(x['score']) for x in self.data}
        self.vmin = float(np.min(self.scores))
        self.vmax = float(np.max(self.scores))
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((pre_crop_size, pre_crop_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        item = self.data[idx]
        score = float(item['score'])
        
        # Load image
        image_path = os.path.join(self.img_root, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score, dtype=torch.float32)

    def get_all_samples(self):
        image_paths = [os.path.join(self.img_root, x['image']) for x in self.data]
        scores = [float(x['score']) for x in self.data]
        return image_paths, scores


def get_default_transforms(image_size=224, pre_crop_size=272, train=True):
    """Get default transforms for image preprocessing"""
    if train:
        return transforms.Compose([
            transforms.Resize((pre_crop_size, pre_crop_size)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((pre_crop_size, pre_crop_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, scores = zip(*batch)
    images = torch.stack(images, 0)
    scores = torch.stack(scores, 0)
    return images, scores