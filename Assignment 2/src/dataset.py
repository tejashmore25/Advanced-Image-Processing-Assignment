import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
import cv2
from collections import defaultdict

class FootballPlayerDataset(Dataset):
    def __init__(self, image_dir, annotation_path, train_test_split_path, split='train', transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = self.transform.transforms[0].size

        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        with open(train_test_split_path, 'r') as f:
            splits = json.load(f)
        
        # image_id -> image metadata for the split = train or test
        self.valid_filenames = set(splits.get(split, []))
        self.images_info = {}
        for img in self.coco_data.get('images', []):
            if img['file_name'] in self.valid_filenames:
                self.images_info[img['id']] = img
        
        self.image_ids = list(self.images_info.keys())

        self.annotations_map = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            if ann['image_id'] in self.images_info:
                self.annotations_map[ann['image_id']].append(ann)
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        file_name = img_info['file_name']

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        height = img_info['height']
        width = img_info['width']
        mask_np = np.zeros((height, width), dtype = np.uint8)

        anns = self.annotations_map[img_id]
        for ann in anns:
            category_id = ann['category_id']
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_np, [poly], color = category_id)
        
        if self.transform:
            image = self.transform(image)

        mask_resized = cv2.resize(mask_np, self.target_size, interpolation=cv2.INTER_NEAREST)

        mask = torch.tensor(mask_resized, dtype = torch.long)
        return image, mask
        
