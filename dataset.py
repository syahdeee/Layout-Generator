import os
import sys
import dgl
import PIL
import json
import math
import torch
import random
import torchvision.transforms as T
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)
    
class DatasetLoader(Dataset):
    def _build_vocab(self):
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }

        # build object_name_to_idx
        self.vocab['object_name_to_idx']['__poster__'] = 0
        self.vocab['object_name_to_idx']['info'] = 1
        self.vocab['object_name_to_idx']['logo'] = 2
        self.vocab['object_name_to_idx']['primary_image'] = 3
        self.vocab['object_name_to_idx']['subtitle'] = 4
        self.vocab['object_name_to_idx']['title'] = 5

        # build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # build pred_idx_to_name
        self.vocab['pred_idx_to_name'] = [
            '__in_poster__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]

        # build pred_name_to_idx
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def set_image_size(self, image_size):
        normalize_images=True
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def __init__(self, image_size = (64,64)):
        direktori_magazine = './dataset'
        ekstensi_file = '.json'
        def is_json_file(file):
            return file.endswith(ekstensi_file)

        files = os.listdir(direktori_magazine)
        json_files = filter(is_json_file, files)
        file_layout_json = map(lambda file: os.path.join(direktori_magazine, file), json_files)
        self._build_vocab()
        self.set_image_size(image_size)
        
        def _process_layout_item(layout_path):
            with open(layout_path) as f:
                layout = json.load(f)
                
            layout_data = []

            for category in layout['components'].keys():
                for item in layout['components'][category]:
                    temp_data = {
                        'category_id': self.vocab['object_name_to_idx'][category],
                        'bbox': item
                    }
                    layout_data.append(temp_data)

            objs, boxes = [], []
            WW = HH = 64.
            
            scale_factor_width = 64 / layout['width']
            scale_factor_height = 64 / layout['height']
            
            for object_data in layout_data:
                objs.append(object_data['category_id'])
                x0, y0, x1, y1 = object_data['bbox']
                
                x0 = x0 * scale_factor_width
                y0 = y0 * scale_factor_height
                x1 = x1 * scale_factor_width
                y1 = y1 * scale_factor_height

                boxes.append(torch.FloatTensor(
                     [x0 / WW, y0 / HH, x1 / WW, y1 / HH]))

            objs.append(self.vocab['object_name_to_idx']['__poster__'])
            boxes.append(torch.FloatTensor([0, 0, 1, 1]))

            objs = torch.LongTensor(objs)
            boxes = torch.stack(boxes, dim=0)
            
            obj_centers = []
            for i, obj_idx in enumerate(objs):
                x0, y0, x1, y1 = boxes[i]
                obj_centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
                
            obj_centers = torch.FloatTensor(obj_centers)

            triples = []
            __image__ = self.vocab['object_name_to_idx']['__poster__']
            
            for item_idx, item in enumerate(objs):
                choices = [obj for obj in range(len(objs)) if (
                    obj != item_idx and obj != __image__)]
                if len(choices) == 0:
                    break

                other = random.choice(choices)
                if random.random() > 0.5:
                    s, o = item_idx, other
                else:
                    s, o = other, item_idx
                sx0, sy0, sx1, sy1 = boxes[s]
                ox0, oy0, ox1, oy1 = boxes[o]
                d = obj_centers[s] - obj_centers[o]
                theta = math.atan2(d[1], d[0])
                
                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = 'surrounding'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = 'inside'
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = 'left of'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = 'above'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = 'right of'
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = 'below'
                p = self.vocab['pred_name_to_idx'][p]
                triples.append([s, p, o])
            triples = torch.LongTensor(triples)
            return objs, boxes, triples

        self.data = []

        for layout_path in file_layout_json:
            self.data.append(_process_layout_item(layout_path))

        self.num_samples = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples