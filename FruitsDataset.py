import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from xml.etree import ElementTree as et
import glob


def preprocess_img(img):
    img = torch.tensor(img).permute(2, 0 ,1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return img.to(device).float()

class FruitsDataset(Dataset):
    def __init__(self, root='fruit-images-for-object-detection/train_zip/train/', transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_paths = sorted(glob.glob(self.root + '/*.jpg'))
        self.xml_paths = sorted(glob.glob(self.root + '/*.xml'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        w, h = 224, 224
        img_path = self.img_paths[idx]
        xml_path = self.xml_paths[idx]
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        img = np.array(img.resize((w, h), resample=Image.BILINEAR))/255.
        xml = et.parse(xml_path)
        objects = xml.findall('object')
        labels = []
        boxes = []
        for obj in objects:
            label = obj.find('name').text
            labels.append(label)
            XMin = float(obj.find('bndbox').find('xmin').text)
            YMin = float(obj.find('bndbox').find('ymin').text)
            XMax = float(obj.find('bndbox').find('xmax').text)
            YMax = float(obj.find('bndbox').find('ymax').text)
            bbox = [XMin / W, YMin / H, XMax / W, YMax / H]
            bbox = (bbox * np.array([w, h, w, h])).astype(np.int16).tolist()
            boxes.append(bbox)
        target = {}

        labels = ['background', 'orange', 'apple', 'banana']
        label2targets = {l: t for t, l in enumerate(labels)}

        target['labels'] = torch.tensor([label2targets[label] for label in labels]).long()
        target['boxes'] = torch.tensor(boxes).float()
        img = preprocess_img(img)
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))


