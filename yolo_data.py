import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from os import path
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import transforms
import random


VOC_CLASSES = [
  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
  'bus', 'car', 'cat', 'chair', 'cow',
  'diningtable', 'dog', 'horse', 'motorbike', 'person',
  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
  """
  Args:
    voc_root (str): path to VOCdevkit
    voc_list (list[tuple[str, list]]): e.g. [('2007', ['trainval', 'test']), ('2012', ['trainval'])]
  """
  def __init__(self, voc_root: str, voc_list: list[tuple[str, list]]=[('2007', ['train'])], transform=None):
    super().__init__()

    self.img_files, self.label_files = self._load_files(voc_root, voc_list)
    self.length = len(self.img_files)

    self.transform=transform if transform is not None else lambda *x: x
  
  def _load_files(self, voc_root, voc_list):
    img_files, label_files = [], []

    for year, splits in voc_list:
      voc_dir = path.join(voc_root, f'VOC{year}')

      for split in splits:
        split_file = path.join(voc_dir, 'ImageSets', 'Main', f'{split}.txt')
        
        with open(split_file, 'r') as f:
          img_names = [clean_line for line in f if (clean_line := line.strip().lower())]
        
        for img_id in img_names:
          img_file = path.join(voc_dir, 'JPEGImages', f'{img_id}.jpg')
          label_file = path.join(voc_dir, 'Annotations', f'{img_id}.xml')

          if path.exists(img_file) and path.exists(label_file):
            img_files.append(img_file)
            label_files.append(label_file)

    return img_files, label_files
  
  def __len__(self):
    return self.length
  
  def __getitem__(self, idx):
    idx = idx % self.length
    img_file, label_file = self.img_files[idx], self.label_files[idx]

    img = Image.open(img_file).convert('RGB')
    label = self._parse_xml(label_file)
    
    return self.transform(img, label)

  def _parse_xml(self, label_file):
    root = ET.parse(label_file).getroot()
    
    bboxes, classes = [], []
    for obj in root.iter('object'):
      
      bbox = obj.find('bndbox')
      x1 = int(bbox.find('xmin').text)
      y1 = int(bbox.find('ymin').text)
      x2 = int(bbox.find('xmax').text)
      y2 = int(bbox.find('ymax').text)
      
      cls = obj.find('name').text
      if cls not in VOC_CLASSES:
        continue
      cls_idx = CLASS_TO_IDX[cls]
      
      bboxes.append([x1, y1, x2, y2])
      classes.append(cls_idx)
    return torch.tensor(bboxes, dtype=torch.float), torch.tensor(classes)


class RandomLetterbox:
  def __init__(self, out_size: tuple[int, int]=(416, 416), rand_scale: bool=False, rand_offset: bool=False):
    self.out_size = out_size
    self.rand_scale = rand_scale
    self.rand_offset = rand_offset

  def __call__(self, img: PILImage):
    return self.resize_img(img)[0]

  def resize_img(self, img: PILImage):
    img_w, img_h = img.size
    out_h, out_w = self.out_size
    scale = min(out_w / img_w, out_h / img_h)
    if self.rand_scale:
      scale *= random.uniform(0.7, 1)
    resized_w, resized_h = int(img_w * scale), int(img_h * scale)

    pad_w = out_w - resized_w
    pad_h = out_h - resized_h
    if self.rand_offset:
      offset_x = int(pad_w * random.random())
      offset_y = int(pad_h * random.random())
    else:
      offset_x = pad_w // 2
      offset_y = pad_h // 2

    resized_img = img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    img = Image.new('RGB', (out_w, out_h), (128, 128, 128))
    img.paste(resized_img, (offset_x, offset_y))
    return img, scale, offset_x, offset_y
  
  def resize_label(self, label: tuple[Tensor, Tensor], scale, offset_x, offset_y):
    bboxes, classes = label
    if len(bboxes) > 0:
      bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + offset_x
      bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + offset_y

      bboxes_w = bboxes[:, 2] - bboxes[:, 0]
      bboxes_h = bboxes[:, 3] - bboxes[:, 1]
      keep = (bboxes_w > 1) & (bboxes_h > 1)

      bboxes = bboxes[keep]  # discard invalid bboxes
      classes = classes[keep]
    return bboxes, classes


class Letterbox(RandomLetterbox):
  def __init__(self, out_size: tuple[int, int]=(416, 416)):
    super().__init__(out_size=out_size, rand_scale=False, rand_offset=False)
  
  @staticmethod
  def reverse_bbox(bboxes: Tensor, img_size: tuple[int, int], out_size: tuple[int, int]):
    img_h, img_w = img_size
    out_h, out_w = out_size
    scale = min(out_w / img_w, out_h / img_h)
    resized_w, resized_h = int(img_w * scale), int(img_h * scale)
    
    pad_w = out_w - resized_w
    pad_h = out_h - resized_h
    offset_x = pad_w // 2
    offset_y = pad_h // 2

    if len(bboxes) > 0:
      bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - offset_x) / scale
      bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - offset_y) / scale
    return bboxes


class VOCRandomLetterbox:
  def __init__(self, out_size: tuple[int, int]=(416, 416), rand_scale=True, rand_offset=True):
    self.impl = RandomLetterbox(out_size, rand_scale, rand_offset)

  def __call__(self, img: PILImage, label: tuple[Tensor, Tensor]):
    img, scale, offset_x, offset_y = self.impl.resize_img(img)
    label = self.impl.resize_label(label, scale, offset_x, offset_y)
    return img, label


class VOCLetterbox(VOCRandomLetterbox):
  def __init__(self, out_size: tuple[int, int]=(416, 416)):
    super().__init__(out_size=out_size, rand_scale=False, rand_offset=False)


class VOCRandomHorizontalFlip:
  def __init__(self, p: float=.5):
    self.p = p

  def __call__(self, img: PILImage, label: tuple[Tensor, Tensor]):
    if random.random() < self.p:
      img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
      bboxes, classes = label
      if len(bboxes) > 0:
        bboxes[:, [0, 2]] = img.size[0] - bboxes[:, [2, 0]]
      label = bboxes, classes
    return img, label


class VOCRandomColorJitter:
  def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
    self.jitter = transforms.ColorJitter(brightness, contrast, saturation)

  def __call__(self, img: PILImage, label: tuple[Tensor, Tensor]):
    img = self.jitter(img)
    return img, label


class VOCToTensor:
  def __init__(self):
    self.transform = transforms.ToTensor()

  def __call__(self, img: PILImage, label: tuple[Tensor, Tensor]):
    img = self.transform(img)
    return img, label


class VOCCompose:
  def __init__(self, transforms: list):
    self.transforms = transforms

  def __call__(self, img, label):
    for t in self.transforms:
      img, label = t(img, label)
    return img, label


def voc_collate(batch):
  """
  [(img, label)] -> (imgs, [label])
  """
  imgs = []
  labels = []
  for img, label in batch:
    imgs.append(img)
    labels.append(label)
  imgs = torch.stack(imgs)
  return imgs, labels


def voc_loader(voc_root: str, img_size: int=416, batch_size=32):
  train_transform = VOCCompose([
    VOCRandomLetterbox((img_size, img_size)),
    VOCRandomHorizontalFlip(),
    VOCRandomColorJitter(),
    VOCToTensor()
  ])
  val_transform = VOCCompose([
    VOCLetterbox((img_size, img_size)),
    VOCToTensor()
  ])
  train_data = VOCDataset(voc_root, [('2007', ['trainval']), ('2012', ['trainval'])], train_transform)
  val_data = VOCDataset(voc_root, [('2007', ['test'])], val_transform)
  train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=voc_collate)
  val_loader = DataLoader(val_data, batch_size, shuffle=False, collate_fn=voc_collate)
  return train_loader, val_loader


if __name__ == '__main__':
  from utils import show_bboxes

  train_data = VOCDataset('data/VOCdevkit', transform=VOCCompose([
    VOCRandomLetterbox((416, 416)),
    VOCRandomHorizontalFlip(),
    VOCRandomColorJitter(),
    VOCToTensor()
  ]))
  print(f'len(train_data) {len(train_data)}')

  img, label = train_data[1]
  print(f'img.shape {img.shape}')
  print(f'len(label) {len(label)}')
  bboxes, cls = label
  print(f'bboxes.dtype {bboxes.dtype}')
  print(f'cls.dtype {cls.dtype}')
  
  show_bboxes(img.permute(1, 2, 0), bboxes, [VOC_CLASSES[c] for c in cls])

  train_loader = DataLoader(train_data, 4, shuffle=False, collate_fn=voc_collate)
  imgs, labels = next(iter(train_loader))
  print(f'imgs.shape {imgs.shape}')
  for img, (bboxes, cls) in zip(imgs, labels):
    print(f'  - {len(bboxes)} bboxes {[VOC_CLASSES[c] for c in cls]}')
