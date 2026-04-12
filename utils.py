import os
from pathlib import Path
from torch import Tensor
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.patches import Rectangle
from PIL import Image
from PIL.Image import Image as PILImage
from random import uniform
import torch


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def is_image(path: str):
  return path.lower().endswith(IMAGE_EXTENSIONS)


def load_images(path: str) -> list[PILImage]:
  if not os.path.exists(path):
    raise FileNotFoundError(f'no such file or directory: {path}')
  imgs, fnames = [], []
  if os.path.isfile(path):
    if is_image(path):
      img = Image.open(path).convert('RGB')
      imgs.append(img)
      fnames.append(Path(path).stem)
  elif os.path.isdir(path):
    for filename in os.listdir(path):
      file = os.path.join(path, filename)
      if os.path.isfile(file) and is_image(file):
        img = Image.open(file).convert('RGB')
        imgs.append(img)
        fnames.append(Path(file).stem)
  return imgs, fnames


def get_device():
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif torch.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')
  return device


def init_weights(m: nn.Module):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)


def box_cxcywh(boxes: Tensor):
  """[x1, y1, x2, y2] -> [cx, cy, w, h]"""
  out_boxes = torch.zeros_like(boxes)
  out_boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
  out_boxes[..., :2] = boxes[..., :2] + 0.5 * out_boxes[..., 2:]
  return out_boxes


def box_xyxy(boxes: Tensor):
  """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
  out_boxes = torch.zeros_like(boxes)
  out_boxes[..., :2] = boxes[..., :2] - 0.5 * boxes[..., 2:]
  out_boxes[..., 2:] = out_boxes[..., :2] + boxes[..., 2:]
  return out_boxes


def box_origin_wh(boxes: Tensor):
  """[x1, y1, x2, y2] -> [0, 0, w, h]"""
  out_boxes = torch.zeros_like(boxes)
  out_boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
  return out_boxes


def box_norm(boxes: Tensor, stride_w: int, stride_h: int):
  boxes[..., [0, 2]] /= stride_w
  boxes[..., [1, 3]] /= stride_h
  return boxes


def box_denorm(boxes: Tensor, stride_w: int, stride_h: int):
  boxes[..., [0, 2]] *= stride_w
  boxes[..., [1, 3]] *= stride_h
  return boxes


def box_to_rect(box: Tensor, color='r', border=2):
  x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
  xy = (x1, y1)
  w = x2 - x1
  h = y2 - y1
  return Rectangle(xy, w, h, fill=False, edgecolor=color, linewidth=border)


def random_color():
  r = uniform(0, 1)
  g = uniform(0, 1)
  b = uniform(0, 1)
  return r, g, b


def plot_boxes(ax: Axes, boxes: list[Tensor], border=2) -> None:
  for box in boxes:
    ax.add_patch(box_to_rect(box, random_color(), border))


def show_boxes(img: PILImage, boxes: list[Tensor]) -> None:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.imshow(img)
  plot_boxes(ax, boxes)
  plt.show()


def plot_bboxes(ax: Axes, boxes: list[Tensor], labels: list[str], border=2, fontsize=10) -> None:
  for box, label in zip(boxes, labels):
    color = random_color()
    rect = box_to_rect(box, color, border)
    x, y = rect.xy
    ax.add_patch(rect)
    bbox=dict(facecolor=color, edgecolor=color, linewidth=border, pad=0)
    ax.text(x, y, label, bbox=bbox, fontsize=fontsize, backgroundcolor=color, color='w', va='bottom')


def show_bboxes(img: PILImage, boxes: list[Tensor], labels: list[str]) -> None:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.imshow(img)
  plot_bboxes(ax, boxes, labels)
  plt.show()


def save_bboxes(img: PILImage, boxes: list[Tensor], labels: list[str], path: str) -> None:
  img_w, img_h = img.size
  dpi = 100
  fig = plt.figure(figsize=(img_w/dpi, img_h/dpi))
  fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
  ax = fig.add_subplot()
  ax.set_axis_off()
  ax.imshow(img)
  plot_bboxes(ax, boxes, labels, 1, 10)
  fig.savefig(path)
  plt.close(fig)


def show_image(img: Tensor) -> None:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.imshow(img)
  plt.show()


def iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
  """
  Args:
    boxes1 (Tensor): (n, 4)
    boxes2 (Tensor): (m, 4)
  Returns:
    ious (Tensor): (n, m)
  """
  inter_lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
  inter_rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
  inter_sizes = torch.max(inter_rb - inter_lt, torch.zeros_like(inter_lt))
  inter_areas = inter_sizes[..., 0] * inter_sizes[..., 1]

  boxes_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  areas1 = boxes_area(boxes1)  # (n,)
  areas2 = boxes_area(boxes2)  # (m,)
  union_areas = areas1[:, None] + areas2[None, :] - inter_areas

  return inter_areas / union_areas


def box_iou(boxes1: Tensor, boxes2: Tensor, eps=1e-7) -> Tensor:
  inter_lt = torch.max(boxes1[..., :2], boxes2[..., :2])
  inter_rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
  inter_wh = (inter_rb - inter_lt).clamp(min=0.0)
  inter_area = inter_wh[..., 0] * inter_wh[..., 1]

  boxes1_wh = boxes1[..., 2:] - boxes1[..., :2]
  boxes2_wh = boxes2[..., 2:] - boxes2[..., :2]
  area1 = boxes1_wh[..., 0] * boxes1_wh[..., 1]
  area2 = boxes2_wh[..., 0] * boxes2_wh[..., 1]

  union_area = area1 + area2 - inter_area
  iou = inter_area / (union_area + eps)
  return iou


def box_giou(boxes1: Tensor, boxes2: Tensor, eps=1e-7) -> Tensor:
  inter_lt = torch.max(boxes1[..., :2], boxes2[..., :2])
  inter_rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
  inter_wh = (inter_rb - inter_lt).clamp(min=0.0)
  inter_area = inter_wh[..., 0] * inter_wh[..., 1]

  boxes1_wh = boxes1[..., 2:] - boxes1[..., :2]
  boxes2_wh = boxes2[..., 2:] - boxes2[..., :2]
  area1 = boxes1_wh[..., 0] * boxes1_wh[..., 1]
  area2 = boxes2_wh[..., 0] * boxes2_wh[..., 1]

  union_area = area1 + area2 - inter_area
  iou = inter_area / (union_area + eps)

  enclose_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
  enclose_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
  enclose_wh = (enclose_rb - enclose_lt).clamp(min=0.0)
  enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

  giou = iou - (enclose_area - union_area) / (enclose_area + eps)
  return giou
