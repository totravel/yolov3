import argparse
import os
import torch
from torch import Tensor
from PIL.Image import Image as PILImage
from utils import load_images, get_device, box_cxcywh, box_xyxy, box_denorm, nms, save_bboxes
from yolo_model import YOLOv3
from yolo_data import Letterbox, VOC_CLASSES
from yolo_prior import YOLOv3Prior
from torchvision import transforms


def parse_args():
  parser = argparse.ArgumentParser(description='YOLOv3 Inference')
  parser.add_argument('--weights', type=str, default='checkpoints/yolo_best.pth', help='model weights')
  parser.add_argument('--img', type=str, default='data/img', help='input images')
  parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
  parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
  parser.add_argument('--out_dir', type=str, default='out', help='output directory')
  return parser.parse_args()


def preprocess(imgs: list[PILImage], img_size: tuple[int, int]):
  transform = transforms.Compose([
    Letterbox(img_size),
    transforms.ToTensor()
  ])
  return torch.stack([transform(img) for img in imgs])


def postprocess(preds: tuple[Tensor, Tensor, Tensor], fmap_anchors: list[Tensor], img_h: int, img_w: int, conf_thresh: float=0.5, nms_thresh: float=0.5):
  results: list[tuple[Tensor, Tensor, Tensor]] = []
  fmap_anchors = [box_cxcywh(a) for a in fmap_anchors]

  for fmaps in zip(*preds):
    bboxes, classes, scores = [], [], []

    for fmap_idx, fmap in enumerate(fmaps):
      fmap_h, fmap_w = fmap.shape[-2:]
      stride_h, stride_w = img_h/fmap_h, img_w/fmap_w

      # (C, FH, FW) -> (FH, FW, C) -> (FH*FW*3, 25)
      pred = fmap.permute(1, 2, 0).reshape(-1, 25)

      xy_pred = pred[..., 0:2]  # (FH*FW*3, 2)
      wh_pred = pred[..., 2:4]  # (FH*FW*3, 2)
      obj_pred = torch.sigmoid(pred[..., 4])  # (FH*FW*3,)
      cls_pred = pred[..., 5:]  # (FH*FW*3, 20)

      obj_mask = obj_pred >= conf_thresh  # (FH*FW*3, 2)
      num_objects = obj_mask.sum()
      if num_objects == 0:
        continue

      xy_pred = torch.sigmoid(xy_pred[obj_mask])  # (num_objects, 2)
      wh_pred = torch.exp(wh_pred[obj_mask])  # (num_objects, 2)
      cls_prob, cls_idx = torch.sigmoid(cls_pred[obj_mask]).max(dim=-1)  # (num_objects,)

      anchors = fmap_anchors[fmap_idx][obj_mask]  # (num_objects, 4)
      anchors[..., :2] += xy_pred
      anchors[..., 2:] *= wh_pred
      anchors = box_xyxy(anchors)
      anchors[..., [0, 2]].clamp_(min=0, max=fmap_w)
      anchors[..., [1, 3]].clamp_(min=0, max=fmap_h)
      anchors = box_denorm(anchors, stride_w, stride_h)

      bboxes.append(anchors)
      classes.append(cls_idx)
      scores.append(cls_prob)

    bboxes = torch.cat(bboxes)
    classes = torch.cat(classes)
    scores = torch.cat(scores)

    keep = nms(bboxes, scores, nms_thresh)
    results.append((bboxes[keep], classes[keep], scores[keep]))

  return results


def main():
  args = parse_args()

  if not os.path.exists(args.weights):
    raise FileNotFoundError(f'no such file: {args.weights}')
  print(f'model: {args.weights}')

  imgs, img_fnames = load_images(args.img)
  num_imgs = len(imgs)
  if num_imgs == 0:
    raise FileNotFoundError(f'no images found')
  print(f'{num_imgs} images loaded')

  device = get_device()
  print(f'device: {device}')

  weights = torch.load(args.weights, map_location=device)
  model = YOLOv3().to(device)
  model.load_state_dict(weights['model'])
  val_loss = weights['val_loss']
  print(f'model loaded (val_loss {val_loss:.4f})')
  
  prior = YOLOv3Prior(device)
  out_size = prior.img_h, prior.img_w

  x = preprocess(imgs, out_size).to(device)
  model.eval()
  with torch.no_grad():
    fmaps = model(x)
  
  results = postprocess(fmaps, prior.fmap_anchors, prior.img_h, prior.img_w)
  
  os.makedirs(args.out_dir, exist_ok=True)
  for idx, (result, fname) in enumerate(zip(results, img_fnames)):
    bboxes, classes, scores = result

    img = imgs[idx]
    img_size = img.size[1], img.size[0]
    bboxes = Letterbox.reverse_bbox(bboxes, img_size, out_size).cpu()

    labels = [f'{VOC_CLASSES[c]} {s:.0%}' for c, s in zip(classes, scores)]
    print(f'#{idx} {fname} - {', '.join(labels)}')
    
    out_path =  os.path.join(args.out_dir, f'{fname}.png')
    save_bboxes(img, bboxes, labels, out_path)


if __name__ == '__main__':
  main()
