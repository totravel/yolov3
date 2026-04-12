import argparse
import os
import torch
from torch import Tensor
from PIL.Image import Image as PILImage
from utils import load_images, get_device, box_cxcywh, box_xyxy, box_denorm, save_bboxes
from yolo_model import YOLOv3
from yolo_data import Letterbox, VOC_CLASSES
from yolo_prior import YOLOv3Prior
from torchvision import transforms
from torchvision.ops import batched_nms


def parse_args():
  parser = argparse.ArgumentParser(description='YOLOv3 Inference')
  parser.add_argument('--weights', type=str, default='out/checkpoints/yolo_best.pth', help='model weights')
  parser.add_argument('--img', type=str, default='data/img', help='input images')
  parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
  parser.add_argument('--nms_thresh', type=float, default=0.4, help='nms threshold')
  parser.add_argument('--out_dir', type=str, default='out/img', help='output directory')
  return parser.parse_args()


def preprocess(imgs: list[PILImage], img_size: tuple[int, int]):
  transform = transforms.Compose([
    Letterbox(img_size),
    transforms.ToTensor()
  ])
  return torch.stack([transform(img) for img in imgs])


def postprocess(preds: tuple[Tensor, Tensor, Tensor], prior: YOLOv3Prior, conf_thresh: float=0.5, nms_thresh: float=0.5):
  results: list[tuple[Tensor, Tensor, Tensor]] = []

  for fmaps in zip(*preds):
    # results of each fmaps
    bboxes, classes, scores = [], [], []

    for size_idx, pred in enumerate(fmaps):
      fmap_h, fmap_w = pred.shape[-2:]
      stride_h, stride_w = prior.img_h/fmap_h, prior.img_w/fmap_w

      # (C, FH, FW) -> (FH, FW, C) -> (FH*FW*3, 25)
      pred = pred.permute(1, 2, 0).reshape(-1, 25)
      pred_offsets = pred[..., :4]  # (FH*FW*3, 4)
      pred_obj = torch.sigmoid(pred[..., 4])  # (FH*FW*3,)
      pred_scores, pred_classes = torch.sigmoid(pred[..., 5:]).max(dim=-1)

      pred_conf = pred_obj * pred_scores
      obj_mask = pred_conf >= conf_thresh  # (FH*FW*3, 20)
      num_objects = obj_mask.sum()
      if num_objects == 0:
        continue

      # (num_objects, 4)
      prior_bboxes = box_cxcywh(prior.prior_bboxes[size_idx][obj_mask])
      pred_bboxes = YOLOv3Prior.decode_pred_bbox(pred_offsets[obj_mask], prior_bboxes)
      pred_bboxes = box_denorm(box_xyxy(pred_bboxes), stride_w, stride_h)
      pred_bboxes[..., [0, 2]] = pred_bboxes[..., [0, 2]].clamp(min=0, max=prior.img_w)
      pred_bboxes[..., [1, 3]] = pred_bboxes[..., [1, 3]].clamp(min=0, max=prior.img_h)

      bboxes.append(pred_bboxes)
      classes.append(pred_classes[obj_mask])
      scores.append(pred_conf[obj_mask])
    
    bboxes = torch.cat(bboxes)
    classes = torch.cat(classes)
    scores = torch.cat(scores)
    # The operator 'torchvision::nms' is not currently implemented for the MPS device.
    # keep = batched_nms(bboxes, scores, classes, nms_thresh)
    keep = batched_nms(bboxes.cpu(), scores.cpu(), classes.cpu(), nms_thresh)
    results.append((bboxes[keep], classes[keep], scores[keep]))
  return results


def main():
  args = parse_args()

  if not os.path.exists(args.weights):
    print(f'model not found: {args.weights}')
    return
  print(f'model: {args.weights}')

  imgs, img_fnames = load_images(args.img)
  if not imgs:
    print(f'no images found ({args.img})')
    return
  print(f'{len(imgs)} images loaded')

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
    preds = model(x)
  
  results = postprocess(preds, prior, args.conf_thresh, args.nms_thresh)
  
  os.makedirs(args.out_dir, exist_ok=True)
  for idx, (result, fname) in enumerate(zip(results, img_fnames)):
    bboxes, classes, scores = result

    img = imgs[idx]
    img_size = img.size[1], img.size[0]
    bboxes = Letterbox.reverse_bbox(bboxes, img_size, out_size).cpu()
    labels = [f'{VOC_CLASSES[c]} {s:.0%}' for c, s in zip(classes, scores)]
    out_path =  os.path.join(args.out_dir, f'{fname}.png')
    save_bboxes(img, bboxes, labels, out_path)

    labels = ', '.join(labels)
    print(f'#{idx} {fname} - {labels}')


if __name__ == '__main__':
  main()
