import torch
from torch import Tensor
from torch.nn import functional as F
from yolo_prior import YOLOv3Prior
from utils import box_cxcywh, iou_matrix
from utils import show_image
import math


class YOLOv3Loss(YOLOv3Prior):
  def __init__(self, device=torch.device('cpu')):
    super().__init__(device)
    self.num_classes = 20
    self.ignore_threshold = 0.4
    self.bbox_loss_weight = 5
    self.obj_loss_weight = 1
    self.cls_loss_weight = 1
    self.noobj_loss_weight = 0.5

  def __call__(self, fmaps: tuple[Tensor, Tensor, Tensor], labels: list[tuple[Tensor, Tensor]]):
    batch_size = len(labels)
    batch_assigned_anchors = [self._assign_anchor(bboxes) for bboxes, _ in labels]

    batch_bbox_loss = .0
    batch_obj_loss = .0
    batch_cls_loss = .0
    batch_noobj_loss = .0
    batch_total_loss = []

    for scale_idx, preds in enumerate(fmaps):
      fmap_h, fmap_w = preds.shape[-2:]
      stride_h = self.img_h / fmap_h
      stride_w = self.img_w / fmap_w
      # print(f'fmap_size {fmap_h}x{fmap_w}')

      targets = torch.zeros(batch_size, fmap_h, fmap_w, self.num_anchors, 5 + self.num_classes, device=self.device)
      bbox_loss_scale = torch.ones(batch_size, fmap_h, fmap_w, self.num_anchors, device=self.device)
      noobj_masks = []

      for img_idx, (assigned_anchors, label) in enumerate(zip(batch_assigned_anchors, labels)):
        scale_indices, ratio_indices = assigned_anchors
        bboxes, classes = label
        
        scale_mask = scale_indices == scale_idx
        ratio_indices = ratio_indices[scale_mask]
        bboxes = bboxes[scale_mask]
        classes = classes[scale_mask]
        # print(f'assigned {len(bboxes)} bboxes at {scale_idx} scale in image {img_idx}')
        
        if len(bboxes) > 0:
          bbox_scale = torch.tensor([stride_w, stride_h, stride_w, stride_h], device=self.device)
          fmap_bboxes = bboxes / bbox_scale
          for ratio_idx, bbox, cls in zip(ratio_indices, box_cxcywh(fmap_bboxes), classes):
            cx, cy, w, h = bbox.tolist()

            grid_x, grid_y = int(cx), int(cy)
            tx = cx - grid_x
            ty = cy - grid_y

            anchor_w, anchor_h = self.fmap_anchor_sizes[scale_idx][ratio_idx]
            tw = math.log(w / anchor_w)
            th = math.log(h / anchor_h)

            targets[img_idx, grid_y, grid_x, ratio_idx, :4] = torch.tensor([tx, ty, tw, th], device=self.device)  # coordinates
            targets[img_idx, grid_y, grid_x, ratio_idx, 4] = 1  # objectness
            targets[img_idx, grid_y, grid_x, ratio_idx, 5+cls] = 1  # class
            bbox_loss_scale[img_idx, grid_y, grid_x, ratio_idx] = w/fmap_w * h/fmap_h

          noobj_mask = self._get_negatives(fmap_bboxes, self.fmap_anchors[scale_idx])
          noobj_masks.append(noobj_mask.reshape(fmap_h, fmap_w, self.num_anchors))
        else:
          noobj_masks.append(torch.full(size=(fmap_h, fmap_w, self.num_anchors), fill_value=True, device=self.device))

      # (N, C, FH, FW) -> (N, FH, FW, num_anchors, 5 + num_classes)
      preds = preds.permute(0, 2, 3, 1).reshape(batch_size, fmap_h, fmap_w, self.num_anchors, -1)
      obj_mask = targets[..., 4] == 1
      noobj_mask = torch.stack(noobj_masks, dim=0)

      # if scale_idx == 0:
      #   mat = obj_mask[-1].max(dim=-1).values
      #   show_image(mat)
      #   mat = noobj_mask[-1].min(dim=-1).values
      #   show_image(mat)

      bbox_loss_scale = 2 - bbox_loss_scale
      loss_components = self._criterion(preds, targets, bbox_loss_scale, obj_mask, noobj_mask)
      
      bbox_loss, obj_loss, cls_loss, noobj_loss = loss_components
      bbox_loss = bbox_loss / batch_size
      obj_loss = obj_loss / batch_size
      cls_loss = cls_loss / batch_size
      noobj_loss = noobj_loss / batch_size
      total_loss = bbox_loss + obj_loss + cls_loss + noobj_loss

      batch_bbox_loss += bbox_loss.item()
      batch_obj_loss += obj_loss.item()
      batch_cls_loss += cls_loss.item()
      batch_noobj_loss += noobj_loss.item()
      batch_total_loss.append(total_loss)

    return sum(batch_total_loss), batch_bbox_loss, batch_obj_loss, batch_cls_loss, batch_noobj_loss

  def _get_negatives(self, bboxes, anchors):
    """
    Args:
      bboxes (Tensor): (num_bboxes, 4)
      anchors (Tensor): (FH x FW x num_anchors, 4)
    Returns:
      noobj_mask (Tensor): (FH x FW x num_anchors,)
    """
    ious = iou_matrix(anchors, bboxes)  # (FH x FW x num_anchors, num_bboxes)
    ious = ious.max(dim=1).values  # (FH x FW x num_anchors,)
    noobj_mask = ious < self.ignore_threshold
    # print(f'len(anchors) {len(anchors)}, len(bboxes) {len(bboxes)}, len(noobj) {noobj_mask.sum()}')
    return noobj_mask

  def _criterion(self, preds: Tensor, targets: Tensor, bbox_loss_scale: Tensor, obj_mask: Tensor, noobj_mask: Tensor):
    """
    Args:
      preds (Tensor): (N, FH, FW, num_anchors, 4+1+num_classes)
      targets (Tensor): (N, FH, FW, num_anchors, 4+1+num_classes)
      bbox_loss_scale (Tensor): (N, FH, FW, num_anchors)
      obj_mask: (Tensor): (N, FH, FW, num_anchors)
      noobj_mask: (Tensor): (N, FH, FW, num_anchors)
    """
    # (N, FH, FW, num_anchors, 2) -> (num_objects, 2)
    xy_preds = torch.sigmoid(preds[..., 0:2][obj_mask])
    xy_targets = targets[..., 0:2][obj_mask]

    wh_preds = preds[..., 2:4][obj_mask]
    wh_targets = targets[..., 2:4][obj_mask]

    xy_loss = F.mse_loss(xy_preds, xy_targets, reduction='none')
    wh_loss = F.mse_loss(wh_preds, wh_targets, reduction='none')
    bbox_loss = (xy_loss + wh_loss).sum(dim=-1) * bbox_loss_scale[obj_mask]
    bbox_loss = bbox_loss.sum()

    # (N, FH, FW, num_anchors) -> (num_objects,)
    obj_preds = preds[..., 4][obj_mask]
    obj_targets = targets[..., 4][obj_mask]
    obj_loss = F.binary_cross_entropy_with_logits(obj_preds, obj_targets, reduction='sum')

    # (N, FH, FW, num_anchors, num_classes) -> (num_objects, num_classes)
    cls_preds = preds[..., 5:][obj_mask]
    cls_targets = targets[..., 5:][obj_mask]
    cls_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_targets, reduction='sum')

    noobj_preds = preds[..., 4][noobj_mask]
    noobj_targets = targets[..., 4][noobj_mask]
    noobj_loss = F.binary_cross_entropy_with_logits(noobj_preds, noobj_targets, reduction='sum')

    bbox_loss *= self.bbox_loss_weight
    obj_loss *= self.obj_loss_weight
    cls_loss *= self.cls_loss_weight
    noobj_loss *= self.noobj_loss_weight
    return bbox_loss, obj_loss, cls_loss, noobj_loss


if __name__ == '__main__':
  from utils import get_device
  from yolo_model import YOLOv3
  from yolo_data import voc_loader
  
  device = get_device()
  model = YOLOv3().to(device)

  _, val_loader = voc_loader('data/VOCdevkit', batch_size=16)
  val_loader = iter(val_loader)
  x, y = next(val_loader)
  x = x.to(device)
  y = [(bboxes.to(device), classes.to(device)) for (bboxes, classes) in y]

  y_hat = model(x)
  for preds in y_hat:
    print(f'shape {preds.shape}')

  criterion = YOLOv3Loss(device)
  loss, *loss_components = criterion(y_hat, y)
  print(f'loss {loss:.4f}')
