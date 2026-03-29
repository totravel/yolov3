import torch
from torch import Tensor
from utils import box_xyxy, box_cxcywh, iou_matrix
from utils import show_boxes


class YOLOv3Prior:
  def __init__(self, device=torch.device('cpu')):
    self.device = device

    self.img_h, self.img_w = 416, 416
    self.num_fmaps = 3
    self.num_anchors = 3

    self.fmap_sizes = [[52, 52], [26, 26], [13, 13]]

    # anchor sizes for 416x416 images
    self.anchor_sizes = torch.tensor([
      [10, 13], [16, 30], [33, 23],
      [30, 61], [62, 45], [59, 119],
      [116, 90], [156, 198], [373, 326]
    ]).to(self.device)

    # anchors for 416x416 images
    self.size_only_anchors = torch.zeros(len(self.anchor_sizes), 4, device=self.device)
    self.size_only_anchors[:, 2:] = self.anchor_sizes
    self.size_only_anchors = box_xyxy(self.size_only_anchors)

    # anchors and anchor sizes for each fmaps
    self.fmap_anchors, self.fmap_anchor_sizes = [], []
    anchor_sizes = self.anchor_sizes.reshape(self.num_fmaps, self.num_anchors, 2)

    for scale_idx, (fmap_h, fmap_w) in enumerate(self.fmap_sizes):
      stride_h = self.img_h / fmap_h
      stride_w = self.img_w / fmap_w

      anchor_size_scale = torch.tensor([stride_w, stride_h], device=self.device)
      fmap_anchor_sizes = anchor_sizes[scale_idx] / anchor_size_scale

      fmap_anchors = self._generate_anchors(fmap_h, fmap_w, fmap_anchor_sizes)

      self.fmap_anchors.append(box_xyxy(fmap_anchors).reshape(-1, 4))  # (fmap_h*fmap_w*num_anchors, 4)
      self.fmap_anchor_sizes.append(fmap_anchor_sizes)

  def _generate_anchors(self, fmap_h, fmap_w, fmap_anchor_sizes: Tensor):
    """
    Args:
      fmap_h (int): fmap height
      fmap_w (int): fmap width
      fmap_anchor_sizes (Tensor): fmap anchor sizes
    Returns:
      anchors (Tensor): tensor of shape (FH, FW, num_anchors, 4)
    """
    grid_y = torch.arange(fmap_h, device=self.device)
    grid_x = torch.arange(fmap_w, device=self.device)
    shift_y, shift_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    anchor_xy = torch.stack([shift_x, shift_y], dim=-1)  # (FH, FW, 2)
    # (FH, FW, num_anchors, 2)
    anchor_xy = anchor_xy.reshape(fmap_h, fmap_w, 1, 2).repeat(1, 1, self.num_anchors, 1)

    # (FH, FW, num_anchors, 2)
    anchor_wh = fmap_anchor_sizes.reshape(1, 1, self.num_anchors, 2).repeat(fmap_h, fmap_w, 1, 1)
    
    # (FH, FW, num_anchors, 4)
    anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)
    # show_boxes(torch.zeros(fmap_h, fmap_w, 3), box_xyxy(anchors[fmap_h//2, fmap_w//2]))
    return anchors

  def _assign_anchor(self, bboxes: Tensor):
    # print(f'found {len(bboxes)} bboxes')
    # show_boxes(torch.zeros(self.img_h, self.img_w, 3), bboxes)
    size_only_bboxes = box_cxcywh(bboxes)
    size_only_bboxes[:, :2] = 0
    size_only_bboxes = box_xyxy(size_only_bboxes)

    ious = iou_matrix(self.size_only_anchors, size_only_bboxes)
    best_anchor_indices = ious.argmax(dim=0)
    
    scale_indices = best_anchor_indices // self.num_anchors
    ratio_indices = best_anchor_indices % self.num_anchors
    return scale_indices, ratio_indices
